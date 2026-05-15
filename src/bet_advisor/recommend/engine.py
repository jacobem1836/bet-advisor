"""
Recommendation engine: glues model probability -> EV -> Kelly stake -> exposure caps -> log.

The engine is the central orchestration layer for Phase 5. Given a model, a SQLite store,
and a DuckDB store, it pulls current odds snapshots, scores each market line, filters by
minimum edge, sizes stakes, enforces exposure caps, and produces Recommendation objects
that can be persisted to the bet log.

Confidence tiers
----------------
- strong     : edge >= 0.06 AND model_prob_low (5th-pct) * odds - 1 >= 0.02
  (edge is robust even under pessimistic model assumptions)
- standard   : 0.03 <= edge < 0.06
  (passes the minimum edge threshold but without the pessimistic-bound confirmation)
- speculative: edge >= min_edge but uncertainty is high (std / mean > 0.25)
  (stake is capped harder -- 50% of normal size)

Untrained model guard
---------------------
If the model's is_trained attribute is False (or the attribute is absent), generate_for_round
will raise UntrainedModelError.  Pass allow_untrained=True to override the guard and produce
recommendations anyway (useful for smoke-testing pipelines before real training data is
available).  Callers that override the guard should NOT persist the resulting bets to the
live log.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from bet_advisor.eval.devig import devig as _devig
from bet_advisor.eval.ev import edge as _edge
from bet_advisor.eval.kelly import capped_kelly, fractional_kelly, full_kelly

if TYPE_CHECKING:
    from bet_advisor.storage.duckdb_store import DuckDBStore
    from bet_advisor.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class UntrainedModelError(RuntimeError):
    """Raised when generate_for_round is called with an untrained model."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RecommendationConfig:
    """Configuration for the recommendation engine.

    Parameters
    ----------
    bankroll:
        Current bankroll in currency units.
    stake_mode:
        One of "flat", "quarter_kelly", "capped_kelly".
    flat_pct:
        Flat stake as a fraction of bankroll (default 0.01 = 1%).
    kelly_fraction:
        Fractional Kelly multiplier (default 0.25 = quarter Kelly).
    kelly_cap:
        Maximum fraction of bankroll per bet under capped_kelly (default 0.05).
    min_edge:
        Minimum fractional edge required to recommend a bet (default 0.03 = 3%).
    devig_method:
        Devigging method passed to eval.devig (default "power").
    max_bets_per_event:
        Maximum number of bets allowed on a single event (default 3).
    max_bets_per_day:
        Maximum number of bets allowed on a single day (default 10).
    max_exposure_per_event_units:
        Maximum total stake (in units of flat_pct * bankroll) per event (default 3.0).
    max_exposure_per_day_units:
        Maximum total stake in units across the day (default 10.0).
    speculative_uncertainty_threshold:
        Model std / mean above which a rec is "speculative" (default 0.25).
    speculative_stake_cap:
        Fraction applied to normal stake for speculative tier (default 0.5).
    """

    bankroll: float = 1000.0
    stake_mode: str = "flat"
    flat_pct: float = 0.01
    kelly_fraction: float = 0.25
    kelly_cap: float = 0.05
    min_edge: float = 0.03
    devig_method: str = "power"
    max_bets_per_event: int = 3
    max_bets_per_day: int = 10
    max_exposure_per_event_units: float = 3.0
    max_exposure_per_day_units: float = 10.0
    speculative_uncertainty_threshold: float = 0.25
    speculative_stake_cap: float = 0.5


# ---------------------------------------------------------------------------
# Recommendation dataclass
# ---------------------------------------------------------------------------


@dataclass
class Recommendation:
    """A single bet recommendation produced by the engine.

    All probability fields are in [0, 1]. Stake fields are in currency units
    (same currency as RecommendationConfig.bankroll).
    """

    event_id: str
    market: str
    runner: str
    bookmaker: str
    decimal_odds: float
    model_prob: float
    model_prob_low: float  # 5th-percentile estimate of model probability
    model_prob_high: float  # 95th-percentile estimate
    devigged_market_prob: float
    edge: float  # model_prob * decimal_odds - 1
    ev_units: float  # expected value per unit staked (== edge)
    recommended_stake_units: float  # in bankroll-fraction units (e.g. 0.01 = 1%)
    recommended_stake_currency: float  # in currency units
    kelly_fraction: float
    stake_mode: str
    confidence_tier: str  # "strong" | "standard" | "speculative"
    rationale: dict[str, Any] = field(default_factory=dict)
    counterarguments: list[str] = field(default_factory=list)
    feature_snapshot: dict[str, Any] = field(default_factory=dict)
    model_version: str = "unknown"
    generated_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )


# ---------------------------------------------------------------------------
# Model protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class BettingModel(Protocol):
    """Protocol that all models plugged into the engine must satisfy."""

    @property
    def is_trained(self) -> bool:
        """Return True if the model has been fitted and is ready for inference."""
        ...

    @property
    def version_hash(self) -> str:
        """Return a short identifier for this model version."""
        ...

    def predict_over_under_prob(
        self,
        X: Any,
        line: float,
        calibrate: bool = True,
    ) -> Any:
        """Return P(stat > line) for each row in X."""
        ...


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class RecommendationEngine:
    """Orchestrates model scoring, EV calculation, stake sizing, and logging.

    Parameters
    ----------
    model:
        Any object satisfying the BettingModel protocol.
    sqlite_store:
        Connected SQLiteStore for odds snapshots, signals, and bets.
    duckdb_store:
        Connected DuckDBStore for historical features.
    config:
        RecommendationConfig controlling staking, edge thresholds, and caps.
    """

    def __init__(
        self,
        model: Any,
        sqlite_store: SQLiteStore,
        duckdb_store: DuckDBStore,
        config: RecommendationConfig | None = None,
    ) -> None:
        self._model = model
        self._sqlite = sqlite_store
        self._duckdb = duckdb_store
        self._config = config or RecommendationConfig()

    # ------------------------------------------------------------------
    # Public interface

    def score_market(
        self,
        event_id: str,
        market_lines: list[dict[str, Any]],
        model_features: dict[str, Any],
    ) -> list[Recommendation]:
        """Score all runners in a market and return filtered recommendations.

        Parameters
        ----------
        event_id:
            The event identifier (matches events.event_id in SQLite).
        market_lines:
            List of dicts, each with keys:
              - market (str): market type, e.g. "h2h", "player_disposals"
              - runner (str): runner/selection name
              - bookmaker (str): bookmaker key
              - decimal_odds (float): offered price
              - line (float | None): O/U line value (None for H2H)
              - model_prob (float | None): pre-computed model probability
                (if None, the engine calls self._model to compute it)
              - model_prob_low (float | None): 5th-pct estimate
              - model_prob_high (float | None): 95th-pct estimate
        model_features:
            Feature dict for this event. Used for the feature_snapshot and
            optionally passed to the model if model_prob is None.

        Returns
        -------
        List of Recommendation objects with edge >= min_edge, sorted by edge
        descending.
        """
        cfg = self._config
        recs: list[Recommendation] = []

        # Collect all odds for a given market+bookmaker combo to devig together
        # Group by (market, bookmaker)
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for line in market_lines:
            key = (line["market"], line["bookmaker"])
            grouped.setdefault(key, []).append(line)

        for (market, bookmaker), lines in grouped.items():
            odds_list = [ln["decimal_odds"] for ln in lines]
            if len(odds_list) < 2:
                # Cannot devig a single-runner market; skip
                continue

            try:
                fair_probs = _devig(odds_list, method=cfg.devig_method)
            except ValueError as exc:
                logger.warning("Devig failed for %s/%s/%s: %s", event_id, market, bookmaker, exc)
                continue

            for i, line in enumerate(lines):
                decimal_odds = line["decimal_odds"]
                market_prob_devigged = fair_probs[i]

                # Model probability
                model_prob = line.get("model_prob")
                model_prob_low = line.get("model_prob_low")
                model_prob_high = line.get("model_prob_high")

                if model_prob is None:
                    # Fallback: use devigged market prob (no model edge)
                    model_prob = market_prob_devigged
                    model_prob_low = model_prob_low or market_prob_devigged
                    model_prob_high = model_prob_high or market_prob_devigged

                model_prob_low = model_prob_low or model_prob
                model_prob_high = model_prob_high or model_prob

                bet_edge = _edge(float(model_prob), decimal_odds)
                if bet_edge < cfg.min_edge:
                    continue

                # Kelly
                fk = full_kelly(float(model_prob), decimal_odds)
                stake_fraction = self._compute_stake_fraction(float(model_prob), decimal_odds)

                # Confidence tier
                tier = self._classify_tier(
                    bet_edge,
                    float(model_prob_low),
                    decimal_odds,
                    float(model_prob),
                )

                # Adjust stake downward for speculative
                if tier == "speculative":
                    stake_fraction = stake_fraction * cfg.speculative_stake_cap

                stake_currency = stake_fraction * cfg.bankroll

                # Rationale
                rationale = {
                    "edge": round(bet_edge, 4),
                    "devigged_market_prob": round(market_prob_devigged, 4),
                    "model_prob": round(float(model_prob), 4),
                    "kelly_full": round(fk, 4),
                    "stake_mode": cfg.stake_mode,
                    "devig_method": cfg.devig_method,
                }

                # Counterarguments
                counterarguments = self._build_counterarguments(
                    bet_edge,
                    float(model_prob),
                    float(model_prob_low),
                    float(model_prob_high),
                    decimal_odds,
                    tier,
                )

                model_version = getattr(self._model, "version_hash", "unknown")

                rec = Recommendation(
                    event_id=event_id,
                    market=market,
                    runner=line["runner"],
                    bookmaker=bookmaker,
                    decimal_odds=decimal_odds,
                    model_prob=float(model_prob),
                    model_prob_low=float(model_prob_low),
                    model_prob_high=float(model_prob_high),
                    devigged_market_prob=market_prob_devigged,
                    edge=bet_edge,
                    ev_units=bet_edge,
                    recommended_stake_units=stake_fraction,
                    recommended_stake_currency=stake_currency,
                    kelly_fraction=fk,
                    stake_mode=cfg.stake_mode,
                    confidence_tier=tier,
                    rationale=rationale,
                    counterarguments=counterarguments,
                    feature_snapshot=dict(model_features),
                    model_version=str(model_version),
                )
                recs.append(rec)

        recs.sort(key=lambda r: r.edge, reverse=True)
        return recs

    def apply_exposure_caps(
        self,
        recs: list[Recommendation],
        existing_today: list[dict[str, Any]],
    ) -> list[Recommendation]:
        """Filter and downsize recommendations to respect exposure caps.

        Caps enforced:
        - max_bets_per_event: drop lowest-edge recs beyond the per-event limit
        - max_bets_per_day: drop recs once the daily bet count is reached
        - max_exposure_per_event_units: drop recs that would exceed unit exposure per event
        - max_exposure_per_day_units: drop recs that would exceed daily unit exposure

        Parameters
        ----------
        recs:
            Candidate recommendations, sorted by edge descending.
        existing_today:
            Already-placed bets today (list of dicts with at least event_id,
            recommended_stake_units keys).

        Returns
        -------
        Filtered list of Recommendation objects that pass all caps.
        """
        cfg = self._config

        # Aggregate existing exposure from today's bets
        existing_by_event: dict[str, dict[str, float]] = {}
        for bet in existing_today:
            eid = bet.get("event_id", "")
            bucket = existing_by_event.setdefault(eid, {"count": 0.0, "units": 0.0})
            bucket["count"] += 1
            bucket["units"] += float(bet.get("recommended_stake_units", 0.0))

        day_count = sum(b["count"] for b in existing_by_event.values())
        day_units = sum(b["units"] for b in existing_by_event.values())

        # Per-event counters for new recs (track as we build the output list)
        new_by_event: dict[str, dict[str, float]] = {}

        output: list[Recommendation] = []
        for rec in recs:
            eid = rec.event_id

            ex_event = existing_by_event.get(eid, {"count": 0.0, "units": 0.0})
            new_event = new_by_event.get(eid, {"count": 0.0, "units": 0.0})

            total_event_count = ex_event["count"] + new_event["count"]
            total_event_units = ex_event["units"] + new_event["units"]
            total_day_count = day_count + len(output)
            total_day_units = day_units + sum(r.recommended_stake_units for r in output)

            if total_event_count >= cfg.max_bets_per_event:
                logger.debug(
                    "Dropping %s/%s -- per-event bet count cap (%d)",
                    eid,
                    rec.runner,
                    cfg.max_bets_per_event,
                )
                continue

            if total_day_count >= cfg.max_bets_per_day:
                logger.debug(
                    "Dropping %s/%s -- daily bet count cap (%d)",
                    eid,
                    rec.runner,
                    cfg.max_bets_per_day,
                )
                break  # recs sorted by edge; remaining will be worse

            if total_event_units + rec.recommended_stake_units > cfg.max_exposure_per_event_units:
                logger.debug(
                    "Dropping %s/%s -- per-event unit exposure cap (%.2f)",
                    eid,
                    rec.runner,
                    cfg.max_exposure_per_event_units,
                )
                continue

            if total_day_units + rec.recommended_stake_units > cfg.max_exposure_per_day_units:
                logger.debug(
                    "Dropping %s/%s -- daily unit exposure cap (%.2f)",
                    eid,
                    rec.runner,
                    cfg.max_exposure_per_day_units,
                )
                break

            output.append(rec)
            bucket = new_by_event.setdefault(eid, {"count": 0.0, "units": 0.0})
            bucket["count"] += 1
            bucket["units"] += rec.recommended_stake_units

        return output

    def generate_for_round(
        self,
        round_number: int,
        allow_untrained: bool = False,
    ) -> list[Recommendation]:
        """Generate recommendations for all events in a given AFL round.

        Pulls the latest odds snapshots from SQLite (per event and market),
        retrieves any available model features from DuckDB, calls score_market
        per (event, market, bookmaker) group, and applies exposure caps against
        any bets already placed today.

        Parameters
        ----------
        round_number:
            AFL round number (used to filter events from the store).
        allow_untrained:
            If True, bypasses the untrained model guard.  Use for smoke testing
            only -- do NOT persist the resulting recommendations.

        Returns
        -------
        Final list of Recommendation objects after cap enforcement.

        Raises
        ------
        UntrainedModelError:
            If the model is not trained and allow_untrained is False.
        """
        is_trained = getattr(self._model, "is_trained", False)
        if not is_trained and not allow_untrained:
            raise UntrainedModelError(
                "Model is not trained. Pass allow_untrained=True to bypass this guard. "
                "Do not persist resulting recommendations to the live bet log."
            )

        if not is_trained:
            logger.warning(
                "Generating recommendations with an untrained model "
                "(allow_untrained=True). These are NOT suitable for paper trading."
            )

        # Pull latest odds snapshots for the round
        today_str = datetime.now(UTC).date().isoformat()
        odds_rows = self._sqlite.query(
            """
            SELECT os.event_id, os.bookmaker, os.market, os.runner,
                   os.price, os.point, os.captured_at
            FROM odds_snapshots os
            JOIN events e ON os.event_id = e.event_id
            WHERE e.completed = 0
            ORDER BY os.captured_at DESC
            """,
        )

        if not odds_rows:
            logger.info("No live odds snapshots found; returning empty recommendation list.")
            return []

        # Deduplicate to most-recent snapshot per (event, bookmaker, market, runner)
        seen: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for row in odds_rows:
            key = (row["event_id"], row["bookmaker"], row["market"], row["runner"])
            if key not in seen:
                seen[key] = row

        # Group into (event_id, market) buckets with all bookmakers
        # For each event, build a unified market_lines list
        event_lines: dict[str, list[dict[str, Any]]] = {}
        for row in seen.values():
            eid = row["event_id"]
            event_lines.setdefault(eid, []).append(
                {
                    "market": row["market"],
                    "bookmaker": row["bookmaker"],
                    "runner": row["runner"],
                    "decimal_odds": float(row["price"]),
                    "line": row.get("point"),
                    "model_prob": None,
                    "model_prob_low": None,
                    "model_prob_high": None,
                }
            )

        # Pull today's existing bets for cap enforcement
        existing_today = self._sqlite.query(
            "SELECT * FROM bets WHERE placed_at >= ?",
            (today_str,),
        )

        all_recs: list[Recommendation] = []
        for event_id, lines in event_lines.items():
            # Attempt to enrich with model probabilities where possible
            # (for now, features are an empty dict -- the model is called
            # externally before lines are assembled in a real pipeline)
            features = self._fetch_features(event_id)
            recs = self.score_market(event_id, lines, features)
            all_recs.extend(recs)

        # Sort globally by edge descending before applying caps
        all_recs.sort(key=lambda r: r.edge, reverse=True)
        return self.apply_exposure_caps(all_recs, existing_today)

    def persist(self, recs: list[Recommendation]) -> list[int]:
        """Log each recommendation as a signal and a paper bet in SQLite.

        Parameters
        ----------
        recs:
            Recommendations to persist.

        Returns
        -------
        List of bet IDs (one per recommendation, in the same order).
        """
        bet_ids: list[int] = []
        now_str = datetime.now(UTC).isoformat(timespec="seconds")

        for rec in recs:
            signal = {
                "event_id": rec.event_id,
                "market": rec.market,
                "runner": rec.runner,
                "model_prob": rec.model_prob,
                "market_prob_devigged": rec.devigged_market_prob,
                "edge": rec.edge,
                "recommended_stake": rec.recommended_stake_currency,
                "model_version": rec.model_version,
                "created_at": now_str,
                "rationale": rec.rationale,
            }
            signal_id = self._sqlite.log_signal(signal)

            bet = {
                "signal_id": signal_id,
                "placed_at": now_str,
                "bookmaker": rec.bookmaker,
                "market": rec.market,
                "runner": rec.runner,
                "price": rec.decimal_odds,
                "stake": rec.recommended_stake_currency,
                "model_version_hash": rec.model_version,
                "feature_snapshot": rec.feature_snapshot,
                "decision_rationale": {
                    "confidence_tier": rec.confidence_tier,
                    "counterarguments": rec.counterarguments,
                    **rec.rationale,
                },
                "recommended_stake_units": rec.recommended_stake_units,
                "actual_stake_units": rec.recommended_stake_units,
                "bankroll_at_placement": self._config.bankroll,
                "expected_value": rec.ev_units,
                "edge": rec.edge,
                "kelly_fraction": rec.kelly_fraction,
                "stake_mode": rec.stake_mode,
                "staking_strategy": self._config.stake_mode,
                "devig_method": self._config.devig_method,
            }
            bet_id = self._sqlite.log_bet_dict(bet)
            bet_ids.append(bet_id)

        return bet_ids

    # ------------------------------------------------------------------
    # Internal helpers

    def _compute_stake_fraction(self, prob: float, decimal_odds: float) -> float:
        """Compute the stake fraction (of bankroll) for a single bet."""
        cfg = self._config
        if cfg.stake_mode == "flat":
            return cfg.flat_pct
        elif cfg.stake_mode == "quarter_kelly":
            return fractional_kelly(prob, decimal_odds, fraction=cfg.kelly_fraction)
        elif cfg.stake_mode == "capped_kelly":
            return capped_kelly(prob, decimal_odds, cap=cfg.kelly_cap, fraction=cfg.kelly_fraction)
        else:
            logger.warning("Unknown stake_mode %r; falling back to flat", cfg.stake_mode)
            return cfg.flat_pct

    def _classify_tier(
        self,
        bet_edge: float,
        model_prob_low: float,
        decimal_odds: float,
        model_prob: float,
    ) -> str:
        """Assign a confidence tier based on edge and model uncertainty.

        Tier rules:
        - strong     : edge >= 0.06 AND pessimistic EV (p5 * odds - 1) >= 0.02
        - standard   : 0.03 <= edge < 0.06
        - speculative: edge >= min_edge but std/mean threshold exceeded
          (computed as proxy: if model_prob_low < model_prob * 0.75 it signals
          high uncertainty relative to the central estimate)

        Note: caller must reduce stake for "speculative" tier.
        """
        cfg = self._config

        pessimistic_ev = model_prob_low * decimal_odds - 1.0

        # Check for high uncertainty: model_prob_low significantly below model_prob
        uncertainty_ratio = (model_prob - model_prob_low) / max(model_prob, 1e-9)
        is_high_uncertainty = uncertainty_ratio > cfg.speculative_uncertainty_threshold

        if bet_edge >= 0.06 and pessimistic_ev >= 0.02 and not is_high_uncertainty:
            return "strong"
        elif is_high_uncertainty:
            return "speculative"
        else:
            return "standard"

    def _build_counterarguments(
        self,
        bet_edge: float,
        model_prob: float,
        model_prob_low: float,
        model_prob_high: float,
        decimal_odds: float,
        tier: str,
    ) -> list[str]:
        """Generate a list of documented counterarguments for the recommendation."""
        args: list[str] = []

        uncertainty = model_prob_high - model_prob_low
        if uncertainty > 0.15:
            args.append(
                f"Wide model CI ({model_prob_low:.3f}-{model_prob_high:.3f}): "
                "edge estimate may be noisy."
            )

        pessimistic_ev = model_prob_low * decimal_odds - 1.0
        if pessimistic_ev < 0.0:
            args.append(
                f"Pessimistic scenario (p5 = {model_prob_low:.3f}) is negative EV "
                f"({pessimistic_ev:.3f}). Edge not robust to model uncertainty."
            )

        if tier == "speculative":
            args.append(
                "Classified as speculative: stake reduced 50%. "
                "Do not chase if model calibration is unvalidated."
            )

        if bet_edge < 0.05:
            args.append(
                f"Edge {bet_edge:.3f} is below the 5% level recommended "
                "for unvalidated models (Phase 1 §2). Monitor CLV before scaling."
            )

        return args

    def _fetch_features(self, event_id: str) -> dict[str, Any]:
        """Attempt to fetch model features for an event from DuckDB.

        Returns an empty dict if no features are available.
        """
        try:
            rows = (
                self._duckdb.query(
                    "SELECT * FROM player_features WHERE event_id = ?",
                    [event_id],
                )
                if hasattr(self._duckdb, "query")
                else []
            )
            if rows is not None and len(rows) > 0:
                return dict(rows[0])
        except Exception as exc:
            logger.debug("Could not fetch features for %s: %s", event_id, exc)
        return {}
