"""
CLV reference adapter: resolves a fair closing probability for CLV computation.

The "reference close" is the benchmark against which each bet's closing line
value is measured. This module provides a configurable adapter that supports
four modes:

- multi_book_consensus  (default): devig each AU book independently, then
  average the resulting no-vig probabilities across books, optionally
  weighted by overround (sharper books weighted more).
- betfair_delayed: use the Betfair Exchange delayed-key closing price when
  sufficient volume is matched; fall back to consensus otherwise.
- sportsbet_only: devig the Sportsbet closing line only.
- single_book: devig any one configured book's closing line.

See research/06_betfair_alternatives.md for the rationale behind this design
and the accuracy trade-offs vs a live Betfair Exchange reference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from bet_advisor.eval.devig import devig as _devig
from bet_advisor.eval.devig import overround as _overround

logger = logging.getLogger(__name__)

# Default AU bookmakers available via The Odds API.
_DEFAULT_CONSENSUS_BOOKS: list[str] = [
    "sportsbet",
    "tab",
    "ladbrokes",
    "pointsbet",
    "betr",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ClvReferenceConfig:
    """Configuration for the CLV reference resolver.

    Parameters
    ----------
    mode:
        Resolution strategy. One of:
        - ``"multi_book_consensus"`` (default): devig each book independently,
          then average no-vig runner probs across books.
        - ``"betfair_delayed"``: use Betfair Exchange delayed-key closing price
          when matched volume meets the threshold; otherwise fall back to
          consensus.
        - ``"sportsbet_only"``: devig Sportsbet closing odds only.
        - ``"single_book"``: devig the book named in ``single_book``.
    devig_method:
        Devigging method applied per-book. Passed to ``eval.devig.devig``.
        Defaults to ``"power"`` (recommended for two-runner AFL markets).
    consensus_books:
        List of bookmaker keys to include in consensus mode. Defaults to the
        five main AU books available via The Odds API.
    consensus_min_books:
        Minimum number of books required to compute a full consensus. If fewer
        books have data, the resolver uses whatever is available and emits a
        warning. Default 3.
    consensus_weighting:
        How to weight books in the consensus average.
        - ``"equal"``: arithmetic mean across books.
        - ``"by_overround"`` (default): weight = 1 / (1 + overround_i),
          normalised so weights sum to 1. Lower overround = sharper book =
          higher weight.
    betfair_delayed_min_volume:
        Minimum AUD matched on the Betfair market for the delayed-key price to
        be used. Below this threshold the resolver falls back to consensus and
        appends a warning. Default 1000.
    single_book:
        Book key to use when ``mode="single_book"``. Must be set if
        ``mode="single_book"`` is used.
    """

    mode: Literal[
        "multi_book_consensus",
        "betfair_delayed",
        "sportsbet_only",
        "single_book",
    ] = "multi_book_consensus"
    devig_method: str = "power"
    consensus_books: list[str] = field(default_factory=lambda: list(_DEFAULT_CONSENSUS_BOOKS))
    consensus_min_books: int = 3
    consensus_weighting: Literal["equal", "by_overround"] = "by_overround"
    betfair_delayed_min_volume: float = 1000.0
    single_book: str | None = None


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class ClvReferenceResolver:
    """Resolve a reference closing probability per runner for CLV measurement.

    Parameters
    ----------
    config:
        ClvReferenceConfig controlling mode and tuning parameters.
    """

    def __init__(self, config: ClvReferenceConfig | None = None) -> None:
        self._config = config or ClvReferenceConfig()

    # ------------------------------------------------------------------
    # Public interface

    def resolve(self, event_market_close: dict[str, Any]) -> dict[str, Any]:
        """Resolve reference closing probabilities for all runners in a market.

        Parameters
        ----------
        event_market_close:
            Snapshot dict with the following structure::

                {
                    "runners": [
                        {
                            "name": "Richmond",
                            "books": {
                                "sportsbet": 1.80,
                                "tab": 1.85,
                                "ladbrokes": 1.78,
                                "pointsbet": 1.82,
                                "betr": 1.83
                            },
                            "betfair_delayed": {   # optional
                                "price": 1.83,
                                "volume_matched": 2500.0
                            }
                        },
                        ...
                    ]
                }

        Returns
        -------
        dict with keys:
            - ``runners``: list of ``{"name": str, "ref_prob": float,
              "ref_source": str}`` -- one entry per runner.
            - ``method``: the resolution mode that was applied.
            - ``books_used``: list of book keys that contributed to the result.
            - ``warnings``: list of warning strings emitted during resolution.

        Raises
        ------
        ValueError
            If ``event_market_close`` is malformed, empty, or no probability
            can be resolved for any runner.
        """
        self._validate_input(event_market_close)
        cfg = self._config

        if cfg.mode == "multi_book_consensus":
            return self._resolve_consensus(event_market_close)
        elif cfg.mode == "betfair_delayed":
            return self._resolve_betfair(event_market_close)
        elif cfg.mode == "sportsbet_only":
            return self._resolve_sportsbet_only(event_market_close)
        elif cfg.mode == "single_book":
            if not cfg.single_book:
                raise ValueError(
                    "ClvReferenceConfig.single_book must be set when mode='single_book'"
                )
            return self._resolve_single_book(event_market_close, cfg.single_book)
        else:
            raise ValueError(f"Unknown CLV reference mode: {cfg.mode!r}")

    # ------------------------------------------------------------------
    # Resolution strategies

    def _resolve_consensus(
        self,
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        """Multi-book consensus: devig each book, then average runner probs.

        Weighting follows ``config.consensus_weighting``:
        - equal: arithmetic mean.
        - by_overround: weight_i = 1 / (1 + overround_i), normalised.
          A book with 4% overround receives more weight than one with 10%.
        """
        cfg = self._config
        runners = snapshot["runners"]
        warnings: list[str] = []

        # Gather per-book odds for all runners in the market.
        # Structure: {book: [odds_runner0, odds_runner1, ...]}
        book_odds: dict[str, list[float]] = {}
        for book in cfg.consensus_books:
            market_odds: list[float] = []
            for r in runners:
                price = r.get("books", {}).get(book)
                if price is not None:
                    market_odds.append(float(price))
            if len(market_odds) == len(runners):
                # Only include books where all runners have a price.
                book_odds[book] = market_odds

        n_books = len(book_odds)
        if n_books == 0:
            raise ValueError(
                "No configured consensus book has prices for all runners in this market."
            )
        if n_books < cfg.consensus_min_books:
            warnings.append(
                f"Only {n_books} book(s) available; minimum configured is "
                f"{cfg.consensus_min_books}. Using available books: {list(book_odds)}."
            )
            logger.warning(
                "CLV consensus: %d book(s) available, minimum %d configured.",
                n_books,
                cfg.consensus_min_books,
            )

        # Devig each book and compute overround for weighting.
        book_probs: dict[str, list[float]] = {}
        book_overrounds: dict[str, float] = {}
        for book, odds in book_odds.items():
            try:
                fair = _devig(odds, method=cfg.devig_method)
                book_probs[book] = fair
                book_overrounds[book] = _overround(odds)
            except ValueError as exc:
                warnings.append(f"Devig failed for book {book!r}: {exc}. Skipping.")
                logger.warning("Devig failed for book %r: %s", book, exc)

        if not book_probs:
            raise ValueError("All books failed devigging; cannot compute consensus.")

        # Compute weights.
        books_used = list(book_probs)
        if cfg.consensus_weighting == "by_overround":
            raw_weights = [1.0 / (1.0 + book_overrounds[b]) for b in books_used]
            total_w = sum(raw_weights)
            weights = [w / total_w for w in raw_weights]
        else:
            # Equal weighting.
            n = len(books_used)
            weights = [1.0 / n] * n

        # Weighted average of fair probabilities per runner.
        n_runners = len(runners)
        consensus_probs = [0.0] * n_runners
        for w, book in zip(weights, books_used):
            for i, p in enumerate(book_probs[book]):
                consensus_probs[i] += w * p

        # Normalise to ensure sum == 1.0 (corrects floating-point drift).
        total = sum(consensus_probs)
        if total > 0:
            consensus_probs = [p / total for p in consensus_probs]

        runner_results = [
            {
                "name": runners[i]["name"],
                "ref_prob": consensus_probs[i],
                "ref_source": f"consensus:{','.join(books_used)}",
            }
            for i in range(n_runners)
        ]

        return {
            "runners": runner_results,
            "method": "multi_book_consensus",
            "books_used": books_used,
            "warnings": warnings,
        }

    def _resolve_betfair(
        self,
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        """Betfair delayed-key mode.

        Uses ``betfair_delayed.price`` per runner when all runners have a
        Betfair price and total matched volume meets
        ``config.betfair_delayed_min_volume``. Falls back to consensus
        otherwise, with a warning.

        Note on the delayed key: the free Betfair delayed app key provides
        ``lastPriceTraded`` with a 1-180 second delay. It cannot place bets.
        Price quality on thin AFL player prop markets may be low; use consensus
        for those markets and Betfair for match-odds markets with sufficient
        volume.
        """
        cfg = self._config
        runners = snapshot["runners"]
        warnings: list[str] = []

        # Check all runners have a betfair_delayed entry.
        bf_prices: list[float] = []
        total_volume = 0.0
        all_have_bf = True
        for r in runners:
            bf = r.get("betfair_delayed")
            if bf is None or "price" not in bf:
                all_have_bf = False
                break
            price = float(bf["price"])
            if price <= 1.0:
                all_have_bf = False
                break
            bf_prices.append(price)
            total_volume += float(bf.get("volume_matched", 0.0))

        if not all_have_bf or total_volume < cfg.betfair_delayed_min_volume:
            reason = (
                "missing Betfair delayed prices for some runners"
                if not all_have_bf
                else f"Betfair volume {total_volume:.0f} AUD below threshold "
                f"{cfg.betfair_delayed_min_volume:.0f} AUD"
            )
            warnings.append(f"Betfair delayed mode falling back to consensus: {reason}.")
            logger.warning("CLV Betfair mode: falling back to consensus (%s)", reason)
            result = self._resolve_consensus(snapshot)
            result["warnings"] = warnings + result["warnings"]
            return result

        # Devig the Betfair closing prices.
        try:
            fair_probs = _devig(bf_prices, method=cfg.devig_method)
        except ValueError as exc:
            warnings.append(f"Betfair devig failed ({exc}); falling back to consensus.")
            logger.warning("Betfair devig failed: %s; falling back to consensus.", exc)
            result = self._resolve_consensus(snapshot)
            result["warnings"] = warnings + result["warnings"]
            return result

        runner_results = [
            {
                "name": runners[i]["name"],
                "ref_prob": fair_probs[i],
                "ref_source": "betfair_delayed",
            }
            for i in range(len(runners))
        ]

        return {
            "runners": runner_results,
            "method": "betfair_delayed",
            "books_used": ["betfair_delayed"],
            "warnings": warnings,
        }

    def _resolve_sportsbet_only(
        self,
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        """Devig only the Sportsbet closing line."""
        return self._resolve_single_book(snapshot, "sportsbet")

    def _resolve_single_book(
        self,
        snapshot: dict[str, Any],
        book: str,
    ) -> dict[str, Any]:
        """Devig a single configured book's closing line."""
        cfg = self._config
        runners = snapshot["runners"]
        warnings: list[str] = []

        odds: list[float] = []
        for r in runners:
            price = r.get("books", {}).get(book)
            if price is None:
                raise ValueError(f"Runner {r.get('name')!r} has no price for book {book!r}.")
            odds.append(float(price))

        try:
            fair_probs = _devig(odds, method=cfg.devig_method)
        except ValueError as exc:
            raise ValueError(f"Devig failed for single-book {book!r}: {exc}") from exc

        runner_results = [
            {
                "name": runners[i]["name"],
                "ref_prob": fair_probs[i],
                "ref_source": f"single_book:{book}",
            }
            for i in range(len(runners))
        ]

        return {
            "runners": runner_results,
            "method": f"single_book:{book}",
            "books_used": [book],
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Internal helpers

    @staticmethod
    def _validate_input(snapshot: dict[str, Any]) -> None:
        """Raise ValueError for empty or malformed snapshots."""
        if not isinstance(snapshot, dict):
            raise ValueError(f"event_market_close must be a dict; got {type(snapshot).__name__!r}")
        runners = snapshot.get("runners")
        if not runners:
            raise ValueError("event_market_close['runners'] must be a non-empty list")
        if not isinstance(runners, list):
            raise ValueError(
                f"event_market_close['runners'] must be a list; got {type(runners).__name__!r}"
            )
        for i, r in enumerate(runners):
            if not isinstance(r, dict):
                raise ValueError(f"runners[{i}] must be a dict; got {type(r).__name__!r}")
            if "name" not in r:
                raise ValueError(f"runners[{i}] is missing required key 'name'")
            books = r.get("books", {})
            if not isinstance(books, dict):
                raise ValueError(
                    f"runners[{i}]['books'] must be a dict; got {type(books).__name__!r}"
                )


def build_default_resolver() -> ClvReferenceResolver:
    """Return a ClvReferenceResolver with production defaults.

    Mode: multi_book_consensus
    Books: sportsbet, tab, ladbrokes, pointsbet, betr
    Devig: power
    Weighting: by_overround
    Min books: 3
    """
    return ClvReferenceResolver(ClvReferenceConfig())
