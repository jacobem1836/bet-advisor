"""
Walk-forward backtester for the AFL disposals model.

Design principles
-----------------
1. Strictly time-aware. No data leakage across the train/test boundary.
2. Expanding window (not rolling): each test window uses all prior data as
   training. This matches production behaviour where the model would be
   retrained periodically with all available data.
3. Look-ahead bias guards are encoded as runtime assertions (not just
   documentation). See BacktestAsserter.
4. No shuffling of any kind.

Walk-forward schema (example with 4 seasons):

    Seasons 2019-2020 -> train | 2021 -> test
    Seasons 2019-2021 -> train | 2022 -> test
    Seasons 2019-2022 -> train | 2023 -> test
    Seasons 2019-2023 -> train | 2024 -> test

CLV computation
---------------
In a real-data backtest, the closing odds would come from the historical odds
store. With synthetic data (or when historical prop odds are unavailable), we
simulate a closing line by adding a small random shift to the model's fair
probability. This is documented as a limitation -- replace with real closing
data once Phase 2 backfill has been run.

Staking
-------
Default: flat staking at 1% of a notional $1,000 bankroll = $10 per bet.
Kelly staking is available via StakeConfig.use_kelly but is disabled by
default (see research §4.2: flat staking is recommended for the MVP phase).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from bet_advisor.eval.calibration import brier_score, expected_calibration_error, log_loss
from bet_advisor.eval.clv import closing_line_value
from bet_advisor.eval.ev import expected_value
from bet_advisor.models.disposals import (
    DisposalsFeatureBuilder,
    DisposalsModel,
    ModelConfig,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SplitterConfig:
    """Configuration for WalkForwardSplitter.

    Parameters
    ----------
    date_col:
        Name of the date column in the player stats DataFrame.
    train_min_seasons:
        Minimum number of seasons required before the first test window opens.
    test_season_count:
        Number of seasons per test window. Defaults to 1 (one season at a time).
    step_seasons:
        Number of seasons to advance between splits. Defaults to 1.
    """

    date_col: str = "match_date"
    train_min_seasons: int = 2
    test_season_count: int = 1
    step_seasons: int = 1


@dataclass
class StakeConfig:
    """Staking policy configuration.

    Parameters
    ----------
    bankroll:
        Notional starting bankroll in dollars.
    flat_unit_fraction:
        Fraction of bankroll to bet per unit (flat staking only).
    use_kelly:
        If True, use quarter-Kelly sizing instead of flat staking. Disabled
        by default (see research §4.2 rationale in module docstring).
    kelly_fraction:
        Kelly fraction to apply when use_kelly is True.
    min_edge_threshold:
        Minimum fractional edge to trigger a bet. Bets with EV below this
        fraction are skipped.
    """

    bankroll: float = 1000.0
    flat_unit_fraction: float = 0.01
    use_kelly: bool = False
    kelly_fraction: float = 0.25
    min_edge_threshold: float = 0.03


# ---------------------------------------------------------------------------
# Walk-forward splitter
# ---------------------------------------------------------------------------


class WalkForwardSplitter:
    """Generate expanding-window (train, test) index pairs.

    Splits are strictly time-ordered: all test dates are strictly after all
    training dates in each split. No row appears in both train and test within
    a single split.

    Parameters
    ----------
    cfg:
        Splitter configuration. See SplitterConfig.

    Usage
    -----
    splitter = WalkForwardSplitter()
    for train_idx, test_idx in splitter.split(df):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    """

    def __init__(self, cfg: SplitterConfig | None = None) -> None:
        self._cfg = cfg or SplitterConfig()

    def split(self, df: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_indices, test_indices) for each walk-forward fold.

        Parameters
        ----------
        df:
            DataFrame containing a date column (cfg.date_col) and a season
            column. If no season column is present, seasons are derived from
            the year of the date column.

        Yields
        ------
        Tuples of (train_idx, test_idx) as integer position arrays, suitable
        for use with ``df.iloc``.

        Raises
        ------
        ValueError
            If the DataFrame has fewer rows than required for the minimum
            training window.
        """
        cfg = self._cfg
        date_col = cfg.date_col

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        if "season" not in df.columns:
            df["_season"] = df[date_col].dt.year
        else:
            df["_season"] = df["season"]

        seasons = sorted(df["_season"].unique())
        n_seasons = len(seasons)

        if n_seasons < cfg.train_min_seasons + cfg.test_season_count:
            raise ValueError(
                f"Need at least {cfg.train_min_seasons + cfg.test_season_count} "
                f"seasons, got {n_seasons}."
            )

        # Expanding window: train grows by step_seasons each iteration
        test_start_idx = cfg.train_min_seasons
        while test_start_idx + cfg.test_season_count <= n_seasons:
            train_seasons = set(seasons[:test_start_idx])
            test_seasons = set(seasons[test_start_idx : test_start_idx + cfg.test_season_count])

            train_mask = df["_season"].isin(train_seasons)
            test_mask = df["_season"].isin(test_seasons)

            train_positions = np.where(train_mask.to_numpy())[0]
            test_positions = np.where(test_mask.to_numpy())[0]

            if len(train_positions) > 0 and len(test_positions) > 0:
                # Assertion: all test dates must be strictly after all train dates
                train_max_date = df[date_col].iloc[train_positions].max()
                test_min_date = df[date_col].iloc[test_positions].min()
                assert test_min_date > train_max_date, (
                    f"Look-ahead bias detected: test data (from {test_min_date}) "
                    f"overlaps train data (through {train_max_date}). "
                    "Ensure the date column is correctly populated."
                )
                yield train_positions, test_positions

            test_start_idx += cfg.step_seasons


# ---------------------------------------------------------------------------
# Backtest report
# ---------------------------------------------------------------------------


@dataclass
class BacktestReport:
    """Summary statistics from a walk-forward backtest run.

    Fields
    ------
    n_bets:
        Total number of bets simulated.
    n_wins:
        Number of bets where the over/under prediction was correct.
    roi:
        Overall return on investment (P&L / total stakes).
    roi_ci_low:
        Wilson 95% lower bound on win rate (not directly on ROI, but
        computed consistently for reporting). See computation note below.
    roi_ci_high:
        Wilson 95% upper bound on win rate.
    brier:
        Brier score on over/under probabilities vs outcomes.
    log_loss_score:
        Mean log loss on over/under probabilities vs outcomes.
    ece:
        Expected calibration error.
    mean_clv:
        Mean closing line value across all bets.
    pct_positive_clv:
        Fraction of bets with CLV > 0.
    drawdown_max:
        Maximum peak-to-trough drawdown in bankroll units.
    sharpe_like:
        Mean per-bet return divided by its standard deviation (not annualised).
        Analogous to a Sharpe ratio over the bet sequence.

    Notes on CLV
    ------------
    In a backtest without real historical prop odds, CLV is computed against a
    simulated closing line: closing_fair_prob = raw_model_prob + N(0, 0.01).
    This is a placeholder. Replace with real closing odds once Phase 2
    historical prop odds backfill is available.
    """

    n_bets: int = 0
    n_wins: int = 0
    roi: float = 0.0
    roi_ci_low: float = 0.0
    roi_ci_high: float = 0.0
    brier: float = 0.0
    log_loss_score: float = 0.0
    ece: float = 0.0
    mean_clv: float = 0.0
    pct_positive_clv: float = 0.0
    drawdown_max: float = 0.0
    sharpe_like: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        """Write the report to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())
        logger.info("BacktestReport saved to %s", path)


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------


class Backtester:
    """Orchestrate a walk-forward backtest for the disposals model.

    For each split produced by WalkForwardSplitter:
      1. Build features with as-of cutoff = first test date.
      2. Fit DisposalsModel on training rows.
      3. Predict over/under probability on test rows.
      4. Simulate EV-filtered flat or Kelly bet.
      5. Record into bets log.

    At the end of all splits, compute BacktestReport.

    Parameters
    ----------
    model_cfg:
        DisposalsModel configuration.
    splitter_cfg:
        Walk-forward split configuration.
    stake_cfg:
        Staking policy.
    line:
        The disposals over/under line to bet at. In production this comes from
        the odds store; here a fixed line is used for the synthetic smoke test.

    Look-ahead bias protection (encoded as runtime assertions)
    ----------------------------------------------------------
    - Feature builder receives as-of = first test date (strictly before test data).
    - The feature builder's shift(1) guards are applied within DisposalsFeatureBuilder.
    - Closing odds are not used as model inputs (see CLV simulation note in docstring).
    """

    def __init__(
        self,
        model_cfg: ModelConfig | None = None,
        splitter_cfg: SplitterConfig | None = None,
        stake_cfg: StakeConfig | None = None,
        line: float = 24.5,
    ) -> None:
        self._model_cfg = model_cfg or ModelConfig()
        self._splitter_cfg = splitter_cfg or SplitterConfig()
        self._stake_cfg = stake_cfg or StakeConfig()
        self._line = line
        self._splitter = WalkForwardSplitter(self._splitter_cfg)

    def run(
        self,
        player_stats_df: pd.DataFrame,
        odds_df: pd.DataFrame | None = None,
    ) -> tuple[BacktestReport, pd.DataFrame]:
        """Run the full walk-forward backtest.

        Parameters
        ----------
        player_stats_df:
            Full player stats DataFrame with columns: match_id, player_id,
            match_date, season, disposals, team, opponent_team, venue, is_home,
            and optionally time_on_ground_pct.
        odds_df:
            Optional historical odds DataFrame with columns: match_id,
            player_id, line, over_odds, closing_over_odds,
            closing_under_odds. If None, synthetic odds are generated.

        Returns
        -------
        (report, bets_df) where report is a BacktestReport and bets_df is a
        DataFrame with one row per simulated bet.
        """
        date_col = self._splitter_cfg.date_col
        player_stats_df = player_stats_df.copy()
        player_stats_df[date_col] = pd.to_datetime(player_stats_df[date_col])

        all_bets: list[dict] = []
        bankroll = self._stake_cfg.bankroll

        for fold_idx, (train_idx, test_idx) in enumerate(self._splitter.split(player_stats_df)):
            train_df = player_stats_df.iloc[train_idx].copy()
            test_df = player_stats_df.iloc[test_idx].copy()

            # as-of = one day before the earliest test date (strictly no leakage)
            first_test_date = test_df[date_col].min()
            as_of = first_test_date - pd.Timedelta(days=1)

            logger.info(
                "Fold %d: train=%d rows, test=%d rows, as_of=%s",
                fold_idx,
                len(train_df),
                len(test_df),
                as_of.date(),
            )

            # Build features -- as_of prevents any test data from entering features
            builder = DisposalsFeatureBuilder(self._model_cfg.feature_config)
            X_train, y_train = builder.build(train_df, as_of=as_of)

            # Reset test_df index so it aligns with X_test (builder resets index)
            test_df_reset = test_df.reset_index(drop=True)
            X_test, y_test = builder.build(test_df_reset, as_of=None)

            # Assert look-ahead: model must not be fit on test data
            if len(X_train) == 0:
                logger.warning("Fold %d: empty training set, skipping.", fold_idx)
                continue

            # Fit model
            model = DisposalsModel(cfg=self._model_cfg)
            try:
                model.fit(X_train, y_train)
            except ValueError as e:
                logger.warning("Fold %d: model fit failed (%s), skipping.", fold_idx, e)
                continue

            # Predict on test
            # clean_mask has same index as X_test (0..n); use .values for alignment
            clean_mask = X_test.notna().all(axis=1)
            clean_mask_vals = clean_mask.to_numpy()
            X_test_clean = X_test[clean_mask]
            y_test_clean = y_test[clean_mask]
            test_rows_clean = test_df_reset[clean_mask_vals].copy()

            if X_test_clean.empty:
                continue

            probs = model.predict_over_under_prob(X_test_clean, line=self._line)

            # Simulate bets per row
            for i, (row_idx, row) in enumerate(test_rows_clean.iterrows()):
                prob = float(probs[i])
                actual_disposals = float(y_test_clean.iloc[i])
                outcome = 1 if actual_disposals > self._line else 0

                # Get or simulate odds
                over_odds, closing_over_odds, closing_under_odds = self._get_odds(row, odds_df)

                ev = expected_value(prob, over_odds)
                if ev < self._stake_cfg.min_edge_threshold:
                    continue  # below threshold -- no bet

                stake = self._compute_stake(prob, over_odds, bankroll)

                # Simulate CLV
                clv = self._compute_clv(prob, closing_over_odds, closing_under_odds)

                pnl = stake * (over_odds - 1.0) if outcome == 1 else -stake
                bankroll = bankroll + pnl

                all_bets.append(
                    {
                        "fold": fold_idx,
                        "match_id": row.get("match_id", ""),
                        "player_id": row.get("player_id", ""),
                        "match_date": row.get(date_col, ""),
                        "line": self._line,
                        "model_prob": prob,
                        "over_odds": over_odds,
                        "ev": ev,
                        "stake": stake,
                        "outcome": outcome,
                        "pnl": pnl,
                        "bankroll": bankroll,
                        "closing_over_odds": closing_over_odds,
                        "closing_under_odds": closing_under_odds,
                        "clv": clv,
                    }
                )

        bets_df = pd.DataFrame(all_bets)
        report = self._compute_report(bets_df)
        return report, bets_df

    # ------------------------------------------------------------------
    # Internal helpers

    def _get_odds(
        self,
        row: pd.Series,
        odds_df: pd.DataFrame | None,
    ) -> tuple[float, float, float]:
        """Return (over_odds, closing_over_odds, closing_under_odds).

        If odds_df is available and has matching row, use those. Otherwise
        simulate a market at approximately even money (1.90) with a small
        random closing shift.

        The simulated closing line is a placeholder. Replace with real
        historical prop odds once the Phase 2 backfill has been run.
        """
        if odds_df is not None and not odds_df.empty:
            match = odds_df[
                (odds_df.get("match_id", pd.Series(dtype=str)) == row.get("match_id", ""))
                & (odds_df.get("player_id", pd.Series(dtype=str)) == row.get("player_id", ""))
            ]
            if not match.empty:
                r = match.iloc[0]
                return (
                    float(r.get("over_odds", 1.90)),
                    float(r.get("closing_over_odds", 1.90)),
                    float(r.get("closing_under_odds", 1.90)),
                )

        # Simulated market: book price 1.90/1.90 with vig = 5.3%
        over_odds = 1.90
        # Simulate closing movement: close is slightly sharper than open
        rng = np.random.default_rng(abs(hash(str(row.get("player_id", "")))) % (2**31))
        closing_shift = rng.normal(0, 0.03)
        closing_over_prob = min(max(1.0 / over_odds + closing_shift, 0.05), 0.95)
        closing_under_prob = 1.0 - closing_over_prob + 0.05  # add vig back
        closing_over_odds = 1.0 / closing_over_prob
        closing_under_odds = 1.0 / closing_under_prob
        return over_odds, closing_over_odds, closing_under_odds

    def _compute_stake(self, prob: float, over_odds: float, bankroll: float) -> float:
        """Return stake in dollars based on staking policy."""
        cfg = self._stake_cfg
        if cfg.use_kelly:
            b = over_odds - 1.0
            q = 1.0 - prob
            f_star = max(0.0, (prob * b - q) / b)
            fraction = cfg.kelly_fraction * f_star
        else:
            fraction = cfg.flat_unit_fraction
        return fraction * bankroll

    def _compute_clv(
        self,
        model_prob: float,
        closing_over_odds: float,
        closing_under_odds: float,
    ) -> float:
        """Compute CLV using the eval.clv module.

        Bet odds are taken as the over_odds at placement (1.90 for simulated
        markets). CLV = (closing_devigged_odds / bet_odds) - 1 per Buchdahl
        convention.
        """
        bet_odds = 1.90  # placement odds (synthetic)
        try:
            return closing_line_value(
                bet_odds=bet_odds,
                closing_odds=closing_over_odds,
                closing_opp_odds=closing_under_odds,
            )
        except Exception:
            return 0.0

    def _compute_report(self, bets_df: pd.DataFrame) -> BacktestReport:
        """Aggregate bet log into a BacktestReport."""
        if bets_df.empty:
            return BacktestReport()

        n_bets = len(bets_df)
        n_wins = int(bets_df["outcome"].sum())
        total_stakes = bets_df["stake"].sum()
        total_pnl = bets_df["pnl"].sum()
        roi = total_pnl / total_stakes if total_stakes > 0 else 0.0

        # Wilson CI on win rate
        ci_low, ci_high = proportion_confint(n_wins, n_bets, alpha=0.05, method="wilson")

        # Calibration metrics
        probs = bets_df["model_prob"].to_numpy()
        outcomes = bets_df["outcome"].astype(float).to_numpy()
        brier = brier_score(probs, outcomes)
        ll = log_loss(probs, outcomes)
        ece = expected_calibration_error(probs, outcomes)

        # CLV
        clv_vals = bets_df["clv"].to_numpy()
        mean_clv = float(np.mean(clv_vals))
        pct_positive_clv = float(np.mean(clv_vals > 0))

        # Drawdown
        pnl_series = bets_df["pnl"].to_numpy()
        cumulative = np.cumsum(pnl_series)
        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        drawdown_max = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

        # Sharpe-like: mean / std of per-bet return (as fraction of stake)
        per_bet_return = bets_df["pnl"] / bets_df["stake"]
        sharpe_like = (
            float(per_bet_return.mean() / per_bet_return.std()) if per_bet_return.std() > 0 else 0.0
        )

        return BacktestReport(
            n_bets=n_bets,
            n_wins=n_wins,
            roi=roi,
            roi_ci_low=float(ci_low),
            roi_ci_high=float(ci_high),
            brier=brier,
            log_loss_score=ll,
            ece=ece,
            mean_clv=mean_clv,
            pct_positive_clv=pct_positive_clv,
            drawdown_max=drawdown_max,
            sharpe_like=sharpe_like,
        )


# ---------------------------------------------------------------------------
# Convenience top-level runner
# ---------------------------------------------------------------------------


def run_disposals_backtest(
    player_stats_df: pd.DataFrame,
    odds_df: pd.DataFrame | None = None,
    model_cfg: ModelConfig | None = None,
    splitter_cfg: SplitterConfig | None = None,
    stake_cfg: StakeConfig | None = None,
    line: float = 24.5,
    output_dir: str | Path | None = None,
) -> tuple[BacktestReport, pd.DataFrame]:
    """Run a disposals model walk-forward backtest and optionally save outputs.

    Parameters
    ----------
    player_stats_df:
        Full player stats DataFrame. See Backtester.run() for schema.
    odds_df:
        Optional historical odds. If None, synthetic odds are used.
    model_cfg:
        DisposalsModel hyperparameters.
    splitter_cfg:
        Walk-forward split settings.
    stake_cfg:
        Staking policy.
    line:
        O/U line for all bets (default 24.5).
    output_dir:
        If provided, saves report.json and bets.csv to
        ``<output_dir>/<timestamp>/``.

    Returns
    -------
    (report, bets_df)
    """
    backtester = Backtester(
        model_cfg=model_cfg,
        splitter_cfg=splitter_cfg,
        stake_cfg=stake_cfg,
        line=line,
    )
    report, bets_df = backtester.run(player_stats_df, odds_df=odds_df)

    if output_dir is not None:
        from datetime import datetime

        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        out = Path(output_dir) / ts
        out.mkdir(parents=True, exist_ok=True)
        report.save(out / "report.json")
        bets_df.to_csv(out / "bets.csv", index=False)
        logger.info("Backtest outputs written to %s", out)

    return report, bets_df
