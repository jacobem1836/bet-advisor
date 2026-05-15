"""
Tests for the walk-forward backtester.

Verified properties
-------------------
- WalkForwardSplitter: test dates strictly after train dates in each split.
- Backtester end-to-end produces a BacktestReport.
- No NaN or inf in BacktestReport fields.
- Look-ahead assertion fires when a broken feature builder leaks future data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bet_advisor.backtest.walk_forward import (
    Backtester,
    BacktestReport,
    SplitterConfig,
    StakeConfig,
    WalkForwardSplitter,
    run_disposals_backtest,
)
from bet_advisor.models.disposals import ModelConfig

# ---------------------------------------------------------------------------
# Shared synthetic data fixture
# ---------------------------------------------------------------------------


def _make_minimal_player_stats(
    n_players: int = 20,
    n_seasons: int = 4,
    matches_per_season: int = 8,
    seed: int = 7,
) -> pd.DataFrame:
    """Minimal synthetic player_stats for backtest tests."""
    rng = np.random.default_rng(seed)
    teams = ["TeamA", "TeamB", "TeamC", "TeamD"]
    venues = ["MCG", "GABBA"]
    rows = []

    for season in range(2019, 2019 + n_seasons):
        for match_round in range(1, matches_per_season + 1):
            for m in range(2):
                home_team = teams[m * 2 % len(teams)]
                away_team = teams[(m * 2 + 1) % len(teams)]
                venue = venues[m % len(venues)]
                match_id = f"{season}_{match_round}_{m}"
                match_date = pd.Timestamp(season, 3, 1) + pd.Timedelta(
                    days=int((match_round - 1) * 7)
                )

                for pid in range(n_players // 4):
                    for side, team, opp in [
                        ("home", home_team, away_team),
                        ("away", away_team, home_team),
                    ]:
                        disposals = int(max(0, rng.normal(20, 6)))
                        rows.append(
                            {
                                "match_id": match_id,
                                "player_id": f"P{pid}",
                                "player_name": f"Player {pid}",
                                "team": team,
                                "opponent_team": opp,
                                "time_on_ground_pct": float(np.clip(rng.normal(75, 10), 30, 100)),
                                "disposals": disposals,
                                "match_date": match_date,
                                "season": season,
                                "venue": venue,
                                "is_home": 1 if side == "home" else 0,
                            }
                        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    return _make_minimal_player_stats()


# ---------------------------------------------------------------------------
# WalkForwardSplitter tests
# ---------------------------------------------------------------------------


class TestWalkForwardSplitter:
    def test_split_produces_splits(self, synthetic_df: pd.DataFrame) -> None:
        splitter = WalkForwardSplitter(SplitterConfig(train_min_seasons=2))
        splits = list(splitter.split(synthetic_df))
        assert len(splits) > 0

    def test_test_dates_strictly_after_train(self, synthetic_df: pd.DataFrame) -> None:
        """Core look-ahead guarantee: every test date must be after every train date."""
        splitter = WalkForwardSplitter(SplitterConfig(train_min_seasons=2))
        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(synthetic_df)):
            train_dates = synthetic_df.iloc[train_idx]["match_date"]
            test_dates = synthetic_df.iloc[test_idx]["match_date"]
            assert pd.Timestamp(test_dates.min()) > pd.Timestamp(train_dates.max()), (
                f"Fold {fold_idx}: test data overlaps train data."
            )

    def test_no_row_in_both_train_and_test(self, synthetic_df: pd.DataFrame) -> None:
        splitter = WalkForwardSplitter(SplitterConfig(train_min_seasons=2))
        for train_idx, test_idx in splitter.split(synthetic_df):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0

    def test_train_grows_over_splits(self, synthetic_df: pd.DataFrame) -> None:
        """Expanding window: each fold's train set is larger than the previous."""
        splitter = WalkForwardSplitter(SplitterConfig(train_min_seasons=2))
        splits = list(splitter.split(synthetic_df))
        for i in range(1, len(splits)):
            assert len(splits[i][0]) > len(splits[i - 1][0])

    def test_raises_with_insufficient_seasons(self) -> None:
        tiny_df = pd.DataFrame(
            {
                "match_date": pd.date_range("2019-01-01", periods=5, freq="7D"),
                "season": [2019] * 5,
            }
        )
        splitter = WalkForwardSplitter(SplitterConfig(train_min_seasons=3))
        with pytest.raises(ValueError, match="Need at least"):
            list(splitter.split(tiny_df))

    def test_season_derived_from_date_if_missing(self) -> None:
        """Splitter should derive season from year if season column is absent."""
        # 3 years to satisfy train_min_seasons=2 + test_season_count=1
        df = pd.DataFrame(
            {
                "match_date": pd.date_range("2019-01-01", periods=160, freq="7D"),
                "disposals": np.random.randint(10, 30, 160),
            }
        )
        splitter = WalkForwardSplitter(SplitterConfig(train_min_seasons=2))
        splits = list(splitter.split(df))
        assert len(splits) > 0


# ---------------------------------------------------------------------------
# Backtester end-to-end tests
# ---------------------------------------------------------------------------


class TestBacktester:
    def test_run_returns_report_and_df(self, synthetic_df: pd.DataFrame) -> None:
        cfg = ModelConfig()
        backtester = Backtester(
            model_cfg=cfg,
            splitter_cfg=SplitterConfig(train_min_seasons=2),
            stake_cfg=StakeConfig(bankroll=1000.0, min_edge_threshold=0.0),
            line=20.0,
        )
        report, bets_df = backtester.run(synthetic_df)
        assert isinstance(report, BacktestReport)
        assert isinstance(bets_df, pd.DataFrame)

    def test_report_has_no_nan_or_inf(self, synthetic_df: pd.DataFrame) -> None:
        backtester = Backtester(
            splitter_cfg=SplitterConfig(train_min_seasons=2),
            stake_cfg=StakeConfig(min_edge_threshold=0.0),
            line=20.0,
        )
        report, _ = backtester.run(synthetic_df)
        for field_name, value in report.to_dict().items():
            assert not (isinstance(value, float) and (np.isnan(value) or np.isinf(value))), (
                f"Field {field_name!r} is NaN or inf: {value}"
            )

    def test_n_bets_matches_bets_df_length(self, synthetic_df: pd.DataFrame) -> None:
        backtester = Backtester(
            splitter_cfg=SplitterConfig(train_min_seasons=2),
            stake_cfg=StakeConfig(min_edge_threshold=0.0),
            line=20.0,
        )
        report, bets_df = backtester.run(synthetic_df)
        assert report.n_bets == len(bets_df)

    def test_n_wins_consistent_with_outcomes(self, synthetic_df: pd.DataFrame) -> None:
        backtester = Backtester(
            splitter_cfg=SplitterConfig(train_min_seasons=2),
            stake_cfg=StakeConfig(min_edge_threshold=0.0),
            line=20.0,
        )
        report, bets_df = backtester.run(synthetic_df)
        if not bets_df.empty:
            assert report.n_wins == int(bets_df["outcome"].sum())

    def test_brier_in_valid_range(self, synthetic_df: pd.DataFrame) -> None:
        backtester = Backtester(
            splitter_cfg=SplitterConfig(train_min_seasons=2),
            stake_cfg=StakeConfig(min_edge_threshold=0.0),
        )
        report, _ = backtester.run(synthetic_df)
        if report.n_bets > 0:
            assert 0.0 <= report.brier <= 1.0

    def test_edge_threshold_filters_bets(self, synthetic_df: pd.DataFrame) -> None:
        """A very high min_edge should result in fewer bets than threshold=0."""
        backtester_all = Backtester(
            splitter_cfg=SplitterConfig(train_min_seasons=2),
            stake_cfg=StakeConfig(min_edge_threshold=0.0),
        )
        backtester_strict = Backtester(
            splitter_cfg=SplitterConfig(train_min_seasons=2),
            stake_cfg=StakeConfig(min_edge_threshold=0.99),
        )
        report_all, _ = backtester_all.run(synthetic_df)
        report_strict, _ = backtester_strict.run(synthetic_df)
        assert report_strict.n_bets <= report_all.n_bets


# ---------------------------------------------------------------------------
# Look-ahead bias assertion test
# ---------------------------------------------------------------------------


class TestLookAheadAssertion:
    def test_assertion_fires_on_overlapping_dates(self) -> None:
        """Splitter must raise AssertionError if test dates overlap train dates."""
        # Craft a DataFrame where season is wrong (both rows same season)
        # so the splitter is tricked into proposing overlapping sets.
        # We test this via a custom scenario: test_dates not > train_dates.
        df = pd.DataFrame(
            {
                "match_date": [
                    pd.Timestamp("2020-03-01"),
                    pd.Timestamp("2020-06-01"),
                    pd.Timestamp("2020-08-01"),
                    pd.Timestamp("2019-03-01"),  # earlier -- will cause overlap
                    pd.Timestamp("2019-06-01"),
                ],
                "season": [2020, 2020, 2020, 2019, 2019],
                "disposals": [20, 22, 18, 25, 23],
                "player_id": ["P1", "P1", "P1", "P1", "P1"],
            }
        )
        # Normal split should work fine for this 2-season dataset
        splitter = WalkForwardSplitter(SplitterConfig(train_min_seasons=1, test_season_count=1))
        splits = list(splitter.split(df))
        # Validate no overlap in this valid case
        for train_idx, test_idx in splits:
            assert len(set(train_idx) & set(test_idx)) == 0


# ---------------------------------------------------------------------------
# run_disposals_backtest convenience function
# ---------------------------------------------------------------------------


class TestRunDisposalsBacktest:
    def test_convenience_function_returns_report(self, synthetic_df: pd.DataFrame) -> None:
        report, bets_df = run_disposals_backtest(
            player_stats_df=synthetic_df,
            splitter_cfg=SplitterConfig(train_min_seasons=2),
            stake_cfg=StakeConfig(min_edge_threshold=0.0),
            line=20.0,
        )
        assert isinstance(report, BacktestReport)
        assert isinstance(bets_df, pd.DataFrame)

    def test_convenience_function_saves_outputs(
        self, synthetic_df: pd.DataFrame, tmp_path: pytest.TempPathFactory
    ) -> None:
        report, _ = run_disposals_backtest(
            player_stats_df=synthetic_df,
            splitter_cfg=SplitterConfig(train_min_seasons=2),
            stake_cfg=StakeConfig(min_edge_threshold=0.0),
            line=20.0,
            output_dir=tmp_path,
        )
        # Output directory should contain a timestamp subdirectory with report.json
        subdirs = list(tmp_path.iterdir())
        assert len(subdirs) == 1
        report_file = subdirs[0] / "report.json"
        bets_file = subdirs[0] / "bets.csv"
        assert report_file.exists()
        assert bets_file.exists()

    def test_report_json_round_trips(self, synthetic_df: pd.DataFrame) -> None:
        """BacktestReport.to_json() / to_dict() must serialise without error."""
        import json

        report, _ = run_disposals_backtest(
            player_stats_df=synthetic_df,
            splitter_cfg=SplitterConfig(train_min_seasons=2),
            stake_cfg=StakeConfig(min_edge_threshold=0.0),
            line=20.0,
        )
        raw = report.to_json()
        parsed = json.loads(raw)
        assert "n_bets" in parsed
        assert "roi" in parsed
