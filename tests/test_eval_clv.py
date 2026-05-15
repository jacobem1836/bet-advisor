"""Tests for bet_advisor.eval.clv -- closing line value computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bet_advisor.eval.clv import (
    aggregate_clv,
    closing_line_value,
    closing_line_value_market,
    clv_significance,
)

# ---------------------------------------------------------------------------
# closing_line_value (two-runner)
# ---------------------------------------------------------------------------


class TestClosingLineValue:
    def test_positive_clv_when_beat_close(self) -> None:
        # Bettor took 2.10; market closed at 2.00 (shorter) -- bettor beat close.
        clv = closing_line_value(
            bet_odds=2.10,
            closing_odds=2.00,
            closing_opp_odds=1.90,
        )
        assert clv > 0

    def test_negative_clv_when_missed_close(self) -> None:
        # Bettor took 1.85; market closed at 2.05 (longer) -- missed the move.
        clv = closing_line_value(
            bet_odds=1.85,
            closing_odds=2.05,
            closing_opp_odds=1.85,
        )
        assert clv < 0

    def test_zero_clv_at_fair_closing(self) -> None:
        # Bet at the fair closing odds exactly (no opp odds at placement, so
        # raw implied == devigged closing). Approximately zero.
        # Fair market at 2.0/2.0: devigged prob = 0.5; bet_implied = 0.5.
        clv = closing_line_value(
            bet_odds=2.0,
            closing_odds=2.0,
            closing_opp_odds=2.0,
        )
        assert abs(clv) < 0.01

    def test_invalid_bet_odds_raises(self) -> None:
        with pytest.raises(ValueError, match="bet_odds"):
            closing_line_value(bet_odds=0.9, closing_odds=2.0, closing_opp_odds=2.0)

    def test_invalid_closing_odds_raises(self) -> None:
        with pytest.raises(ValueError, match="closing_odds"):
            closing_line_value(bet_odds=2.0, closing_odds=0.5, closing_opp_odds=2.0)

    def test_invalid_closing_opp_raises(self) -> None:
        with pytest.raises(ValueError, match="closing_opp_odds"):
            closing_line_value(bet_odds=2.0, closing_odds=2.0, closing_opp_odds=-1.0)

    def test_devig_methods_consistent(self) -> None:
        # Power and proportional should give similar results for typical markets.
        kwargs = {"bet_odds": 2.10, "closing_odds": 2.0, "closing_opp_odds": 1.90}
        clv_power = closing_line_value(**kwargs, devig_method="power")
        clv_prop = closing_line_value(**kwargs, devig_method="proportional")
        assert abs(clv_power - clv_prop) < 0.02


# ---------------------------------------------------------------------------
# closing_line_value_market (n-runner)
# ---------------------------------------------------------------------------


class TestClosingLineValueMarket:
    def test_two_runner_consistent_with_simple(self) -> None:
        bet_odds = 2.10
        closing = [2.0, 1.90]
        market_clv = closing_line_value_market(bet_odds, 0, closing)
        simple_clv = closing_line_value(bet_odds, closing[0], closing[1])
        assert abs(market_clv - simple_clv) < 1e-6

    def test_runner_idx_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="bet_runner_idx"):
            closing_line_value_market(2.10, 5, [2.0, 1.90])

    def test_positive_clv_three_runner(self) -> None:
        # Bet at 3.50 on runner 1; market closes at 3.00 -- beat the close.
        clv = closing_line_value_market(
            bet_odds=3.50,
            bet_runner_idx=1,
            market_close_odds=[2.10, 3.00, 4.50],
        )
        assert clv > 0

    def test_negative_clv_three_runner(self) -> None:
        # Bet at 2.00 on runner 0; market closes with runner 0 at 2.50 (drifted out).
        clv = closing_line_value_market(
            bet_odds=2.00,
            bet_runner_idx=0,
            market_close_odds=[2.50, 3.00, 4.00],
        )
        assert clv < 0


# ---------------------------------------------------------------------------
# aggregate_clv
# ---------------------------------------------------------------------------


class TestAggregateCLV:
    def _make_df(self, clv_values: list[float]) -> pd.DataFrame:
        return pd.DataFrame({"clv_pct": clv_values})

    def test_mean_and_median(self) -> None:
        df = self._make_df([0.02, 0.04, -0.01, 0.03])
        result = aggregate_clv(df)
        assert result["mean_clv"] == pytest.approx(np.mean([0.02, 0.04, -0.01, 0.03]), abs=1e-9)
        assert result["median_clv"] == pytest.approx(np.median([0.02, 0.04, -0.01, 0.03]), abs=1e-9)

    def test_pct_positive(self) -> None:
        df = self._make_df([0.01, 0.02, -0.01, -0.02])
        result = aggregate_clv(df)
        assert result["pct_positive"] == pytest.approx(0.5, abs=1e-9)

    def test_wilson_interval_shape(self) -> None:
        df = self._make_df([0.01] * 80 + [-0.01] * 20)
        result = aggregate_clv(df)
        assert result["wilson_lower"] < result["pct_positive"]
        assert result["wilson_upper"] > result["pct_positive"]
        assert 0.0 <= result["wilson_lower"] <= 1.0
        assert 0.0 <= result["wilson_upper"] <= 1.0

    def test_n_excludes_nulls(self) -> None:
        df = pd.DataFrame({"clv_pct": [0.01, None, 0.02, None, -0.01]})
        result = aggregate_clv(df)
        assert result["n"] == 3

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"other": [1, 2, 3]})
        with pytest.raises(ValueError, match="clv_pct"):
            aggregate_clv(df)

    def test_all_null_raises(self) -> None:
        df = pd.DataFrame({"clv_pct": [None, None]})
        with pytest.raises(ValueError, match="No non-null"):
            aggregate_clv(df)

    def test_all_positive_rate_one(self) -> None:
        df = self._make_df([0.01, 0.02, 0.03])
        result = aggregate_clv(df)
        assert result["pct_positive"] == pytest.approx(1.0, abs=1e-9)

    def test_small_sample_wide_wilson(self) -> None:
        # With n=5 the Wilson CI should be wide (high uncertainty).
        df = self._make_df([0.01, 0.02, -0.01, 0.03, -0.02])
        result = aggregate_clv(df)
        width = result["wilson_upper"] - result["wilson_lower"]
        assert width > 0.2


# ---------------------------------------------------------------------------
# clv_significance
# ---------------------------------------------------------------------------


class TestCLVSignificance:
    def _make_df(self, clv_values: list[float]) -> pd.DataFrame:
        return pd.DataFrame({"clv_pct": clv_values})

    def test_clearly_positive_clv_low_p_value(self) -> None:
        # 200 bets with large positive CLV -- p-value should be very small.
        rng = np.random.default_rng(0)
        clv = rng.normal(loc=0.03, scale=0.01, size=200).tolist()
        df = self._make_df(clv)
        result = clv_significance(df)
        assert result["p_value"] < 0.001
        assert result["t_stat"] > 0

    def test_zero_clv_non_significant(self) -> None:
        # CLV centred on 0 -- t-stat should be near 0.
        rng = np.random.default_rng(1)
        clv = rng.normal(loc=0.0, scale=0.02, size=300).tolist()
        df = self._make_df(clv)
        result = clv_significance(df)
        assert result["p_value"] > 0.05

    def test_small_sample_wide_bootstrap_ci(self) -> None:
        # With n=10 the bootstrap CI should be wide.
        df = self._make_df([0.01, -0.01, 0.02, -0.02, 0.01, 0.01, -0.01, 0.02, -0.02, 0.01])
        result = clv_significance(df)
        width = result["boot_ci_upper"] - result["boot_ci_lower"]
        assert width > 0.01

    def test_dict_keys(self) -> None:
        df = self._make_df([0.01, 0.02, -0.01, 0.03])
        result = clv_significance(df)
        expected = {"mean_clv", "t_stat", "p_value", "boot_ci_lower", "boot_ci_upper", "n"}
        assert expected.issubset(set(result.keys()))

    def test_insufficient_data_raises(self) -> None:
        df = self._make_df([0.01])
        with pytest.raises(ValueError, match="at least 2"):
            clv_significance(df)

    def test_missing_column_raises(self) -> None:
        df = pd.DataFrame({"other": [0.1, 0.2]})
        with pytest.raises(ValueError, match="clv_pct"):
            clv_significance(df)
