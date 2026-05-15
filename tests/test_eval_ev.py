"""Tests for bet_advisor.eval.ev -- EV, edge, threshold, uncertainty."""

from __future__ import annotations

import numpy as np
import pytest

from bet_advisor.eval.ev import (
    edge,
    ev_with_uncertainty,
    expected_value,
    min_edge_threshold,
)


class TestExpectedValue:
    def test_positive_ev(self) -> None:
        # 58% model prob at 1.90 odds: EV = 0.58 * 1.90 - 1 = 0.102
        ev = expected_value(0.58, 1.90)
        assert ev == pytest.approx(0.102, abs=1e-6)

    def test_negative_ev(self) -> None:
        # Model prob below implied: clear negative EV.
        ev = expected_value(0.45, 1.90)
        assert ev < 0

    def test_break_even(self) -> None:
        # If model prob == implied prob: EV = 0.
        implied = 1 / 1.90
        ev = expected_value(implied, 1.90)
        assert ev == pytest.approx(0.0, abs=1e-6)

    def test_stake_scales_ev(self) -> None:
        ev_unit = expected_value(0.58, 1.90, stake=1.0)
        ev_hundred = expected_value(0.58, 1.90, stake=100.0)
        assert ev_hundred == pytest.approx(ev_unit * 100.0, abs=1e-6)

    def test_certainty_win(self) -> None:
        # prob=1.0 at odds 2.0: EV should equal net profit = 1.0.
        ev = expected_value(1.0, 2.0)
        assert ev == pytest.approx(1.0, abs=1e-9)

    def test_certainty_loss(self) -> None:
        # prob=0.0: EV should be -stake.
        ev = expected_value(0.0, 2.0, stake=1.0)
        assert ev == pytest.approx(-1.0, abs=1e-9)


class TestEdge:
    def test_equals_ev_per_unit(self) -> None:
        for prob, odds in [(0.55, 2.0), (0.60, 1.80), (0.40, 3.0)]:
            assert edge(prob, odds) == pytest.approx(expected_value(prob, odds, 1.0), abs=1e-9)

    def test_positive_edge_means_value(self) -> None:
        assert edge(0.58, 1.90) > 0

    def test_negative_edge_no_value(self) -> None:
        assert edge(0.40, 1.90) < 0


class TestMinEdgeThreshold:
    def test_default_is_three_percent(self) -> None:
        assert min_edge_threshold() == pytest.approx(0.03, abs=1e-9)

    def test_higher_uncertainty_raises_threshold(self) -> None:
        low = min_edge_threshold(model_uncertainty=0.0)
        high = min_edge_threshold(model_uncertainty=0.10)
        assert high > low

    def test_higher_vig_raises_threshold(self) -> None:
        low = min_edge_threshold(vig=0.05)
        high = min_edge_threshold(vig=0.10)
        assert high > low

    def test_combined_adjustment(self) -> None:
        # uncertainty=0.10 adds 0.5*0.10=0.05; vig=0.10 adds 0.2*0.05=0.01; total=0.09
        result = min_edge_threshold(model_uncertainty=0.10, vig=0.10)
        assert result == pytest.approx(0.03 + 0.05 + 0.01, abs=1e-9)

    def test_vig_at_or_below_five_pct_no_vig_adjustment(self) -> None:
        # vig=0.04 is below 5%; no additional vig adjustment.
        assert min_edge_threshold(vig=0.04) == pytest.approx(0.03, abs=1e-9)


class TestEvWithUncertainty:
    def test_deterministic_samples(self) -> None:
        # All samples identical: mean, p5, p95 all the same.
        samples = np.full(1000, 0.55)
        result = ev_with_uncertainty(samples, 1.95)
        assert result["mean_ev"] == pytest.approx(0.55 * 1.95 - 1, abs=1e-6)
        assert result["p5_ev"] == pytest.approx(result["mean_ev"], abs=1e-6)
        assert result["p95_ev"] == pytest.approx(result["mean_ev"], abs=1e-6)

    def test_positive_ev_fraction(self) -> None:
        # Samples all above break-even -> p_positive should be 1.
        samples = np.full(500, 0.60)
        result = ev_with_uncertainty(samples, 1.90)
        assert result["p_positive"] == pytest.approx(1.0, abs=1e-6)

    def test_negative_ev_fraction(self) -> None:
        # Samples well below break-even -> p_positive should be 0.
        samples = np.full(500, 0.30)
        result = ev_with_uncertainty(samples, 1.90)
        assert result["p_positive"] == pytest.approx(0.0, abs=1e-6)

    def test_p5_le_mean_le_p95(self) -> None:
        rng = np.random.default_rng(0)
        samples = rng.beta(5, 4, size=10000)
        result = ev_with_uncertainty(samples, 2.0)
        assert result["p5_ev"] <= result["mean_ev"] <= result["p95_ev"]

    def test_dict_keys(self) -> None:
        samples = np.array([0.5, 0.6, 0.7])
        result = ev_with_uncertainty(samples, 2.0)
        assert set(result.keys()) == {"mean_ev", "p5_ev", "p95_ev", "p_positive"}
