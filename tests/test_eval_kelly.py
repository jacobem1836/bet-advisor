"""Tests for bet_advisor.eval.kelly -- Kelly criterion variants and stake sizing."""

from __future__ import annotations

import pytest

from bet_advisor.eval.kelly import (
    capped_kelly,
    fractional_kelly,
    full_kelly,
    portfolio_kelly,
    stake_recommendation,
)


class TestFullKelly:
    def test_known_value(self) -> None:
        # prob=0.55, odds=2.0: f* = (0.55 - 0.45) / 1 = 0.10
        result = full_kelly(0.55, 2.0)
        assert result == pytest.approx(0.10, abs=1e-6)

    def test_zero_edge_returns_zero(self) -> None:
        # Break-even: implied prob = 1/odds = 0.5 at odds 2.0.
        result = full_kelly(0.50, 2.0)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_negative_edge_returns_zero(self) -> None:
        result = full_kelly(0.40, 2.0)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_clamped_to_one(self) -> None:
        # Extreme probability should clamp to 1.
        result = full_kelly(0.999, 2.0)
        assert result <= 1.0

    def test_formula_correctness(self) -> None:
        # f* = (p*b - q) / b where b = odds - 1.
        prob, odds = 0.60, 1.80
        b = odds - 1.0
        expected = (prob * b - (1 - prob)) / b
        assert full_kelly(prob, odds) == pytest.approx(max(0.0, expected), abs=1e-9)

    def test_odds_at_one_returns_zero(self) -> None:
        # odds = 1 means no profit; b = 0 -- should return 0 gracefully.
        result = full_kelly(0.80, 1.0)
        assert result == 0.0


class TestFractionalKelly:
    def test_quarter_reduces_stake(self) -> None:
        fk = full_kelly(0.55, 2.0)
        frac = fractional_kelly(0.55, 2.0, fraction=0.25)
        assert frac == pytest.approx(fk * 0.25, abs=1e-9)

    def test_default_fraction_is_quarter(self) -> None:
        result = fractional_kelly(0.55, 2.0)
        assert result == pytest.approx(fractional_kelly(0.55, 2.0, fraction=0.25), abs=1e-9)

    def test_zero_edge_returns_zero(self) -> None:
        result = fractional_kelly(0.50, 2.0)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_negative_edge_returns_zero(self) -> None:
        result = fractional_kelly(0.30, 2.0)
        assert result == pytest.approx(0.0, abs=1e-9)


class TestCappedKelly:
    def test_cap_enforced(self) -> None:
        # Very high edge -- fractional Kelly might exceed cap.
        result = capped_kelly(0.90, 2.0, cap=0.05, fraction=1.0)
        assert result <= 0.05

    def test_below_cap_unchanged(self) -> None:
        # Small edge: uncapped Kelly should be below 5%.
        frac = fractional_kelly(0.52, 2.0, fraction=0.25)
        capped = capped_kelly(0.52, 2.0, cap=0.05, fraction=0.25)
        if frac < 0.05:
            assert capped == pytest.approx(frac, abs=1e-9)

    def test_default_cap_is_five_percent(self) -> None:
        result = capped_kelly(0.95, 2.0, fraction=1.0)
        assert result <= 0.05

    def test_zero_edge_returns_zero(self) -> None:
        result = capped_kelly(0.50, 2.0)
        assert result == pytest.approx(0.0, abs=1e-9)


class TestPortfolioKelly:
    def test_empty_returns_empty(self) -> None:
        assert portfolio_kelly([]) == []

    def test_single_bet_reasonable(self) -> None:
        result = portfolio_kelly([{"prob": 0.55, "decimal_odds": 2.0}])
        assert len(result) == 1
        assert 0.0 <= result[0] <= 0.05

    def test_three_bets_sane(self) -> None:
        bets = [
            {"prob": 0.55, "decimal_odds": 2.0},
            {"prob": 0.58, "decimal_odds": 1.90},
            {"prob": 0.52, "decimal_odds": 2.10},
        ]
        result = portfolio_kelly(bets)
        assert len(result) == 3
        # All fractions should be in [0, 0.05].
        for f in result:
            assert 0.0 <= f <= 0.05

    def test_zero_edge_bets_zero(self) -> None:
        # A bet with no edge should receive zero or near-zero allocation.
        bets = [{"prob": 0.50, "decimal_odds": 2.0}]
        result = portfolio_kelly(bets)
        assert result[0] == pytest.approx(0.0, abs=0.01)

    def test_higher_edge_gets_more(self) -> None:
        # Higher-edge bet should get at least as much as lower-edge.
        bets = [
            {"prob": 0.60, "decimal_odds": 2.0},  # edge=20%
            {"prob": 0.52, "decimal_odds": 2.0},  # edge=4%
        ]
        result = portfolio_kelly(bets)
        assert result[0] >= result[1]


class TestStakeRecommendation:
    def test_flat_mode(self) -> None:
        rec = stake_recommendation(0.55, 2.0, bankroll=1000.0, mode="flat")
        assert rec["stake"] == pytest.approx(10.0, abs=1e-6)  # 1% of 1000
        assert rec["mode"] == "flat"

    def test_quarter_kelly_mode(self) -> None:
        rec = stake_recommendation(0.55, 2.0, bankroll=1000.0, mode="quarter_kelly")
        fk = full_kelly(0.55, 2.0)
        expected_stake = 0.25 * fk * 1000.0
        assert rec["stake"] == pytest.approx(expected_stake, abs=1e-4)

    def test_capped_kelly_mode(self) -> None:
        rec = stake_recommendation(0.90, 2.0, bankroll=1000.0, mode="capped_kelly")
        assert rec["stake"] <= 0.05 * 1000.0  # 5% cap

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown stake mode"):
            stake_recommendation(0.55, 2.0, bankroll=1000.0, mode="magic")

    def test_dict_keys(self) -> None:
        rec = stake_recommendation(0.55, 2.0, bankroll=1000.0)
        expected_keys = {"stake", "mode", "prob", "odds", "edge", "ev", "bankroll_pct"}
        assert expected_keys.issubset(set(rec.keys()))

    def test_edge_field_matches_ev(self) -> None:
        rec = stake_recommendation(0.60, 1.90, bankroll=500.0)
        assert rec["edge"] == rec["ev"]

    def test_bankroll_pct_consistent(self) -> None:
        rec = stake_recommendation(0.55, 2.0, bankroll=1000.0, mode="flat")
        assert rec["bankroll_pct"] == pytest.approx(rec["stake"] / 1000.0, abs=1e-9)
