"""Tests for bet_advisor.eval.devig -- devigging methods and helpers."""

from __future__ import annotations

import pytest

from bet_advisor.eval.devig import (
    additive,
    devig,
    overround,
    power,
    proportional,
    shin,
)

# AFL H2H two-runner market example.
AFL_H2H_2 = [1.80, 2.10]

# Three-runner market example.
MARKET_3 = [2.10, 3.50, 3.20]

# Tolerance for "sums to 1" checks.
SUM_TOL = 1e-6


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _check_sums_to_one(probs: list[float], tol: float = SUM_TOL) -> None:
    assert abs(sum(probs) - 1.0) < tol, f"Probs do not sum to 1: {sum(probs)}"


def _check_all_positive(probs: list[float]) -> None:
    assert all(p > 0 for p in probs), f"Non-positive probability found: {probs}"


# ---------------------------------------------------------------------------
# overround
# ---------------------------------------------------------------------------


class TestOverround:
    def test_known_two_runner(self) -> None:
        # 1/1.80 + 1/2.10 = 0.5556 + 0.4762 = 1.0317 -> overround ~3.17%
        result = overround(AFL_H2H_2)
        assert pytest.approx(result, abs=1e-4) == (1 / 1.80 + 1 / 2.10) - 1.0

    def test_fair_market(self) -> None:
        # 2.0 / 2.0 is a fair H2H market: overround = 0.
        assert pytest.approx(overround([2.0, 2.0]), abs=1e-9) == 0.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            overround([])

    def test_non_positive_raises(self) -> None:
        with pytest.raises(ValueError):
            overround([1.80, -1.0])


# ---------------------------------------------------------------------------
# proportional
# ---------------------------------------------------------------------------


class TestProportional:
    def test_two_runner_sums_to_one(self) -> None:
        _check_sums_to_one(proportional(AFL_H2H_2))

    def test_three_runner_sums_to_one(self) -> None:
        _check_sums_to_one(proportional(MARKET_3))

    def test_single_odd_returns_one(self) -> None:
        result = proportional([3.0])
        assert result == pytest.approx([1.0])

    def test_values_positive(self) -> None:
        _check_all_positive(proportional(AFL_H2H_2))

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            proportional([])

    def test_fair_market_unchanged(self) -> None:
        # For a fair market (sum=1) proportional should return exact implied probs.
        probs = proportional([2.0, 2.0])
        assert probs == pytest.approx([0.5, 0.5], abs=1e-9)

    def test_order_preserved(self) -> None:
        # The larger implied prob (shorter odds) should come first.
        probs = proportional(AFL_H2H_2)
        assert probs[0] > probs[1]


# ---------------------------------------------------------------------------
# power
# ---------------------------------------------------------------------------


class TestPower:
    def test_two_runner_sums_to_one(self) -> None:
        _check_sums_to_one(power(AFL_H2H_2))

    def test_three_runner_sums_to_one(self) -> None:
        _check_sums_to_one(power(MARKET_3))

    def test_single_odd_returns_one(self) -> None:
        result = power([3.0])
        assert result == pytest.approx([1.0])

    def test_values_positive(self) -> None:
        _check_all_positive(power(AFL_H2H_2))

    def test_fair_market(self) -> None:
        probs = power([2.0, 2.0])
        assert probs == pytest.approx([0.5, 0.5], abs=1e-6)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            power([])

    def test_corrects_longshot_bias(self) -> None:
        # In a market with a heavy favourite, power should give the longshot
        # a lower probability than proportional (correcting longshot bias upward
        # for the favourite).
        heavy = [1.25, 5.0]
        prop = proportional(heavy)
        pw = power(heavy)
        # Power method should reduce the longshot probability vs proportional.
        assert pw[1] <= prop[1]


# ---------------------------------------------------------------------------
# shin
# ---------------------------------------------------------------------------


class TestShin:
    def test_two_runner_sums_to_one(self) -> None:
        _check_sums_to_one(shin(AFL_H2H_2))

    def test_three_runner_sums_to_one(self) -> None:
        _check_sums_to_one(shin(MARKET_3), tol=1e-5)

    def test_single_odd_returns_one(self) -> None:
        result = shin([3.0])
        assert result == pytest.approx([1.0])

    def test_values_positive(self) -> None:
        _check_all_positive(shin(AFL_H2H_2))

    def test_fair_market(self) -> None:
        probs = shin([2.0, 2.0])
        assert probs == pytest.approx([0.5, 0.5], abs=1e-6)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            shin([])

    def test_close_to_power_on_binary(self) -> None:
        # For AFL H2H, Shin and power should converge closely.
        sp = shin(AFL_H2H_2)
        pw = power(AFL_H2H_2)
        for s, p in zip(sp, pw):
            assert abs(s - p) < 0.01, f"Shin {s} vs power {p} diverged > 1pp"


# ---------------------------------------------------------------------------
# additive
# ---------------------------------------------------------------------------


class TestAdditive:
    def test_two_runner_sums_to_one(self) -> None:
        _check_sums_to_one(additive(AFL_H2H_2))

    def test_three_runner_sums_to_one(self) -> None:
        _check_sums_to_one(additive(MARKET_3))

    def test_single_odd_returns_one(self) -> None:
        result = additive([3.0])
        assert result == pytest.approx([1.0])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            additive([])

    def test_can_produce_negative(self) -> None:
        # With a very heavy longshot and large overround, additive can go negative.
        # Odds [1.05, 1.05, 50.0] have a massive overround.
        large_book = [1.05, 1.05, 50.0]
        result = additive(large_book)
        # The long shot's prob will be very negative -- that is the documented flaw.
        assert result[2] < 0


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------


class TestDevig:
    def test_default_method_is_power(self) -> None:
        d = devig(AFL_H2H_2)
        pw = power(AFL_H2H_2)
        assert d == pytest.approx(pw, abs=1e-9)

    def test_proportional(self) -> None:
        d = devig(AFL_H2H_2, method="proportional")
        _check_sums_to_one(d)

    def test_shin(self) -> None:
        d = devig(AFL_H2H_2, method="shin")
        _check_sums_to_one(d)

    def test_additive(self) -> None:
        d = devig(AFL_H2H_2, method="additive")
        _check_sums_to_one(d)

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown devig method"):
            devig(AFL_H2H_2, method="bogus")
