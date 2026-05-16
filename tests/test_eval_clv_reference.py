"""Tests for ClvReferenceResolver -- all 4 modes, fallbacks, and edge cases."""

from __future__ import annotations

import math

import pytest

from bet_advisor.eval.clv_reference import (
    ClvReferenceConfig,
    ClvReferenceResolver,
    build_default_resolver,
)
from bet_advisor.eval.devig import devig as _devig
from bet_advisor.eval.devig import overround as _overround


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _two_runner_snapshot(
    books: dict[str, dict[str, float]],
    runners: tuple[str, str] = ("Richmond", "Melbourne"),
    betfair: dict | None = None,
) -> dict:
    """Build a minimal two-runner close snapshot."""
    runner_list = []
    for name in runners:
        r: dict = {"name": name, "books": {b: odds[name] for b, odds in books.items()}}
        if betfair is not None and name in betfair:
            r["betfair_delayed"] = betfair[name]
        runner_list.append(r)
    return {"runners": runner_list}


def _approx(val: float, abs_tol: float = 1e-6) -> float:
    """Return val to a tolerance for assertions (pytest.approx wrapper)."""
    return pytest.approx(val, abs=abs_tol)


# ---------------------------------------------------------------------------
# 1. Multi-book consensus - equal weighting
# ---------------------------------------------------------------------------


class TestConsensusEqualWeighting:
    """Consensus with equal weighting averages devigged probs across books."""

    def test_consensus_equal_five_books(self) -> None:
        """Consensus of 5 books (equal weight) == arithmetic mean of per-book fair probs."""
        books_odds = {
            "sportsbet": {"A": 1.80, "B": 2.10},
            "tab":       {"A": 1.82, "B": 2.08},
            "ladbrokes": {"A": 1.78, "B": 2.12},
            "pointsbet": {"A": 1.81, "B": 2.09},
            "betr":      {"A": 1.83, "B": 2.07},
        }
        snapshot = _two_runner_snapshot(books_odds, runners=("A", "B"))
        cfg = ClvReferenceConfig(
            mode="multi_book_consensus",
            devig_method="power",
            consensus_weighting="equal",
            consensus_min_books=3,
        )
        resolver = ClvReferenceResolver(cfg)
        result = resolver.resolve(snapshot)

        # Compute expected per-book probs independently.
        per_book_probs_a: list[float] = []
        for odds in books_odds.values():
            fair = _devig([odds["A"], odds["B"]], method="power")
            per_book_probs_a.append(fair[0])

        expected_a = sum(per_book_probs_a) / len(per_book_probs_a)

        assert result["method"] == "multi_book_consensus"
        assert len(result["runners"]) == 2
        assert result["runners"][0]["name"] == "A"
        assert result["runners"][0]["ref_prob"] == _approx(expected_a, abs_tol=1e-6)
        # Probs should sum to 1.
        total = sum(r["ref_prob"] for r in result["runners"])
        assert total == _approx(1.0)

    def test_no_warnings_with_sufficient_books(self) -> None:
        books_odds = {b: {"A": 1.80, "B": 2.10} for b in ["sportsbet", "tab", "ladbrokes"]}
        snapshot = _two_runner_snapshot(books_odds, runners=("A", "B"))
        cfg = ClvReferenceConfig(
            mode="multi_book_consensus",
            consensus_books=["sportsbet", "tab", "ladbrokes"],
            consensus_min_books=3,
            consensus_weighting="equal",
        )
        resolver = ClvReferenceResolver(cfg)
        result = resolver.resolve(snapshot)
        assert result["warnings"] == []


# ---------------------------------------------------------------------------
# 2. Multi-book consensus - by_overround weighting
# ---------------------------------------------------------------------------


class TestConsensusOverroundWeighting:
    """Lower-overround book contributes more weight."""

    def test_by_overround_weighting_biases_toward_sharper_book(self) -> None:
        """Book with 4% overround should receive more weight than one with 10%."""
        # sharp_book: very low overround (~2%)
        # blunt_book: high overround (~10%)
        # For a two-runner market where we control overround exactly:
        # sharp: A=1.97, B=1.97  -> sum(1/o) ~= 1.0152  -> overround ~1.52%
        # blunt: A=1.72, B=1.72  -> sum(1/o) ~= 1.163   -> overround ~16.3%
        books_odds = {
            "sharp_book": {"A": 1.97, "B": 1.97},
            "blunt_book": {"A": 1.72, "B": 1.72},
        }
        snapshot = _two_runner_snapshot(books_odds, runners=("A", "B"))
        cfg = ClvReferenceConfig(
            mode="multi_book_consensus",
            devig_method="power",
            consensus_books=["sharp_book", "blunt_book"],
            consensus_min_books=1,
            consensus_weighting="by_overround",
        )
        resolver = ClvReferenceResolver(cfg)
        result = resolver.resolve(snapshot)

        or_sharp = _overround([1.97, 1.97])
        or_blunt = _overround([1.72, 1.72])
        w_sharp = (1.0 / (1.0 + or_sharp))
        w_blunt = (1.0 / (1.0 + or_blunt))
        total_w = w_sharp + w_blunt
        w_sharp /= total_w
        w_blunt /= total_w

        assert w_sharp > w_blunt, "Sharp book must have higher weight"

        # Both books' devigged prob for runner A is 0.5 (symmetric market).
        # So consensus prob is also 0.5 regardless of weighting.
        assert result["runners"][0]["ref_prob"] == _approx(0.5, abs_tol=1e-4)

    def test_asymmetric_odds_by_overround_weights_correctly(self) -> None:
        """Verify weighted average formula with asymmetric odds."""
        # book1: lower overround, book2: higher overround
        books_odds = {
            "book1": {"A": 1.90, "B": 1.95},   # ~3% overround
            "book2": {"A": 1.80, "B": 1.92},   # ~8% overround
        }
        snapshot = _two_runner_snapshot(books_odds, runners=("A", "B"))
        cfg = ClvReferenceConfig(
            mode="multi_book_consensus",
            devig_method="power",
            consensus_books=["book1", "book2"],
            consensus_min_books=1,
            consensus_weighting="by_overround",
        )
        resolver = ClvReferenceResolver(cfg)
        result = resolver.resolve(snapshot)

        # Compute expected manually.
        or1 = _overround([1.90, 1.95])
        or2 = _overround([1.80, 1.92])
        w1_raw = 1.0 / (1.0 + or1)
        w2_raw = 1.0 / (1.0 + or2)
        total_w = w1_raw + w2_raw
        w1 = w1_raw / total_w
        w2 = w2_raw / total_w

        p1 = _devig([1.90, 1.95], method="power")
        p2 = _devig([1.80, 1.92], method="power")

        expected_a = w1 * p1[0] + w2 * p2[0]
        total = w1 * p1[0] + w2 * p2[0] + w1 * p1[1] + w2 * p2[1]
        expected_a /= total  # normalised

        assert result["runners"][0]["ref_prob"] == _approx(expected_a, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# 3. Min-books fallback
# ---------------------------------------------------------------------------


class TestMinBooksFallback:
    """When fewer books than consensus_min_books are available, warn and continue."""

    def test_fewer_books_than_min_emits_warning(self) -> None:
        books_odds = {
            "sportsbet": {"A": 1.80, "B": 2.10},
            "tab":       {"A": 1.82, "B": 2.08},
        }
        snapshot = _two_runner_snapshot(books_odds, runners=("A", "B"))
        cfg = ClvReferenceConfig(
            mode="multi_book_consensus",
            consensus_books=["sportsbet", "tab", "ladbrokes", "pointsbet", "betr"],
            consensus_min_books=3,
            consensus_weighting="equal",
        )
        resolver = ClvReferenceResolver(cfg)
        result = resolver.resolve(snapshot)

        assert any("2" in w and "minimum" in w for w in result["warnings"]), (
            "Expected a warning about only 2 books being available"
        )
        # Still resolves -- probs sum to 1.
        total = sum(r["ref_prob"] for r in result["runners"])
        assert total == _approx(1.0)
        assert len(result["books_used"]) == 2

    def test_two_books_still_produces_valid_probs(self) -> None:
        books_odds = {
            "sportsbet": {"A": 1.80, "B": 2.10},
            "tab":       {"A": 1.82, "B": 2.08},
        }
        snapshot = _two_runner_snapshot(books_odds, runners=("A", "B"))
        cfg = ClvReferenceConfig(
            mode="multi_book_consensus",
            consensus_books=["sportsbet", "tab"],
            consensus_min_books=3,
            consensus_weighting="equal",
        )
        resolver = ClvReferenceResolver(cfg)
        result = resolver.resolve(snapshot)

        for r in result["runners"]:
            assert 0.0 < r["ref_prob"] < 1.0


# ---------------------------------------------------------------------------
# 4. Betfair delayed mode
# ---------------------------------------------------------------------------


class TestBetfairDelayedMode:
    def _make_bf_snapshot(
        self,
        volume_a: float = 2000.0,
        volume_b: float = 800.0,
        price_a: float = 1.83,
        price_b: float = 2.10,
    ) -> dict:
        return {
            "runners": [
                {
                    "name": "A",
                    "books": {"sportsbet": 1.80, "tab": 1.82},
                    "betfair_delayed": {"price": price_a, "volume_matched": volume_a},
                },
                {
                    "name": "B",
                    "books": {"sportsbet": 2.10, "tab": 2.08},
                    "betfair_delayed": {"price": price_b, "volume_matched": volume_b},
                },
            ]
        }

    def test_betfair_used_when_volume_sufficient(self) -> None:
        snapshot = self._make_bf_snapshot(volume_a=1500.0, volume_b=1500.0)
        cfg = ClvReferenceConfig(
            mode="betfair_delayed",
            betfair_delayed_min_volume=1000.0,
            consensus_books=["sportsbet", "tab"],
            consensus_min_books=1,
        )
        resolver = ClvReferenceResolver(cfg)
        result = resolver.resolve(snapshot)

        assert result["method"] == "betfair_delayed"
        assert result["books_used"] == ["betfair_delayed"]
        assert result["warnings"] == []
        # Probs should be devigged Betfair prices.
        expected = _devig([1.83, 2.10], method="power")
        assert result["runners"][0]["ref_prob"] == _approx(expected[0])
        assert result["runners"][0]["ref_source"] == "betfair_delayed"

    def test_betfair_falls_back_when_volume_below_threshold(self) -> None:
        snapshot = self._make_bf_snapshot(volume_a=400.0, volume_b=200.0)
        cfg = ClvReferenceConfig(
            mode="betfair_delayed",
            betfair_delayed_min_volume=1000.0,
            consensus_books=["sportsbet", "tab"],
            consensus_min_books=1,
            consensus_weighting="equal",
        )
        resolver = ClvReferenceResolver(cfg)
        result = resolver.resolve(snapshot)

        assert result["method"] == "multi_book_consensus"
        assert any("fall" in w.lower() for w in result["warnings"])
        # Probs are from consensus, not Betfair.
        assert result["books_used"] != ["betfair_delayed"]

    def test_betfair_falls_back_when_prices_missing(self) -> None:
        snapshot = {
            "runners": [
                {"name": "A", "books": {"sportsbet": 1.80, "tab": 1.82}},
                {"name": "B", "books": {"sportsbet": 2.10, "tab": 2.08}},
            ]
        }
        cfg = ClvReferenceConfig(
            mode="betfair_delayed",
            betfair_delayed_min_volume=1000.0,
            consensus_books=["sportsbet", "tab"],
            consensus_min_books=1,
            consensus_weighting="equal",
        )
        resolver = ClvReferenceResolver(cfg)
        result = resolver.resolve(snapshot)

        assert result["method"] == "multi_book_consensus"
        assert any("fall" in w.lower() for w in result["warnings"])


# ---------------------------------------------------------------------------
# 5. Sportsbet-only mode
# ---------------------------------------------------------------------------


class TestSportsbetOnlyMode:
    def test_sportsbet_only_uses_single_book(self) -> None:
        books_odds = {
            "sportsbet": {"A": 1.80, "B": 2.10},
            "tab":       {"A": 1.82, "B": 2.08},
        }
        snapshot = _two_runner_snapshot(books_odds, runners=("A", "B"))
        cfg = ClvReferenceConfig(mode="sportsbet_only", devig_method="power")
        resolver = ClvReferenceResolver(cfg)
        result = resolver.resolve(snapshot)

        expected = _devig([1.80, 2.10], method="power")
        assert result["method"] == "single_book:sportsbet"
        assert result["books_used"] == ["sportsbet"]
        assert result["runners"][0]["ref_prob"] == _approx(expected[0])
        assert result["runners"][0]["ref_source"] == "single_book:sportsbet"

    def test_sportsbet_only_raises_when_book_absent(self) -> None:
        snapshot = {
            "runners": [
                {"name": "A", "books": {"tab": 1.82}},
                {"name": "B", "books": {"tab": 2.08}},
            ]
        }
        cfg = ClvReferenceConfig(mode="sportsbet_only")
        resolver = ClvReferenceResolver(cfg)
        with pytest.raises(ValueError, match="sportsbet"):
            resolver.resolve(snapshot)


# ---------------------------------------------------------------------------
# 6. Single-book mode
# ---------------------------------------------------------------------------


class TestSingleBookMode:
    def test_single_book_resolves_named_book(self) -> None:
        books_odds = {
            "sportsbet": {"A": 1.80, "B": 2.10},
            "betr":      {"A": 1.84, "B": 2.05},
        }
        snapshot = _two_runner_snapshot(books_odds, runners=("A", "B"))
        cfg = ClvReferenceConfig(mode="single_book", single_book="betr", devig_method="power")
        resolver = ClvReferenceResolver(cfg)
        result = resolver.resolve(snapshot)

        expected = _devig([1.84, 2.05], method="power")
        assert result["books_used"] == ["betr"]
        assert result["runners"][0]["ref_prob"] == _approx(expected[0])

    def test_single_book_mode_without_single_book_raises(self) -> None:
        books_odds = {"sportsbet": {"A": 1.80, "B": 2.10}}
        snapshot = _two_runner_snapshot(books_odds, runners=("A", "B"))
        cfg = ClvReferenceConfig(mode="single_book", single_book=None)
        resolver = ClvReferenceResolver(cfg)
        with pytest.raises(ValueError, match="single_book must be set"):
            resolver.resolve(snapshot)


# ---------------------------------------------------------------------------
# 7. Empty / malformed input
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_empty_runners_raises(self) -> None:
        resolver = ClvReferenceResolver()
        with pytest.raises(ValueError, match="non-empty"):
            resolver.resolve({"runners": []})

    def test_missing_runners_key_raises(self) -> None:
        resolver = ClvReferenceResolver()
        with pytest.raises(ValueError, match="non-empty"):
            resolver.resolve({})

    def test_not_a_dict_raises(self) -> None:
        resolver = ClvReferenceResolver()
        with pytest.raises(ValueError, match="must be a dict"):
            resolver.resolve("not a dict")  # type: ignore[arg-type]

    def test_runner_missing_name_raises(self) -> None:
        snapshot = {"runners": [{"books": {"sportsbet": 1.80}}, {"books": {"sportsbet": 2.10}}]}
        resolver = ClvReferenceResolver()
        with pytest.raises(ValueError, match="'name'"):
            resolver.resolve(snapshot)

    def test_runner_books_not_dict_raises(self) -> None:
        snapshot = {"runners": [{"name": "A", "books": "bad"}, {"name": "B", "books": "bad"}]}
        resolver = ClvReferenceResolver()
        with pytest.raises(ValueError, match="must be a dict"):
            resolver.resolve(snapshot)

    def test_no_matching_books_raises(self) -> None:
        snapshot = {
            "runners": [
                {"name": "A", "books": {"unknown_book": 1.80}},
                {"name": "B", "books": {"unknown_book": 2.10}},
            ]
        }
        cfg = ClvReferenceConfig(
            mode="multi_book_consensus",
            consensus_books=["sportsbet", "tab"],
            consensus_min_books=1,
        )
        resolver = ClvReferenceResolver(cfg)
        with pytest.raises(ValueError, match="No configured consensus book"):
            resolver.resolve(snapshot)


# ---------------------------------------------------------------------------
# 8. build_default_resolver
# ---------------------------------------------------------------------------


class TestBuildDefaultResolver:
    def test_default_resolver_returns_resolver(self) -> None:
        resolver = build_default_resolver()
        assert isinstance(resolver, ClvReferenceResolver)
        assert resolver._config.mode == "multi_book_consensus"
        assert resolver._config.devig_method == "power"
        assert resolver._config.consensus_weighting == "by_overround"
        assert "sportsbet" in resolver._config.consensus_books
        assert "tab" in resolver._config.consensus_books

    def test_default_resolver_resolves_basic_snapshot(self) -> None:
        books_odds = {
            "sportsbet": {"A": 1.80, "B": 2.10},
            "tab":       {"A": 1.82, "B": 2.08},
            "ladbrokes": {"A": 1.78, "B": 2.12},
        }
        snapshot = _two_runner_snapshot(books_odds, runners=("A", "B"))
        resolver = build_default_resolver()
        result = resolver.resolve(snapshot)
        total = sum(r["ref_prob"] for r in result["runners"])
        assert total == _approx(1.0)


# ---------------------------------------------------------------------------
# 9. Probability sum invariant (parametrised across modes)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,extra",
    [
        ("multi_book_consensus", {}),
        ("sportsbet_only", {}),
        ("single_book", {"single_book": "tab"}),
    ],
)
def test_probs_sum_to_one(mode: str, extra: dict) -> None:
    """Runner probabilities must sum to 1.0 in all modes."""
    books_odds = {
        "sportsbet": {"A": 1.80, "B": 2.10},
        "tab":       {"A": 1.82, "B": 2.08},
        "ladbrokes": {"A": 1.78, "B": 2.12},
    }
    snapshot = _two_runner_snapshot(books_odds, runners=("A", "B"))
    cfg = ClvReferenceConfig(
        mode=mode,  # type: ignore[arg-type]
        consensus_books=["sportsbet", "tab", "ladbrokes"],
        consensus_min_books=1,
        **extra,
    )
    resolver = ClvReferenceResolver(cfg)
    result = resolver.resolve(snapshot)
    total = sum(r["ref_prob"] for r in result["runners"])
    assert math.isclose(total, 1.0, abs_tol=1e-6), f"Probs sum to {total} in mode {mode}"
