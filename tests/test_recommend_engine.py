"""
Tests for the recommendation engine.

All storage interactions use in-memory SQLite databases.
No HTTP calls are made.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from bet_advisor.recommend.engine import (
    Recommendation,
    RecommendationConfig,
    RecommendationEngine,
    UntrainedModelError,
)
from bet_advisor.storage.sqlite_store import SQLiteStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeTrainedModel:
    """Deterministic stub model that is 'trained'."""

    @property
    def is_trained(self) -> bool:
        return True

    @property
    def version_hash(self) -> str:
        return "fake-v1"

    def predict_over_under_prob(self, X: Any, line: float, calibrate: bool = True) -> list:
        return [0.6] * len(X)


class _FakeUntrainedModel:
    """Stub model that is NOT trained."""

    @property
    def is_trained(self) -> bool:
        return False

    @property
    def version_hash(self) -> str:
        return "fake-untrained"

    def predict_over_under_prob(self, X: Any, line: float, calibrate: bool = True) -> list:
        return []


@pytest.fixture()
def tmp_sqlite(tmp_path: Path) -> SQLiteStore:
    """In-memory SQLiteStore for tests."""
    db_path = tmp_path / "test.db"
    store = SQLiteStore(db_path)
    store.connect()
    # Add a test event so odds_snapshots FK is satisfied
    store.upsert_event(
        event_id="evt-001",
        sport_key="aussierules_afl",
        sport_title="AFL",
        commence_time="2026-05-20T19:30:00",
        home_team="Collingwood",
        away_team="Hawthorn",
        completed=False,
    )
    yield store
    store.close()


@pytest.fixture()
def fake_duckdb() -> MagicMock:
    mock = MagicMock()
    mock.query.return_value = []
    return mock


@pytest.fixture()
def default_config() -> RecommendationConfig:
    return RecommendationConfig(
        bankroll=1000.0,
        stake_mode="flat",
        flat_pct=0.01,
        min_edge=0.03,
    )


@pytest.fixture()
def engine(
    tmp_sqlite: SQLiteStore,
    fake_duckdb: MagicMock,
    default_config: RecommendationConfig,
) -> RecommendationEngine:
    model = _FakeTrainedModel()
    return RecommendationEngine(
        model=model,
        sqlite_store=tmp_sqlite,
        duckdb_store=fake_duckdb,
        config=default_config,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_h2h_lines(
    event_id: str = "evt-001",
    bookmaker: str = "sportsbet",
    home_prob: float = 0.6,
    away_prob: float = 0.4,
    overround: float = 0.05,
) -> list[dict]:
    """Build a two-runner H2H market_lines list from fair probs + overround."""
    # Inflate odds by overround so bookmaker margin is present
    raw_home = 1.0 / (home_prob * (1 + overround))
    raw_away = 1.0 / (away_prob * (1 + overround))
    return [
        {
            "market": "h2h",
            "bookmaker": bookmaker,
            "runner": "Collingwood",
            "decimal_odds": round(raw_home, 2),
            "line": None,
            "model_prob": 0.65,  # model thinks home has 65% (vs 60% devigged)
            "model_prob_low": 0.60,
            "model_prob_high": 0.70,
        },
        {
            "market": "h2h",
            "bookmaker": bookmaker,
            "runner": "Hawthorn",
            "decimal_odds": round(raw_away, 2),
            "line": None,
            "model_prob": 0.35,
            "model_prob_low": 0.30,
            "model_prob_high": 0.40,
        },
    ]


# ---------------------------------------------------------------------------
# score_market tests
# ---------------------------------------------------------------------------


class TestScoreMarket:
    def test_returns_only_positive_edge_recs(self, engine: RecommendationEngine) -> None:
        lines = _make_h2h_lines(home_prob=0.6, away_prob=0.4)
        recs = engine.score_market("evt-001", lines, {})
        # Only runners where model_prob * odds - 1 >= 0.03 should be returned
        for rec in recs:
            assert rec.edge >= engine._config.min_edge

    def test_ev_calculation_correct(self, engine: RecommendationEngine) -> None:
        """EV == model_prob * decimal_odds - 1."""
        lines = _make_h2h_lines()
        recs = engine.score_market("evt-001", lines, {})
        for rec in recs:
            expected_ev = rec.model_prob * rec.decimal_odds - 1.0
            assert abs(rec.ev_units - expected_ev) < 1e-9

    def test_kelly_fraction_positive_for_positive_edge(self, engine: RecommendationEngine) -> None:
        lines = _make_h2h_lines()
        recs = engine.score_market("evt-001", lines, {})
        for rec in recs:
            assert rec.kelly_fraction >= 0.0

    def test_flat_stake_uses_flat_pct(self, engine: RecommendationEngine) -> None:
        lines = _make_h2h_lines()
        recs = engine.score_market("evt-001", lines, {})
        bankroll = engine._config.bankroll
        flat_pct = engine._config.flat_pct
        for rec in recs:
            assert rec.stake_mode == "flat"
            assert abs(rec.recommended_stake_units - flat_pct) < 1e-9
            assert abs(rec.recommended_stake_currency - flat_pct * bankroll) < 1e-9

    def test_quarter_kelly_mode(self, tmp_sqlite: SQLiteStore, fake_duckdb: MagicMock) -> None:
        config = RecommendationConfig(
            bankroll=1000.0,
            stake_mode="quarter_kelly",
            kelly_fraction=0.25,
            min_edge=0.03,
        )
        model = _FakeTrainedModel()
        eng = RecommendationEngine(model, tmp_sqlite, fake_duckdb, config)
        lines = _make_h2h_lines()
        recs = eng.score_market("evt-001", lines, {})
        for rec in recs:
            assert rec.stake_mode == "quarter_kelly"
            # Quarter Kelly <= full Kelly
            assert rec.recommended_stake_units <= rec.kelly_fraction + 1e-9

    def test_no_recs_when_model_prob_below_threshold(self, engine: RecommendationEngine) -> None:
        """If model prob is very low, all bets are negative EV."""
        lines = [
            {
                "market": "h2h",
                "bookmaker": "sportsbet",
                "runner": "Collingwood",
                "decimal_odds": 1.80,
                "line": None,
                "model_prob": 0.40,  # 0.40 * 1.80 - 1 = -0.28 (negative EV)
                "model_prob_low": 0.35,
                "model_prob_high": 0.45,
            },
            {
                "market": "h2h",
                "bookmaker": "sportsbet",
                "runner": "Hawthorn",
                "decimal_odds": 2.10,
                "line": None,
                "model_prob": 0.45,  # 0.45 * 2.10 - 1 = -0.055 (negative EV)
                "model_prob_low": 0.40,
                "model_prob_high": 0.50,
            },
        ]
        recs = engine.score_market("evt-001", lines, {})
        assert recs == []

    def test_devigged_prob_stored_correctly(self, engine: RecommendationEngine) -> None:
        lines = _make_h2h_lines(home_prob=0.6, away_prob=0.4, overround=0.05)
        recs = engine.score_market("evt-001", lines, {})
        for rec in recs:
            # Devigged prob must be in (0, 1)
            assert 0.0 < rec.devigged_market_prob < 1.0

    def test_feature_snapshot_stored(self, engine: RecommendationEngine) -> None:
        features = {"ewm_disposals_mean": 22.3, "is_home": 1}
        lines = _make_h2h_lines()
        recs = engine.score_market("evt-001", lines, features)
        for rec in recs:
            assert rec.feature_snapshot == features

    def test_model_version_set(self, engine: RecommendationEngine) -> None:
        lines = _make_h2h_lines()
        recs = engine.score_market("evt-001", lines, {})
        for rec in recs:
            assert rec.model_version == "fake-v1"

    def test_sorted_by_edge_descending(self, engine: RecommendationEngine) -> None:
        lines = _make_h2h_lines()
        recs = engine.score_market("evt-001", lines, {})
        edges = [r.edge for r in recs]
        assert edges == sorted(edges, reverse=True)

    def test_single_runner_market_skipped(self, engine: RecommendationEngine) -> None:
        """Markets with only one runner cannot be devigged; should be skipped."""
        lines = [
            {
                "market": "h2h",
                "bookmaker": "sportsbet",
                "runner": "Collingwood",
                "decimal_odds": 1.80,
                "line": None,
                "model_prob": 0.65,
                "model_prob_low": 0.60,
                "model_prob_high": 0.70,
            },
        ]
        recs = engine.score_market("evt-001", lines, {})
        assert recs == []

    def test_confidence_tier_strong(self, tmp_sqlite: SQLiteStore, fake_duckdb: MagicMock) -> None:
        """A high-edge bet with low uncertainty should be 'strong'."""
        config = RecommendationConfig(bankroll=1000.0, min_edge=0.03)
        model = _FakeTrainedModel()
        eng = RecommendationEngine(model, tmp_sqlite, fake_duckdb, config)
        lines = [
            {
                "market": "h2h",
                "bookmaker": "sportsbet",
                "runner": "Collingwood",
                "decimal_odds": 1.80,
                "line": None,
                "model_prob": 0.70,  # edge = 0.70 * 1.80 - 1 = 0.26 -- very high
                "model_prob_low": 0.68,  # tight CI
                "model_prob_high": 0.72,
            },
            {
                "market": "h2h",
                "bookmaker": "sportsbet",
                "runner": "Hawthorn",
                "decimal_odds": 2.30,
                "line": None,
                "model_prob": 0.30,
                "model_prob_low": 0.28,
                "model_prob_high": 0.32,
            },
        ]
        recs = eng.score_market("evt-001", lines, {})
        collingwood_recs = [r for r in recs if r.runner == "Collingwood"]
        assert collingwood_recs, "Expected at least one Collingwood rec"
        assert collingwood_recs[0].confidence_tier == "strong"

    def test_confidence_tier_speculative_high_uncertainty(
        self, tmp_sqlite: SQLiteStore, fake_duckdb: MagicMock
    ) -> None:
        """Wide CI should push a bet to speculative tier."""
        config = RecommendationConfig(
            bankroll=1000.0,
            min_edge=0.03,
            speculative_uncertainty_threshold=0.25,
        )
        model = _FakeTrainedModel()
        eng = RecommendationEngine(model, tmp_sqlite, fake_duckdb, config)
        lines = [
            {
                "market": "h2h",
                "bookmaker": "sportsbet",
                "runner": "Collingwood",
                "decimal_odds": 1.80,
                "line": None,
                "model_prob": 0.65,
                "model_prob_low": 0.40,  # very wide CI -- uncertainty_ratio = (0.65-0.40)/0.65 = 0.38
                "model_prob_high": 0.90,
            },
            {
                "market": "h2h",
                "bookmaker": "sportsbet",
                "runner": "Hawthorn",
                "decimal_odds": 2.30,
                "line": None,
                "model_prob": 0.35,
                "model_prob_low": 0.10,
                "model_prob_high": 0.60,
            },
        ]
        recs = eng.score_market("evt-001", lines, {})
        spec_recs = [r for r in recs if r.confidence_tier == "speculative"]
        assert any(r.runner == "Collingwood" for r in spec_recs), (
            f"Expected speculative tier for Collingwood, got {[r.confidence_tier for r in recs]}"
        )

    def test_speculative_stake_halved(
        self, tmp_sqlite: SQLiteStore, fake_duckdb: MagicMock
    ) -> None:
        """Speculative recs should have stake reduced by speculative_stake_cap."""
        config = RecommendationConfig(
            bankroll=1000.0,
            stake_mode="flat",
            flat_pct=0.01,
            min_edge=0.03,
            speculative_uncertainty_threshold=0.25,
            speculative_stake_cap=0.5,
        )
        model = _FakeTrainedModel()
        eng = RecommendationEngine(model, tmp_sqlite, fake_duckdb, config)
        lines = [
            {
                "market": "h2h",
                "bookmaker": "sportsbet",
                "runner": "Collingwood",
                "decimal_odds": 1.80,
                "line": None,
                "model_prob": 0.65,
                "model_prob_low": 0.40,  # wide CI => speculative
                "model_prob_high": 0.90,
            },
            {
                "market": "h2h",
                "bookmaker": "sportsbet",
                "runner": "Hawthorn",
                "decimal_odds": 2.30,
                "line": None,
                "model_prob": 0.35,
                "model_prob_low": 0.10,
                "model_prob_high": 0.60,
            },
        ]
        recs = eng.score_market("evt-001", lines, {})
        for rec in recs:
            if rec.confidence_tier == "speculative":
                assert rec.recommended_stake_units == pytest.approx(
                    config.flat_pct * config.speculative_stake_cap, abs=1e-9
                )


# ---------------------------------------------------------------------------
# apply_exposure_caps tests
# ---------------------------------------------------------------------------


class TestExposureCaps:
    def _make_recs(self, n: int, event_id: str = "evt-001") -> list[Recommendation]:
        recs = []
        for i in range(n):
            recs.append(
                Recommendation(
                    event_id=event_id,
                    market="h2h",
                    runner=f"Team-{i}",
                    bookmaker="sportsbet",
                    decimal_odds=1.90,
                    model_prob=0.58,
                    model_prob_low=0.55,
                    model_prob_high=0.62,
                    devigged_market_prob=0.53,
                    edge=0.05,
                    ev_units=0.05,
                    recommended_stake_units=0.01,
                    recommended_stake_currency=10.0,
                    kelly_fraction=0.08,
                    stake_mode="flat",
                    confidence_tier="standard",
                )
            )
        return recs

    def test_per_event_bet_count_cap(self) -> None:
        config = RecommendationConfig(max_bets_per_event=2, max_bets_per_day=10)
        model = _FakeTrainedModel()
        eng = RecommendationEngine(model, MagicMock(), MagicMock(), config)
        recs = self._make_recs(5, event_id="evt-001")
        result = eng.apply_exposure_caps(recs, existing_today=[])
        assert len(result) <= 2

    def test_daily_bet_count_cap(self) -> None:
        config = RecommendationConfig(max_bets_per_event=10, max_bets_per_day=3)
        model = _FakeTrainedModel()
        eng = RecommendationEngine(model, MagicMock(), MagicMock(), config)
        recs = self._make_recs(5, event_id="evt-001")
        result = eng.apply_exposure_caps(recs, existing_today=[])
        assert len(result) <= 3

    def test_existing_today_reduces_available_slots(self) -> None:
        config = RecommendationConfig(max_bets_per_day=3)
        model = _FakeTrainedModel()
        eng = RecommendationEngine(model, MagicMock(), MagicMock(), config)
        recs = self._make_recs(3, event_id="evt-001")
        existing = [
            {"event_id": "evt-999", "recommended_stake_units": 0.01},
            {"event_id": "evt-999", "recommended_stake_units": 0.01},
        ]
        result = eng.apply_exposure_caps(recs, existing_today=existing)
        # 2 existing + max 1 new
        assert len(result) <= 1

    def test_per_event_exposure_cap(self) -> None:
        config = RecommendationConfig(
            max_exposure_per_event_units=0.025,  # 2.5 units
            max_bets_per_event=10,
            max_bets_per_day=20,
        )
        model = _FakeTrainedModel()
        eng = RecommendationEngine(model, MagicMock(), MagicMock(), config)
        # Each rec is 0.01 units; cap at 0.025 means max 2 recs
        recs = self._make_recs(5, event_id="evt-001")
        result = eng.apply_exposure_caps(recs, existing_today=[])
        total_units = sum(r.recommended_stake_units for r in result)
        assert total_units <= 0.025 + 1e-9


# ---------------------------------------------------------------------------
# persist roundtrip tests
# ---------------------------------------------------------------------------


class TestPersist:
    def test_persist_creates_bets_in_sqlite(
        self, engine: RecommendationEngine, tmp_sqlite: SQLiteStore
    ) -> None:
        rec = Recommendation(
            event_id="evt-001",
            market="h2h",
            runner="Collingwood",
            bookmaker="sportsbet",
            decimal_odds=1.90,
            model_prob=0.58,
            model_prob_low=0.55,
            model_prob_high=0.62,
            devigged_market_prob=0.53,
            edge=0.05,
            ev_units=0.05,
            recommended_stake_units=0.01,
            recommended_stake_currency=10.0,
            kelly_fraction=0.08,
            stake_mode="flat",
            confidence_tier="standard",
        )
        bet_ids = engine.persist([rec])
        assert len(bet_ids) == 1
        bets = tmp_sqlite.query("SELECT * FROM bets WHERE id = ?", (bet_ids[0],))
        assert len(bets) == 1
        assert bets[0]["runner"] == "Collingwood"
        assert bets[0]["price"] == pytest.approx(1.90)
        assert bets[0]["stake"] == pytest.approx(10.0)

    def test_persist_creates_signal_linked_to_bet(
        self, engine: RecommendationEngine, tmp_sqlite: SQLiteStore
    ) -> None:
        rec = Recommendation(
            event_id="evt-001",
            market="h2h",
            runner="Hawthorn",
            bookmaker="tab",
            decimal_odds=2.10,
            model_prob=0.52,
            model_prob_low=0.48,
            model_prob_high=0.56,
            devigged_market_prob=0.47,
            edge=0.092,
            ev_units=0.092,
            recommended_stake_units=0.01,
            recommended_stake_currency=10.0,
            kelly_fraction=0.12,
            stake_mode="flat",
            confidence_tier="standard",
        )
        bet_ids = engine.persist([rec])
        bets = tmp_sqlite.query("SELECT * FROM bets WHERE id = ?", (bet_ids[0],))
        signal_id = bets[0]["signal_id"]
        signals = tmp_sqlite.query("SELECT * FROM signals WHERE id = ?", (signal_id,))
        assert len(signals) == 1
        assert signals[0]["event_id"] == "evt-001"

    def test_persist_multiple_returns_all_ids(
        self, engine: RecommendationEngine, tmp_sqlite: SQLiteStore
    ) -> None:
        recs = []
        for i in range(3):
            recs.append(
                Recommendation(
                    event_id="evt-001",
                    market="h2h",
                    runner=f"Team-{i}",
                    bookmaker="sportsbet",
                    decimal_odds=1.90,
                    model_prob=0.58,
                    model_prob_low=0.55,
                    model_prob_high=0.62,
                    devigged_market_prob=0.53,
                    edge=0.05,
                    ev_units=0.05,
                    recommended_stake_units=0.01,
                    recommended_stake_currency=10.0,
                    kelly_fraction=0.08,
                    stake_mode="flat",
                    confidence_tier="standard",
                )
            )
        bet_ids = engine.persist(recs)
        assert len(bet_ids) == 3
        assert len(set(bet_ids)) == 3  # all distinct


# ---------------------------------------------------------------------------
# generate_for_round tests
# ---------------------------------------------------------------------------


class TestGenerateForRound:
    def test_untrained_model_raises_without_flag(
        self, tmp_sqlite: SQLiteStore, fake_duckdb: MagicMock
    ) -> None:
        model = _FakeUntrainedModel()
        engine = RecommendationEngine(model, tmp_sqlite, fake_duckdb)
        with pytest.raises(UntrainedModelError):
            engine.generate_for_round(round_number=5, allow_untrained=False)

    def test_untrained_model_allowed_with_flag(
        self, tmp_sqlite: SQLiteStore, fake_duckdb: MagicMock
    ) -> None:
        model = _FakeUntrainedModel()
        engine = RecommendationEngine(model, tmp_sqlite, fake_duckdb)
        # Should not raise; returns empty list (no odds in DB)
        recs = engine.generate_for_round(round_number=5, allow_untrained=True)
        assert isinstance(recs, list)

    def test_returns_empty_when_no_odds(
        self, tmp_sqlite: SQLiteStore, fake_duckdb: MagicMock
    ) -> None:
        model = _FakeTrainedModel()
        engine = RecommendationEngine(model, tmp_sqlite, fake_duckdb)
        recs = engine.generate_for_round(round_number=5)
        assert recs == []
