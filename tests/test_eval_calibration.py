"""Tests for bet_advisor.eval.calibration -- scoring metrics and calibrators."""

from __future__ import annotations

import numpy as np
import pytest

from bet_advisor.eval.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    brier_score,
    expected_calibration_error,
    log_loss,
    reliability_diagram_data,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _perfect_preds(n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Perfect predictions: prob=1 -> outcome=1, prob=0 -> outcome=0."""
    probs = np.array([1.0] * (n // 2) + [0.0] * (n // 2))
    outcomes = np.array([1.0] * (n // 2) + [0.0] * (n // 2))
    return probs, outcomes


def _uniform_preds(n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """All predictions 0.5, half win."""
    probs = np.full(n, 0.5)
    outcomes = np.array([1.0, 0.0] * (n // 2))
    return probs, outcomes


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------


class TestBrierScore:
    def test_perfect_predictions_zero(self) -> None:
        probs, outcomes = _perfect_preds()
        assert brier_score(probs, outcomes) == pytest.approx(0.0, abs=1e-9)

    def test_all_wrong_worst_case(self) -> None:
        # Predicting 1.0 for all losses and 0.0 for all wins.
        probs = np.array([1.0, 1.0, 0.0, 0.0])
        outcomes = np.array([0.0, 0.0, 1.0, 1.0])
        assert brier_score(probs, outcomes) == pytest.approx(1.0, abs=1e-9)

    def test_baseline_50_pct(self) -> None:
        # Naive 0.5 on all bets: BS = 0.25.
        probs, outcomes = _uniform_preds(200)
        assert brier_score(probs, outcomes) == pytest.approx(0.25, abs=1e-3)

    def test_single_correct_prediction(self) -> None:
        probs = np.array([0.8])
        outcomes = np.array([1.0])
        assert brier_score(probs, outcomes) == pytest.approx((0.8 - 1.0) ** 2, abs=1e-9)


# ---------------------------------------------------------------------------
# Log loss
# ---------------------------------------------------------------------------


class TestLogLoss:
    def test_perfect_predictions_near_zero(self) -> None:
        # prob clipped to [eps, 1-eps], so not exactly 0, but very small.
        probs = np.full(100, 0.9999)
        outcomes = np.ones(100)
        ll = log_loss(probs, outcomes)
        assert ll < 0.01

    def test_all_wrong_high_loss(self) -> None:
        probs = np.full(100, 0.9999)
        outcomes = np.zeros(100)
        ll = log_loss(probs, outcomes)
        assert ll > 5.0

    def test_baseline_50_pct(self) -> None:
        # Log loss at 0.5 on all predictions is ln(2) ~= 0.693.
        probs = np.full(100, 0.5)
        outcomes = np.array([1.0, 0.0] * 50)
        ll = log_loss(probs, outcomes)
        assert ll == pytest.approx(np.log(2), abs=1e-3)

    def test_clipping_prevents_nan(self) -> None:
        probs = np.array([0.0, 1.0])
        outcomes = np.array([1.0, 0.0])
        ll = log_loss(probs, outcomes)
        assert np.isfinite(ll)


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------


class TestExpectedCalibrationError:
    def test_perfect_calibration_near_zero(self) -> None:
        # Build a perfectly calibrated dataset: for each decile bin, fraction
        # of positives equals the midpoint probability.
        rng = np.random.default_rng(42)
        n = 2000
        probs = rng.uniform(0, 1, size=n)
        # Draw outcomes with probability equal to the predicted probability.
        outcomes = (rng.uniform(size=n) < probs).astype(float)
        ece = expected_calibration_error(probs, outcomes, n_bins=10)
        # Allow a generous tolerance given finite sample noise.
        assert ece < 0.05

    def test_overconfident_model_nonzero_ece(self) -> None:
        # Model always predicts 0.9 but only wins 50% of the time.
        probs = np.full(100, 0.9)
        outcomes = np.array([1.0, 0.0] * 50)
        ece = expected_calibration_error(probs, outcomes)
        assert ece > 0.3

    def test_single_bin(self) -> None:
        # All predictions in one bin.
        probs = np.full(50, 0.5)
        outcomes = np.ones(50)
        ece = expected_calibration_error(probs, outcomes, n_bins=10)
        # Bin centre ~0.5, actual freq = 1.0 -> gap = 0.5.
        assert ece == pytest.approx(0.5, abs=0.05)

    def test_non_negative(self) -> None:
        probs, outcomes = _uniform_preds()
        assert expected_calibration_error(probs, outcomes) >= 0.0


# ---------------------------------------------------------------------------
# Reliability diagram data
# ---------------------------------------------------------------------------


class TestReliabilityDiagramData:
    def test_keys_present(self) -> None:
        probs, outcomes = _uniform_preds()
        data = reliability_diagram_data(probs, outcomes)
        assert set(data.keys()) == {"bin_centres", "fraction_positive", "counts"}

    def test_counts_sum_to_n(self) -> None:
        probs, outcomes = _uniform_preds(100)
        data = reliability_diagram_data(probs, outcomes)
        assert sum(data["counts"]) == 100.0

    def test_fractions_in_range(self) -> None:
        probs, outcomes = _uniform_preds()
        data = reliability_diagram_data(probs, outcomes)
        for f in data["fraction_positive"]:
            assert 0.0 <= f <= 1.0

    def test_bin_centres_ordered(self) -> None:
        probs, outcomes = _uniform_preds()
        data = reliability_diagram_data(probs, outcomes)
        centres = data["bin_centres"]
        assert centres == sorted(centres)

    def test_empty_bins_excluded(self) -> None:
        # All probs in [0, 0.1] -- only the first bin should be non-empty.
        probs = np.full(50, 0.05)
        outcomes = np.zeros(50)
        data = reliability_diagram_data(probs, outcomes, n_bins=10)
        assert len(data["bin_centres"]) == 1


# ---------------------------------------------------------------------------
# PlattCalibrator
# ---------------------------------------------------------------------------


class TestPlattCalibrator:
    def _miscalibrated_data(
        self, n: int = 200, seed: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate a miscalibrated model: raw probs skewed too high."""
        rng = np.random.default_rng(seed)
        train_probs = rng.beta(3, 1, n)  # skewed toward 1
        train_outcomes = (rng.uniform(size=n) < train_probs * 0.6).astype(float)
        test_probs = rng.beta(3, 1, 50)
        test_outcomes = (rng.uniform(size=50) < test_probs * 0.6).astype(float)
        return train_probs, train_outcomes, test_probs, test_outcomes

    def test_fit_transform_reduces_ece(self) -> None:
        train_probs, train_outcomes, test_probs, test_outcomes = self._miscalibrated_data()
        ece_before = expected_calibration_error(test_probs, test_outcomes)

        cal = PlattCalibrator()
        cal.fit(train_probs, train_outcomes)
        cal_probs = cal.transform(test_probs)

        ece_after = expected_calibration_error(cal_probs, test_outcomes)
        assert ece_after <= ece_before

    def test_transform_before_fit_raises(self) -> None:
        cal = PlattCalibrator()
        with pytest.raises(RuntimeError, match="fit"):
            cal.transform(np.array([0.5, 0.6]))

    def test_save_load_roundtrip(self, tmp_path) -> None:
        train_probs, train_outcomes, _, _ = self._miscalibrated_data()
        cal = PlattCalibrator()
        cal.fit(train_probs, train_outcomes)

        path = tmp_path / "platt.pkl"
        cal.save(path)
        cal2 = PlattCalibrator.load(path)

        test_probs = np.array([0.3, 0.5, 0.7])
        assert cal.transform(test_probs) == pytest.approx(cal2.transform(test_probs), abs=1e-9)

    def test_output_shape(self) -> None:
        rng = np.random.default_rng(0)
        train_probs = rng.uniform(0, 1, 100)
        train_outcomes = (rng.uniform(size=100) < train_probs).astype(float)
        cal = PlattCalibrator()
        cal.fit(train_probs, train_outcomes)
        out = cal.transform(train_probs[:10])
        assert out.shape == (10,)


# ---------------------------------------------------------------------------
# IsotonicCalibrator
# ---------------------------------------------------------------------------


class TestIsotonicCalibrator:
    def test_fit_transform_shape(self) -> None:
        rng = np.random.default_rng(1)
        probs = rng.uniform(0, 1, 200)
        outcomes = (rng.uniform(size=200) < probs).astype(float)
        cal = IsotonicCalibrator()
        cal.fit(probs, outcomes)
        out = cal.transform(probs[:20])
        assert out.shape == (20,)

    def test_transform_before_fit_raises(self) -> None:
        cal = IsotonicCalibrator()
        with pytest.raises(RuntimeError, match="fit"):
            cal.transform(np.array([0.5]))

    def test_save_load_roundtrip(self, tmp_path) -> None:
        rng = np.random.default_rng(2)
        probs = rng.uniform(0, 1, 100)
        outcomes = (rng.uniform(size=100) < probs).astype(float)
        cal = IsotonicCalibrator()
        cal.fit(probs, outcomes)

        path = tmp_path / "iso.pkl"
        cal.save(path)
        cal2 = IsotonicCalibrator.load(path)

        test_probs = np.array([0.3, 0.5, 0.7])
        assert cal.transform(test_probs) == pytest.approx(cal2.transform(test_probs), abs=1e-9)

    def test_output_in_range(self) -> None:
        rng = np.random.default_rng(3)
        probs = rng.uniform(0, 1, 200)
        outcomes = (rng.uniform(size=200) < probs).astype(float)
        cal = IsotonicCalibrator()
        cal.fit(probs, outcomes)
        out = cal.transform(probs)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)
