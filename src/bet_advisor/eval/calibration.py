"""
Model calibration evaluation and recalibration utilities.

A calibrated model is one where predicted probabilities match observed
frequencies: if the model predicts 70% for a set of outcomes, approximately
70% of those outcomes should occur.  Miscalibration distorts EV calculations
and undermines stake sizing.

Metrics implemented:

  - Brier score: mean squared error between predicted probability and outcome.
  - Log loss: mean negative log-likelihood, penalising overconfident errors.
  - ECE (expected calibration error): weighted mean absolute calibration gap.
  - Reliability diagram data: bin-level accuracy vs confidence.

Calibrators:

  - PlattCalibrator: wraps sklearn LogisticRegression on raw probabilities.
    Preferred for n < 500 games (AFL regime) -- parametric, less overfit risk.
  - IsotonicCalibrator: wraps sklearn IsotonicRegression.
    Better for large samples (player props at scale).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Scoring metrics
# ---------------------------------------------------------------------------


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Return the Brier score (mean squared error) for probabilistic predictions.

    Formula::

        BS = mean((p_i - o_i)^2)

    Lower is better.  For a 50/50 market with naive ``p = 0.5``, BS = 0.25.
    A model with BS > 0.25 on AFL H2H is worse than always predicting 50%.

    The Brier score decomposes into a calibration component (are predictions
    biased?) and a resolution component (does the model discriminate at all?).
    Both matter: a miscalibrated model can still have resolution; a well-
    calibrated model with no resolution is useless for betting.

    Parameters
    ----------
    probs:
        Predicted probabilities, shape (n,), values in [0, 1].
    outcomes:
        Binary outcomes (1 = event occurred, 0 = did not), shape (n,).

    Returns
    -------
    float
        Brier score in [0, 1].
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    return float(np.mean((probs - outcomes) ** 2))


def log_loss(
    probs: np.ndarray,
    outcomes: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """Return the mean negative log-likelihood (log loss).

    Formula::

        LL = -mean(o_i * log(p_i) + (1 - o_i) * log(1 - p_i))

    Probabilities are clipped to [eps, 1 - eps] to avoid log(0).

    Log loss penalises overconfident wrong predictions more heavily than Brier
    score, making it a stronger signal for detecting when the model is
    assigning very high probabilities to the wrong outcomes.

    Parameters
    ----------
    probs:
        Predicted probabilities, shape (n,), clipped to [eps, 1 - eps].
    outcomes:
        Binary outcomes (1 = event occurred, 0 = did not), shape (n,).
    eps:
        Clipping bound to prevent log(0). Defaults to 1e-15.

    Returns
    -------
    float
        Mean log loss (non-negative; lower is better).
    """
    probs = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
    outcomes = np.asarray(outcomes, dtype=float)
    return float(-np.mean(outcomes * np.log(probs) + (1.0 - outcomes) * np.log(1.0 - probs)))


def expected_calibration_error(
    probs: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Return the Expected Calibration Error (ECE).

    ECE is the weighted mean absolute deviation between predicted confidence
    and empirical accuracy across probability bins::

        ECE = sum_bins( |bin| / N * |avg_pred - frac_pos| )

    Bins are uniform over [0, 1].  Empty bins are excluded from the sum.

    Research trigger: recalibrate when ECE > 0.015.  Pause model when
    ECE > 0.02 after a recalibration attempt.

    Parameters
    ----------
    probs:
        Predicted probabilities, shape (n,).
    outcomes:
        Binary outcomes, shape (n,).
    n_bins:
        Number of uniform probability bins. Defaults to 10 (deciles).

    Returns
    -------
    float
        ECE in [0, 1]. 0 = perfect calibration.
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    n = len(probs)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        # Include the upper edge in the last bin.
        if hi == 1.0:
            mask = mask | (probs == 1.0)
        count = mask.sum()
        if count == 0:
            continue
        avg_pred = probs[mask].mean()
        frac_pos = outcomes[mask].mean()
        ece += (count / n) * abs(avg_pred - frac_pos)
    return float(ece)


def reliability_diagram_data(
    probs: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> dict[str, list[float]]:
    """Return binned data for rendering a reliability (calibration) diagram.

    A reliability diagram plots mean predicted probability against the
    empirical fraction of positive outcomes for each bin.  A perfectly
    calibrated model lies on the diagonal (predicted == actual).

    This function returns the raw data; rendering is intentionally left to
    the caller (matplotlib in a notebook, or a later reporting phase).

    Parameters
    ----------
    probs:
        Predicted probabilities, shape (n,).
    outcomes:
        Binary outcomes, shape (n,).
    n_bins:
        Number of uniform bins. Defaults to 10.

    Returns
    -------
    dict with keys:
        ``bin_centres``       -- midpoint of each non-empty bin
        ``fraction_positive`` -- empirical win rate in each bin
        ``counts``            -- number of samples in each bin
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres: list[float] = []
    fractions: list[float] = []
    counts: list[float] = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if hi == 1.0:
            mask = mask | (probs == 1.0)
        count = int(mask.sum())
        if count == 0:
            continue
        centres.append(float((lo + hi) / 2.0))
        fractions.append(float(outcomes[mask].mean()))
        counts.append(float(count))
    return {
        "bin_centres": centres,
        "fraction_positive": fractions,
        "counts": counts,
    }


# ---------------------------------------------------------------------------
# Calibrator classes
# ---------------------------------------------------------------------------


class PlattCalibrator:
    """Post-hoc Platt scaling calibrator.

    Fits a logistic regression on the raw model probabilities (treated as
    single features) to produce calibrated probabilities.  Platt scaling is
    recommended for AFL markets where calibration samples are small
    (n < 500 games per season).

    The logistic regression is fit on the raw probability values, not on
    log-odds, because the raw probabilities already encode the model's
    discrimination signal.

    Usage::

        cal = PlattCalibrator()
        cal.fit(raw_probs_train, outcomes_train)
        calibrated = cal.transform(raw_probs_test)
        cal.save("platt_h2h.pkl")

        cal2 = PlattCalibrator.load("platt_h2h.pkl")
    """

    def __init__(self) -> None:
        self._lr: LogisticRegression | None = None

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> "PlattCalibrator":
        """Fit the logistic regression calibrator.

        Parameters
        ----------
        probs:
            Raw model probabilities, shape (n,).
        outcomes:
            Binary outcomes, shape (n,).

        Returns
        -------
        self (for chaining).
        """
        probs = np.asarray(probs, dtype=float).reshape(-1, 1)
        outcomes = np.asarray(outcomes, dtype=float)
        self._lr = LogisticRegression(C=1e10, solver="lbfgs")
        self._lr.fit(probs, outcomes)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities.

        Parameters
        ----------
        probs:
            Raw model probabilities, shape (n,).

        Returns
        -------
        np.ndarray
            Calibrated probabilities, shape (n,), values in (0, 1).

        Raises
        ------
        RuntimeError
            If ``fit`` has not been called.
        """
        if self._lr is None:
            raise RuntimeError("PlattCalibrator.fit() must be called before transform().")
        x = np.asarray(probs, dtype=float).reshape(-1, 1)
        return self._lr.predict_proba(x)[:, 1]

    def save(self, path: str | Path) -> None:
        """Serialise the fitted calibrator to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self._lr, f)

    @classmethod
    def load(cls, path: str | Path) -> "PlattCalibrator":
        """Load a previously saved calibrator from a pickle file."""
        instance = cls()
        with open(path, "rb") as f:
            instance._lr = pickle.load(f)
        return instance


class IsotonicCalibrator:
    """Post-hoc isotonic regression calibrator.

    Fits a monotone non-parametric mapping from raw probabilities to
    empirical win rates.  More flexible than Platt scaling but requires
    larger calibration sets (500+ samples) to avoid overfitting.

    Recommended for player-prop markets once sufficient historical data
    has accumulated.

    Usage::

        cal = IsotonicCalibrator()
        cal.fit(raw_probs_train, outcomes_train)
        calibrated = cal.transform(raw_probs_test)
        cal.save("isotonic_props.pkl")
    """

    def __init__(self) -> None:
        self._iso: IsotonicRegression | None = None

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> "IsotonicCalibrator":
        """Fit the isotonic regression calibrator.

        Parameters
        ----------
        probs:
            Raw model probabilities, shape (n,).
        outcomes:
            Binary outcomes, shape (n,).

        Returns
        -------
        self (for chaining).
        """
        probs = np.asarray(probs, dtype=float)
        outcomes = np.asarray(outcomes, dtype=float)
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._iso.fit(probs, outcomes)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities.

        Parameters
        ----------
        probs:
            Raw model probabilities, shape (n,).

        Returns
        -------
        np.ndarray
            Calibrated probabilities, shape (n,), values in [0, 1].

        Raises
        ------
        RuntimeError
            If ``fit`` has not been called.
        """
        if self._iso is None:
            raise RuntimeError("IsotonicCalibrator.fit() must be called before transform().")
        return self._iso.predict(np.asarray(probs, dtype=float))

    def save(self, path: str | Path) -> None:
        """Serialise the fitted calibrator to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self._iso, f)

    @classmethod
    def load(cls, path: str | Path) -> "IsotonicCalibrator":
        """Load a previously saved calibrator from a pickle file."""
        instance = cls()
        with open(path, "rb") as f:
            instance._iso = pickle.load(f)
        return instance
