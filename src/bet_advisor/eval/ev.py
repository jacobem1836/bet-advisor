"""
Expected value (EV) and edge utilities.

These functions translate a model probability and bookmaker decimal odds into
actionable metrics: expected profit per unit staked, fractional edge, a
minimum edge threshold heuristic, and an uncertainty-aware EV distribution.

All functions are pure (no I/O, no state) and accept scalar inputs unless
otherwise noted.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Core EV / edge
# ---------------------------------------------------------------------------


def expected_value(
    model_prob: float,
    decimal_odds: float,
    stake: float = 1.0,
) -> float:
    """Return the expected profit (in stake units) for a single bet.

    Formula::

        EV = stake * (model_prob * (decimal_odds - 1) - (1 - model_prob))
           = stake * (model_prob * decimal_odds - 1)

    A positive EV means the bet is +EV: over a large sample of identical bets
    the bettor expects to profit.  A single bet may still lose.

    Parameters
    ----------
    model_prob:
        Model's estimated probability of winning (0 < p < 1).
    decimal_odds:
        Bookmaker decimal odds (e.g. 1.90 for roughly evens).
    stake:
        Stake in units. Defaults to 1.0 so the return is EV per unit.

    Returns
    -------
    float
        Expected profit in the same units as ``stake``.

    Examples
    --------
    >>> expected_value(0.58, 1.90)
    0.10200000000000009
    """
    return stake * (model_prob * decimal_odds - 1.0)


def edge(model_prob: float, decimal_odds: float) -> float:
    """Return the fractional edge of a bet.

    The edge is the expected return per unit staked, expressed as a fraction::

        edge = model_prob * decimal_odds - 1

    Equivalent to ``expected_value(model_prob, decimal_odds, stake=1.0)``.

    A value of 0.03 means a 3% expected return per unit staked -- the
    recommended minimum for AFL MVP bets (see ``min_edge_threshold``).

    Parameters
    ----------
    model_prob:
        Model probability of winning.
    decimal_odds:
        Bookmaker decimal odds.

    Returns
    -------
    float
        Fractional edge (positive = value bet, negative = bad bet).
    """
    return model_prob * decimal_odds - 1.0


# ---------------------------------------------------------------------------
# Edge threshold
# ---------------------------------------------------------------------------


def min_edge_threshold(
    model_uncertainty: float = 0.0,
    vig: float = 0.05,
    default: float = 0.03,
) -> float:
    """Return a minimum edge threshold adjusted for model uncertainty and vig.

    The research document (§2) recommends 3% as the MVP baseline.  This
    function applies an upward adjustment when:

    - ``model_uncertainty`` is high (wide 95% CI on the model probability):
      more uncertainty means more risk that the apparent edge is estimation
      error rather than real edge.
    - ``vig`` is high: a thicker overround means the book's edge is already
      embedded in the price; a larger edge buffer is needed to be confident
      the model is genuinely ahead.

    Heuristic adjustments (additive, not fitted)
    --------------------------------------------
    * +0.5 * model_uncertainty  -- each 1pp of CI half-width raises the bar
      by 0.5pp.  E.g. a 95% CI of ±0.10 adds 5pp to the threshold.
    * +0.2 * max(0, vig - 0.05) -- each 1pp of vig above 5% adds 0.2pp.
      A 10% vig market raises the threshold by 1pp.

    These values are intentionally conservative starting points.  Revisit
    once 300+ bets with confirmed +CLV have been logged.

    Parameters
    ----------
    model_uncertainty:
        Half-width of the 95% credible interval on the model probability.
        E.g. 0.05 means the model prob is ± 5pp at 95% confidence.
        Defaults to 0.0 (point estimate, no uncertainty adjustment).
    vig:
        Bookmaker overround / vig as a fraction (e.g. 0.07 for 7%).
        Defaults to 0.05 (typical AFL Sportsbet H2H margin).
    default:
        Baseline minimum edge. Defaults to 0.03 (3%).

    Returns
    -------
    float
        Adjusted minimum edge threshold.

    Examples
    --------
    >>> min_edge_threshold()
    0.03
    >>> min_edge_threshold(model_uncertainty=0.10, vig=0.10)
    0.08999999999999999
    """
    vig_adjustment = 0.2 * max(0.0, vig - 0.05)
    uncertainty_adjustment = 0.5 * model_uncertainty
    return default + vig_adjustment + uncertainty_adjustment


# ---------------------------------------------------------------------------
# Uncertainty-aware EV
# ---------------------------------------------------------------------------


def ev_with_uncertainty(
    model_prob_samples: np.ndarray,
    decimal_odds: float,
) -> dict[str, float]:
    """Return mean, 5th-percentile, and 95th-percentile EV from a sample array.

    When the model probability is uncertain (e.g. drawn from a Beta posterior
    or a bootstrap distribution), treating it as a point estimate understates
    the true range of outcomes.  This function computes the EV distribution
    implied by the sample array and returns summary statistics that reflect
    model uncertainty.

    If the 5th-percentile EV is still positive, the edge is robust even under
    pessimistic model assumptions.

    Parameters
    ----------
    model_prob_samples:
        1-D array of probability samples, e.g. from
        ``scipy.stats.beta.rvs(a, b, size=10_000)`` or a Bootstrap draw.
        Values should be in (0, 1).
    decimal_odds:
        Bookmaker decimal odds.

    Returns
    -------
    dict with keys:
        ``mean_ev``   -- expected EV across the sample
        ``p5_ev``     -- 5th-percentile EV (pessimistic scenario)
        ``p95_ev``    -- 95th-percentile EV (optimistic scenario)
        ``p_positive`` -- fraction of samples with EV > 0

    Examples
    --------
    >>> import numpy as np
    >>> samples = np.full(1000, 0.55)
    >>> ev_with_uncertainty(samples, 1.95)
    {'mean_ev': 0.0725, 'p5_ev': 0.0725, 'p95_ev': 0.0725, 'p_positive': 1.0}
    """
    ev_samples = model_prob_samples * decimal_odds - 1.0
    return {
        "mean_ev": float(np.mean(ev_samples)),
        "p5_ev": float(np.percentile(ev_samples, 5)),
        "p95_ev": float(np.percentile(ev_samples, 95)),
        "p_positive": float(np.mean(ev_samples > 0)),
    }
