"""
Kelly criterion and stake-sizing utilities.

Implements full Kelly, fractional Kelly, capped Kelly, a numerical portfolio
Kelly solver for simultaneous bets, and a dispatcher for stake recommendation.

Background
----------
Kelly (1956) proved that maximising the expected log of wealth maximises the
long-run bankroll growth rate.  For a binary bet with probability ``p`` at
decimal odds ``d``::

    f* = (p * (d - 1) - (1 - p)) / (d - 1)
       = (p * d - 1) / (d - 1)

Full Kelly is mathematically optimal but practically dangerous:

1. Any overestimation of ``p`` causes overbetting, which degrades long-run
   growth faster than underbetting by the same amount (the growth function
   is concave -- Thorp's result).
2. Full Kelly produces 30--50% drawdowns routinely even on a genuinely
   positive-edge model.
3. With an unvalidated model, estimation error can easily exceed the edge,
   making full Kelly a ruin pathway.

**Recommended MVP default:** quarter Kelly (``fraction=0.25``) with a 5%
bankroll cap per bet.  Graduate to half Kelly only after 300+ bets with
confirmed positive CLV.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]

from bet_advisor.eval.ev import edge as _edge

# ---------------------------------------------------------------------------
# Core Kelly formulas
# ---------------------------------------------------------------------------


def full_kelly(prob: float, decimal_odds: float) -> float:
    """Return the full Kelly fraction of bankroll to stake.

    Formula::

        f* = (prob * (decimal_odds - 1) - (1 - prob)) / (decimal_odds - 1)
           = (prob * decimal_odds - 1) / (decimal_odds - 1)

    Clamped to [0, 1].  Negative Kelly (when ``edge < 0``) means the bet
    has no value; the function returns 0.

    Parameters
    ----------
    prob:
        Model probability of winning.
    decimal_odds:
        Bookmaker decimal odds (must be > 1).

    Returns
    -------
    float
        Fraction of bankroll to stake, in [0, 1].
    """
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    f = (prob * b - (1.0 - prob)) / b
    return max(0.0, min(1.0, f))


def fractional_kelly(
    prob: float,
    decimal_odds: float,
    fraction: float = 0.25,
) -> float:
    """Return a fractional Kelly stake.

    Applies a scaling factor ``fraction`` to the full Kelly fraction, then
    clamps to [0, 1].  Quarter Kelly (``fraction=0.25``) is the recommended
    MVP default: it reduces expected drawdown to roughly one quarter of the
    full Kelly level at the cost of ~44% of the theoretical growth rate.

    Parameters
    ----------
    prob:
        Model probability of winning.
    decimal_odds:
        Bookmaker decimal odds.
    fraction:
        Scaling factor. 0.25 = quarter Kelly (default), 0.5 = half Kelly.

    Returns
    -------
    float
        Scaled Kelly fraction, in [0, 1].
    """
    return max(0.0, min(1.0, fraction * full_kelly(prob, decimal_odds)))


def capped_kelly(
    prob: float,
    decimal_odds: float,
    cap: float = 0.05,
    fraction: float = 0.25,
) -> float:
    """Return a fractional Kelly fraction with a hard maximum cap.

    First computes ``fractional_kelly(prob, decimal_odds, fraction)``, then
    applies ``min(result, cap)``.  The 5% cap prevents any single bet from
    exceeding 5% of bankroll regardless of how large Kelly recommends.

    Parameters
    ----------
    prob:
        Model probability of winning.
    decimal_odds:
        Bookmaker decimal odds.
    cap:
        Maximum fraction of bankroll per bet. Defaults to 0.05 (5%).
    fraction:
        Kelly fraction multiplier. Defaults to 0.25 (quarter Kelly).

    Returns
    -------
    float
        Capped Kelly fraction, in [0, cap].
    """
    fk = fractional_kelly(prob, decimal_odds, fraction)
    return min(fk, cap)


# ---------------------------------------------------------------------------
# Portfolio Kelly
# ---------------------------------------------------------------------------


def portfolio_kelly(bets: list[dict[str, Any]]) -> list[float]:
    """Numerically solve Kelly fractions for a portfolio of simultaneous bets.

    Maximises the expected log-bankroll growth across all ``2^n`` outcome
    combinations of ``n`` independent bets.  The optimisation is via
    ``scipy.optimize.minimize`` (L-BFGS-B) with each fraction bounded in
    [0, 0.05] (matching the ``capped_kelly`` per-bet cap).

    Assumptions
    -----------
    * Bets are treated as statistically independent.  Correlation between
      bets on the same AFL round is a real phenomenon (shared weather,
      umpiring variance, scheduling effects), but modelling a full Gaussian
      copula is deferred to a later phase.  For the MVP, users should apply
      a round-level exposure cap (e.g. 10% of bankroll total) as a safeguard
      against unmodelled correlation.
    * The numerical objective scales as O(2^n), so this function is practical
      for n <= 15.  For larger portfolios, use Monte Carlo sampling of the
      joint distribution instead.

    Parameters
    ----------
    bets:
        List of dicts, each with keys:
          - ``prob`` (float): model win probability
          - ``decimal_odds`` (float): bookmaker decimal odds
          - ``correlation_id`` (str, optional): for future grouped-bet logic;
            currently unused but stored for auditability.

    Returns
    -------
    list[float]
        Optimal Kelly fractions, one per bet, in [0, 0.05].

    Examples
    --------
    >>> portfolio_kelly([
    ...     {"prob": 0.55, "decimal_odds": 1.90},
    ...     {"prob": 0.52, "decimal_odds": 2.10},
    ... ])
    [0.04..., 0.0...]
    """
    n = len(bets)
    if n == 0:
        return []

    probs = [float(b["prob"]) for b in bets]
    odds = [float(b["decimal_odds"]) for b in bets]

    def neg_expected_log_wealth(f: np.ndarray) -> float:
        total = 0.0
        for outcome in range(2**n):
            p_outcome = 1.0
            growth = 1.0
            for i in range(n):
                won = bool((outcome >> i) & 1)
                if won:
                    p_outcome *= probs[i]
                    growth *= 1.0 + f[i] * (odds[i] - 1.0)
                else:
                    p_outcome *= 1.0 - probs[i]
                    growth *= 1.0 - f[i]
            total += p_outcome * np.log(max(growth, 1e-10))
        return -total

    bounds = [(0.0, 0.05)] * n
    x0 = np.array([0.01] * n)
    result = minimize(neg_expected_log_wealth, x0, method="L-BFGS-B", bounds=bounds)
    return result.x.tolist()


# ---------------------------------------------------------------------------
# Stake recommendation dispatcher
# ---------------------------------------------------------------------------


def stake_recommendation(
    prob: float,
    decimal_odds: float,
    bankroll: float,
    mode: str = "flat",
    **kwargs: Any,
) -> dict[str, float | str]:
    """Return a recommended stake in currency units, plus diagnostic fields.

    Modes
    -----
    ``"flat"``
        1% of bankroll per bet, regardless of edge.  The safest default
        for the first 300 bets of an unvalidated model.
    ``"quarter_kelly"``
        Quarter Kelly (``fraction=0.25``) with no cap.
    ``"capped_kelly"``
        Quarter Kelly with a hard 5% bankroll cap.  Default ``cap`` can be
        overridden via ``kwargs``.

    Parameters
    ----------
    prob:
        Model probability of winning.
    decimal_odds:
        Bookmaker decimal odds.
    bankroll:
        Current bankroll in currency units.
    mode:
        Staking mode. One of ``"flat"``, ``"quarter_kelly"``, ``"capped_kelly"``.
    **kwargs:
        Extra keyword arguments forwarded to the underlying Kelly function,
        e.g. ``fraction``, ``cap``.

    Returns
    -------
    dict with keys:
        ``stake``        -- recommended stake in currency units
        ``mode``         -- mode used
        ``prob``         -- input probability
        ``odds``         -- input decimal odds
        ``edge``         -- fractional edge (model_prob * odds - 1)
        ``ev``           -- expected value per unit staked
        ``bankroll_pct`` -- stake as a fraction of bankroll

    Raises
    ------
    ValueError
        For unknown ``mode`` strings.
    """
    ev = _edge(prob, decimal_odds)

    if mode == "flat":
        fraction = 0.01
    elif mode == "quarter_kelly":
        frac = kwargs.get("fraction", 0.25)
        fraction = fractional_kelly(prob, decimal_odds, fraction=frac)
    elif mode == "capped_kelly":
        cap = kwargs.get("cap", 0.05)
        frac = kwargs.get("fraction", 0.25)
        fraction = capped_kelly(prob, decimal_odds, cap=cap, fraction=frac)
    else:
        raise ValueError(
            f"Unknown stake mode {mode!r}. Choose from: 'flat', 'quarter_kelly', 'capped_kelly'."
        )

    stake = fraction * bankroll

    return {
        "stake": stake,
        "mode": mode,
        "prob": prob,
        "odds": decimal_odds,
        "edge": ev,
        "ev": ev,
        "bankroll_pct": fraction,
    }
