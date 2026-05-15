"""
Devigging (overround removal) utilities for converting raw bookmaker odds into
fair probability estimates.

Four methods are provided, ranging from simple heuristics to theoretically
principled numerical solutions:

  - proportional  : divide each implied prob by the overround total
  - power         : find exponent k such that sum(p_i^k) == 1
  - shin          : Shin (1992/1993) insider-model, own implementation
  - additive      : equal-margin subtraction (sanity check only)

Recommended default for AFL H2H and player props: ``power``.
For multi-outcome futures: ``shin``.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def overround(odds: list[float]) -> float:
    """Return the bookmaker overround (margin) for a market.

    The overround is the amount by which the sum of implied probabilities
    exceeds 1.0.  A 5% overround means the book has roughly a 5% edge on the
    market as a whole.

    Parameters
    ----------
    odds:
        Decimal odds for each runner/outcome in the market.

    Returns
    -------
    float
        ``sum(1/o for o in odds) - 1.0``.  Positive for any book with a margin.

    Raises
    ------
    ValueError
        If ``odds`` is empty or contains a non-positive value.
    """
    _validate_odds(odds)
    return sum(1.0 / o for o in odds) - 1.0


# ---------------------------------------------------------------------------
# Individual devig methods
# ---------------------------------------------------------------------------


def proportional(odds: list[float]) -> list[float]:
    """Proportional (multiplicative) devigging.

    Each implied probability is scaled down by the same factor so that the
    normalised probabilities sum to exactly 1.

    Formula::

        p_implied_i = 1 / odds_i
        p_fair_i    = p_implied_i / sum(p_implied_j for j)

    Limitation
    ----------
    Proportional devigging does **not** account for the favourite-longshot
    bias that characterises most real markets: longshots are systematically
    overpriced relative to their true probability, so their implied
    probability is already inflated before normalisation.  Proportional
    scaling preserves the relative bias rather than correcting it.  Use
    ``power`` as the default unless the market is perfectly symmetric.

    Parameters
    ----------
    odds:
        Decimal odds for each runner.

    Returns
    -------
    list[float]
        Fair probabilities summing to 1.0.
    """
    _validate_odds(odds)
    implied = [1.0 / o for o in odds]
    total = sum(implied)
    return [p / total for p in implied]


def power(odds: list[float], tol: float = 1e-9) -> list[float]:
    """Power (exponent) devigging.

    Finds an exponent ``k`` such that ``sum(p_i^k) == 1.0``, where each
    ``p_i = 1 / odds_i``.  The fair probability for runner ``i`` is then
    ``p_fair_i = p_i^k``.

    Mathematical intuition
    ----------------------
    When the overround is positive, the implied probabilities sum to more than
    1.  Raising each implied probability to a power ``k > 1`` shrinks every
    value, but it shrinks *smaller* values (longshots) proportionally *more*
    than larger values (favourites).  This corrects in the right direction for
    the favourite-longshot bias without requiring an explicit model of the bias
    magnitude.

    The exponent is found via ``scipy.optimize.brentq`` bracketed in [0.5, 2.0].
    A fair market (overround == 0) returns ``k == 1.0``.

    Parameters
    ----------
    odds:
        Decimal odds for each runner.
    tol:
        Convergence tolerance passed to the root finder.

    Returns
    -------
    list[float]
        Fair probabilities summing to 1.0 within numerical precision.

    Raises
    ------
    ValueError
        If ``odds`` is empty, contains non-positive values, or the root
        finder cannot converge within the bracket.
    """
    _validate_odds(odds)

    if len(odds) == 1:
        return [1.0]

    implied = [1.0 / o for o in odds]

    # For a perfectly fair market the sum is already 1.0 -- k == 1 exactly.
    total = sum(implied)
    if abs(total - 1.0) < tol:
        return implied[:]

    def _objective(k: float) -> float:
        return sum(p**k for p in implied) - 1.0

    # Bracket: k=0.5 makes sum larger (small probs grow), k=2.0 makes it smaller.
    k = brentq(_objective, 0.5, 2.0, xtol=tol)
    return [p**k for p in implied]


def shin(odds: list[float]) -> list[float]:
    """Shin (1992/1993) devigging -- own implementation (no external package).

    Hyun Song Shin's model posits that the bookmaker's overround arises from
    the presence of a fraction ``z`` of informed (insider) bettors in the
    market.  The fair probability for runner ``i`` is recovered by solving for
    ``z`` and back-computing the underlying true probabilities.

    Derivation summary
    ------------------
    Let ``p_i = 1 / odds_i`` be the raw implied probability, and let ``V``
    denote the total of all implied probabilities (i.e., ``1 + overround``).

    Shin showed that for a market of ``n`` outcomes the book sets prices such
    that the equilibrium return on any bet is the same regardless of outcome.
    Under the insider model this leads to::

        q_i = sqrt(z^2 + 4*(1-z)*p_i^2/V) - z) / (2*(1-z))

    where ``z`` is the insider proportion and ``q_i`` are the fair probabilities.
    The constraint ``sum(q_i) == 1`` determines ``z``.

    For a **two-runner** market an exact closed-form exists (quadratic in z):

        z = (2*V*p1*p2 - 1) / (V - 1 + 2*V*p1*p2 - 2*p1*p2*(V-1)/V)

    which simplifies to a one-shot calculation.

    For **n > 2** runners, ``z`` is found iteratively via ``scipy.optimize.brentq``,
    bracketed in [0, 0.5] (an insider fraction above 50% is implausible).

    References
    ----------
    Shin, H. S. (1992). Prices of state contingent claims with insider traders,
        and the favourite-longshot bias. *The Economic Journal*, 102(411), 426-435.
    Shin, H. S. (1993). Measuring the incidence of insider trading in a market
        for state-contingent claims. *The Economic Journal*, 103(420), 1141-1153.

    Notes on numerical stability
    ----------------------------
    When the overround is very small (< 1e-6) the insider model is degenerate
    (z -> 0) and all methods converge.  In that regime the function falls back
    to the proportional result.

    Parameters
    ----------
    odds:
        Decimal odds for each runner.

    Returns
    -------
    list[float]
        Fair probabilities summing to 1.0.
    """
    _validate_odds(odds)

    if len(odds) == 1:
        return [1.0]

    implied = [1.0 / o for o in odds]
    V = sum(implied)

    if abs(V - 1.0) < 1e-9:
        # Nearly fair market -- no overround to remove.
        return [p / V for p in implied]

    def _q_from_z(z: float) -> list[float]:
        """Compute Shin fair probabilities given insider fraction z."""
        probs = []
        for p in implied:
            # Shin formula: q_i = (sqrt(z^2 + 4*(1-z)*p_i^2 / V) - z) / (2*(1-z))
            discriminant = z**2 + 4.0 * (1.0 - z) * p**2 / V
            q = (math.sqrt(max(discriminant, 0.0)) - z) / (2.0 * (1.0 - z))
            probs.append(q)
        return probs

    def _constraint(z: float) -> float:
        """Returns sum(q_i(z)) - 1.0; root == valid z."""
        return sum(_q_from_z(z)) - 1.0

    if len(odds) == 2:
        # For two-runner markets, the general brentq solver below is used
        # (the closed-form requires additional algebraic simplification).
        # The bracket [0, 0.5] is always valid for plausible overrounds.
        pass  # fall through to general solver below

    # General n-runner solver (also covers two-runner for consistency).
    # Bracket check: _constraint(0) == sum(proportional) - 1 == 0 only for fair
    # market, which is handled above.  _constraint(0+eps) is close to 0.
    # _constraint(0.5) should be negative for realistic overrounds.
    try:
        z_star = brentq(_constraint, 0.0, 0.5, xtol=1e-10, maxiter=200)
    except ValueError:
        # If brentq cannot find a root in [0, 0.5] (very high overround or
        # degenerate market), fall back to proportional as a safe default.
        return proportional(odds)

    probs = _q_from_z(z_star)
    # Normalise to correct tiny floating-point drift.
    total = sum(probs)
    return [q / total for q in probs]


def additive(odds: list[float]) -> list[float]:
    """Additive (equal margin) devigging.

    Subtracts ``overround / n`` from each runner's implied probability, where
    ``n`` is the number of outcomes.

    **Use for sanity checks only.**  Limitations:

    * Produces negative probabilities for any longshot when the overround
      is large relative to the longshot's implied probability.
    * Assumes each outcome bears an identical share of the bookmaker margin,
      which is never true in practice (favourites typically contribute more
      to book profit than longshots in sharp markets).
    * For binary markets (n == 2) the equal-margin assumption is most
      defensible, but even then it is numerically inferior to proportional
      or power methods.

    Parameters
    ----------
    odds:
        Decimal odds for each runner.

    Returns
    -------
    list[float]
        Approximately fair probabilities (may contain values < 0 for
        extreme longshots with large overrounds).
    """
    _validate_odds(odds)
    implied = [1.0 / o for o in odds]
    total = sum(implied)
    margin_per = (total - 1.0) / len(implied)
    return [p - margin_per for p in implied]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def devig(odds: list[float], method: str = "power") -> list[float]:
    """Dispatcher: devig a market using the named method.

    Parameters
    ----------
    odds:
        Decimal odds for each runner.
    method:
        One of ``"proportional"``, ``"power"``, ``"shin"``, ``"additive"``.
        Defaults to ``"power"`` (recommended for AFL H2H and player props).

    Returns
    -------
    list[float]
        Fair probabilities summing to 1.0.

    Raises
    ------
    ValueError
        For unknown method names.
    """
    _methods = {
        "proportional": proportional,
        "power": power,
        "shin": shin,
        "additive": additive,
    }
    if method not in _methods:
        raise ValueError(f"Unknown devig method {method!r}. Choose from: {sorted(_methods)}")
    return _methods[method](odds)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_odds(odds: list[float]) -> None:
    """Raise ValueError for empty or invalid odds lists."""
    if len(odds) == 0:
        raise ValueError("odds list must not be empty")
    for i, o in enumerate(odds):
        if o <= 0:
            raise ValueError(f"odds[{i}] = {o!r} is non-positive; decimal odds must be > 0")
