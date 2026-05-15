"""
Closing Line Value (CLV) computation and aggregation utilities.

CLV measures whether a bet was placed at better odds than where the market
finally settled.  It is the primary signal for betting process quality because
it assesses edge identification independently of outcomes (win/loss).

Positive CLV means the bettor received longer odds than the fair closing price,
i.e. they beat the market's final estimate of true probability.

Key functions:

  - closing_line_value:         single two-runner H2H CLV
  - closing_line_value_market:  full market form (n runners)
  - aggregate_clv:              summary stats over a DataFrame of bets
  - clv_significance:           t-test + bootstrap CI on whether mean CLV > 0

References
----------
Buchdahl, J. (via Pinnacle Odds Dropper): "Closing Line Value demystified"
Unabated (2024): "Getting precise about closing line value"
Sports AI Dev (2024): "CLV and AI model performance"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

from bet_advisor.eval.devig import devig as _devig

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# Single-bet CLV
# ---------------------------------------------------------------------------


def closing_line_value(
    bet_odds: float,
    closing_odds: float,
    closing_opp_odds: float,
    devig_method: str = "power",
) -> float:
    """Compute CLV for a single two-runner H2H market.

    CLV is expressed in probability-space: the difference (in percentage
    points, as a decimal) between the devigged closing fair probability
    and the devigged fair probability implied by the bet odds.

    A positive return means the bettor received longer odds than the fair
    closing price -- they beat the closing line.

    Formula::

        bet_fair_prob     = devig([bet_odds, bet_opp_odds])[0]   (if available)
        closing_fair_prob = devig([closing_odds, closing_opp_odds])[0]

        CLV = closing_fair_prob - bet_implied_prob

    Because we often do not have the opposing odds at bet placement time,
    this simplified version uses the raw implied probability for the bet
    side (1 / bet_odds) and the devigged closing probability for comparison.
    This matches the Unabated convention.

    Buchdahl alternative::

        CLV = (closing_devigged_odds / bet_odds) - 1

    Both conventions produce a positive value when the bettor beat the close.
    This function uses the probability-space form.

    Parameters
    ----------
    bet_odds:
        Decimal odds received at placement.
    closing_odds:
        Closing decimal odds for the bettor's side.
    closing_opp_odds:
        Closing decimal odds for the opposing side (required to devig closing).
    devig_method:
        Devigging method for the closing line. Defaults to ``"power"``.

    Returns
    -------
    float
        CLV as a probability difference.  Positive = beat the close.
        E.g. 0.025 means +2.5 percentage points of devigged probability.
    """
    if bet_odds <= 1.0:
        raise ValueError(f"bet_odds must be > 1.0, got {bet_odds!r}")
    if closing_odds <= 1.0:
        raise ValueError(f"closing_odds must be > 1.0, got {closing_odds!r}")
    if closing_opp_odds <= 1.0:
        raise ValueError(f"closing_opp_odds must be > 1.0, got {closing_opp_odds!r}")

    # Devig the closing line to get the fair closing probability.
    closing_fair_probs = _devig([closing_odds, closing_opp_odds], method=devig_method)
    closing_fair_prob = closing_fair_probs[0]

    # Use raw implied probability for bet side (we may not have opp odds at placement).
    bet_implied_prob = 1.0 / bet_odds

    # Positive = bettor's implied prob is lower (longer odds) than closing fair prob.
    # Equivalently: bettor received better value than the market settled on.
    return closing_fair_prob - bet_implied_prob


def closing_line_value_market(
    bet_odds: float,
    bet_runner_idx: int,
    market_close_odds: list[float],
    devig_method: str = "power",
) -> float:
    """Compute CLV for a bet in an n-runner market.

    Uses the full closing market odds to devig and extract the fair closing
    probability for the betted runner.

    Parameters
    ----------
    bet_odds:
        Decimal odds received at placement for the betted runner.
    bet_runner_idx:
        Index of the betted runner in ``market_close_odds``.
    market_close_odds:
        Closing decimal odds for all runners in the market (same ordering as
        when the bet was placed).
    devig_method:
        Devigging method applied to the closing market. Defaults to ``"power"``.

    Returns
    -------
    float
        CLV as a probability difference.  Positive = beat the close.
    """
    if bet_odds <= 1.0:
        raise ValueError(f"bet_odds must be > 1.0, got {bet_odds!r}")
    if not (0 <= bet_runner_idx < len(market_close_odds)):
        raise ValueError(
            f"bet_runner_idx {bet_runner_idx!r} out of range for "
            f"market of {len(market_close_odds)} runners"
        )

    closing_fair_probs = _devig(market_close_odds, method=devig_method)
    closing_fair_prob = closing_fair_probs[bet_runner_idx]

    bet_implied_prob = 1.0 / bet_odds

    return closing_fair_prob - bet_implied_prob


# ---------------------------------------------------------------------------
# Aggregate CLV over a set of bets
# ---------------------------------------------------------------------------


def aggregate_clv(
    bets: "pd.DataFrame",
    devig_method: str = "power",
) -> dict[str, float | int]:
    """Compute aggregate CLV summary statistics over a DataFrame of bets.

    The DataFrame must contain a ``clv_pct`` column with pre-computed CLV
    values (decimal, e.g. 0.025 for +2.5pp).  If ``clv_pct`` is absent,
    a ValueError is raised.

    Summary statistics returned:

    - ``mean_clv``         -- arithmetic mean CLV across all bets
    - ``median_clv``       -- median CLV
    - ``pct_positive``     -- fraction of bets with CLV > 0
    - ``n``                -- number of bets with non-null CLV
    - ``wilson_lower``     -- lower bound of 95% Wilson CI on positive CLV rate
    - ``wilson_upper``     -- upper bound of 95% Wilson CI on positive CLV rate

    Parameters
    ----------
    bets:
        DataFrame with at least a ``clv_pct`` column.
    devig_method:
        Currently unused; reserved for future on-the-fly devig.

    Returns
    -------
    dict
        Summary statistics.

    Raises
    ------
    ValueError
        If ``clv_pct`` column is missing or all values are null.
    """

    if "clv_pct" not in bets.columns:
        raise ValueError("DataFrame must contain a 'clv_pct' column")

    clv = bets["clv_pct"].dropna()
    n = len(clv)

    if n == 0:
        raise ValueError("No non-null CLV values found in the DataFrame")

    mean_clv = float(clv.mean())
    median_clv = float(clv.median())
    n_positive = int((clv > 0).sum())
    pct_positive = n_positive / n

    # Wilson score confidence interval on the positive-CLV rate.
    wilson_lower, wilson_upper = _wilson_ci(n_positive, n, alpha=0.05)

    return {
        "mean_clv": mean_clv,
        "median_clv": median_clv,
        "pct_positive": pct_positive,
        "n": n,
        "wilson_lower": wilson_lower,
        "wilson_upper": wilson_upper,
    }


def clv_significance(
    bets: "pd.DataFrame",
    n_bootstrap: int = 5000,
    random_state: int = 42,
) -> dict[str, float | int]:
    """Test whether mean CLV is significantly positive.

    Applies two methods:

    1. One-sample t-test (H0: mean CLV == 0, one-tailed p > 0).
    2. Bootstrap percentile CI on mean CLV.

    Parameters
    ----------
    bets:
        DataFrame with a ``clv_pct`` column.
    n_bootstrap:
        Number of bootstrap resamples for the CI. Defaults to 5000.
    random_state:
        Seed for the bootstrap resampler.

    Returns
    -------
    dict with keys:
        ``mean_clv``       -- sample mean CLV
        ``t_stat``         -- one-sample t-statistic (CLV vs 0)
        ``p_value``        -- one-tailed p-value (H1: CLV > 0)
        ``boot_ci_lower``  -- 2.5th-percentile of bootstrap means
        ``boot_ci_upper``  -- 97.5th-percentile of bootstrap means
        ``n``              -- sample size

    Raises
    ------
    ValueError
        If fewer than 2 non-null CLV values are present.
    """
    if "clv_pct" not in bets.columns:
        raise ValueError("DataFrame must contain a 'clv_pct' column")

    clv = bets["clv_pct"].dropna().to_numpy(dtype=float)
    n = len(clv)

    if n < 2:
        raise ValueError(f"Need at least 2 CLV observations for significance testing; got {n}")

    # One-sample t-test.
    t_stat, p_two_tailed = stats.ttest_1samp(clv, popmean=0.0)
    # One-tailed p-value: H1 is CLV > 0.
    p_one_tailed = float(p_two_tailed / 2.0 if t_stat >= 0 else 1.0 - p_two_tailed / 2.0)

    # Bootstrap CI on mean CLV.
    rng = np.random.default_rng(random_state)
    boot_means = np.array(
        [rng.choice(clv, size=n, replace=True).mean() for _ in range(n_bootstrap)]
    )
    boot_ci_lower = float(np.percentile(boot_means, 2.5))
    boot_ci_upper = float(np.percentile(boot_means, 97.5))

    return {
        "mean_clv": float(clv.mean()),
        "t_stat": float(t_stat),
        "p_value": p_one_tailed,
        "boot_ci_lower": boot_ci_lower,
        "boot_ci_upper": boot_ci_upper,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _wilson_ci(
    n_positive: int,
    n_total: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion.

    Parameters
    ----------
    n_positive:
        Number of successes.
    n_total:
        Total trials.
    alpha:
        Significance level; 0.05 gives a 95% CI.

    Returns
    -------
    tuple[float, float]
        (lower bound, upper bound).
    """
    if n_total == 0:
        return 0.0, 1.0

    z = stats.norm.ppf(1.0 - alpha / 2.0)
    p_hat = n_positive / n_total
    denominator = 1.0 + z**2 / n_total
    centre = (p_hat + z**2 / (2 * n_total)) / denominator
    spread = (z * np.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2))) / denominator
    return float(max(0.0, centre - spread)), float(min(1.0, centre + spread))
