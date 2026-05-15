# EV, Staking, and Model Evaluation Framework

**Context:** AFL stats advisor (model-driven value betting, not arbitrage). This document covers the quantitative scaffolding: how to compute expected value, size stakes, and evaluate whether the model is genuinely producing an edge.

---

## Table of Contents

1. [Devigging / Removing the Overround](#1-devigging--removing-the-overround)
2. [Expected Value Computation](#2-expected-value-computation)
3. [Kelly Criterion and Variants](#3-kelly-criterion-and-variants)
4. [Flat Staking vs Kelly](#4-flat-staking-vs-kelly)
5. [Closing Line Value](#5-closing-line-value)
6. [Calibration Evaluation](#6-calibration-evaluation)
7. [Backtest Methodology](#7-backtest-methodology)
8. [Sample Size and Statistical Significance](#8-sample-size-and-statistical-significance)
9. [Multi-Leg Bets (Parlays/SGM)](#9-multi-leg-bets-parlayssgm)
10. [Paper Trading Log Design](#10-paper-trading-log-design)
11. [Bankroll Management](#11-bankroll-management)
12. [Recommended MVP Evaluation Framework](#12-recommended-mvp-evaluation-framework)

---

## 1. Devigging / Removing the Overround

The bookmaker's "overround" (margin, vig, juice) causes raw implied probabilities from decimal odds to sum to more than 1.0. Devigging recovers the market's best estimate of true probability by forcing those probabilities to sum to exactly 1.0.

**Raw implied probability from decimal odds:**

```
p_implied = 1 / decimal_odds
overround = sum(p_implied for each outcome) - 1.0
```

### 1.1 Proportional (Multiplicative) Method

Scales each implied probability down proportionally.

```python
def devig_proportional(odds: list[float]) -> list[float]:
    implied = [1 / o for o in odds]
    total = sum(implied)
    return [p / total for p in implied]
```

**When to use:** Two-way balanced markets where vig is likely distributed evenly (e.g., -110/-110 lines). Fast and transparent. Assumes every outcome absorbs vig in proportion to its probability mass.

**Limitation:** Does not account for favourite-longshot bias -- longshots are systematically overpriced in the raw line, so proportional devigging does not fully correct for the known tendency of sharp markets to load extra margin onto outsiders. [Outlier Bet, 2024]

### 1.2 Power Method

Finds an exponent `k` such that `sum(p_implied^k) = 1.0`, then uses `p_fair = p_implied^k`.

```python
from scipy.optimize import brentq

def devig_power(odds: list[float]) -> list[float]:
    implied = [1 / o for o in odds]
    # Solve for k: sum(p^k) = 1
    k = brentq(lambda k: sum(p ** k for p in implied) - 1.0, 0.5, 3.0)
    return [p ** k for p in implied]
```

**When to use:** General-purpose default. Handles favourite-longshot bias reasonably well without overcorrecting. The de facto standard for two-way betting markets across most professional tooling. Many devig calculators use this by default. [Bet Hero, 2024; Outlier Bet, 2024]

**Mathematical intuition:** When vig is above 0, k > 1. Raising smaller probabilities (longshots) to a power greater than 1 shrinks them proportionally more than it shrinks larger probabilities (favourites), correcting the bias direction.

### 1.3 Shin Method

Based on Hyun Song Shin's 1993 academic work on information asymmetry in betting markets. Models the bookmaker margin as arising from the presence of a proportion `z` of insider bettors. Requires iterative numerical solution.

```python
import shin

# Three-outcome market (e.g., home/draw/away)
odds = [2.6, 2.4, 4.3]
probs = shin.calculate_implied_probabilities(odds)
# Returns: [0.373, 0.405, 0.222]

# With diagnostics (z = insider proportion, iterations)
probs, info = shin.calculate_implied_probabilities(odds, full_output=True)
```

Install: `pip install shin` (requires Python >= 3.9, uses Rust optimizer internally). [mberk/shin, GitHub]

**When to use:** Futures markets, horse racing, multi-outcome markets with large favourite-longshot bias. Theoretically the most principled method when the bias is strongest. For AFL head-to-head (binary market), power and Shin converge closely.

### 1.4 Additive (Equal Margin) Method

Subtracts `overround / num_outcomes` from each implied probability.

```python
def devig_additive(odds: list[float]) -> list[float]:
    implied = [1 / o for o in odds]
    total = sum(implied)
    overround = total - 1.0
    margin_per = overround / len(implied)
    return [p - margin_per for p in implied]
```

**When to use:** Sanity-checking only. Appropriate for perfectly symmetric markets (both sides at identical odds). Produces negative probabilities for heavy longshots when margin is large. Rarely appropriate in practice.

### 1.5 Method Comparison and AFL Guidance

| Market type | Recommended method |
|---|---|
| H2H two-way (balanced, < 5% margin) | Proportional or Power (converge closely) |
| H2H two-way (unbalanced favourite) | Power |
| Line betting (spread) | Proportional (symmetric by design) |
| Player props (multi-outcome) | Shin or Power |
| Futures / Brownlow / Coleman | Shin |
| Pinnacle-sourced benchmark | Any (margin < 2%; all methods nearly identical) |

**Key insight:** Method choice matters most when margin is high (> 7%). For the Sportsbet / TAB AFL market with typical margins of 5--10%, Power is the recommended default. Use Pinnacle's line as the devigged benchmark where available, since its margin is so small the method is irrelevant. [Pinnacle Betting Resources]

---

## 2. Expected Value Computation

### 2.1 Core Formula

```
EV = model_probability * decimal_odds - 1
```

Equivalently, in terms of devigged market probability:

```
edge = model_probability - fair_market_probability
EV   = edge * (decimal_odds - 1) / decimal_odds    # as a fraction of stake
```

For a concrete AFL example:

```python
model_prob   = 0.58   # model assigns 58% win probability
decimal_odds = 1.90   # bookmaker line
ev           = model_prob * decimal_odds - 1
# ev = 0.58 * 1.90 - 1 = 0.102 - 1 = +0.102 (10.2% edge)
```

A positive EV means the expected profit per unit staked is positive. It does not mean the bet wins. Over a sufficiently large sample, positive EV bets will yield profit. [Pinnacle Betting Resources]

### 2.2 Edge Thresholds -- What Is Meaningful?

Professional consensus is that minimum edges of 2--3% are required before placing a bet, to account for:

1. **Model uncertainty** -- your estimated probability has a confidence interval; the true edge is blurry.
2. **Line movement** -- odds may shift adversely between model run and bet placement.
3. **Transaction costs** -- account limits, line restrictions, and minimum bet size friction reduce realised edge.
4. **Model decay** -- even if the model has an edge today, markets adapt; edges compress over time.

```python
MIN_EDGE = 0.03  # 3% minimum edge threshold (conservative starting point)

def should_bet(model_prob: float, decimal_odds: float) -> bool:
    ev = model_prob * decimal_odds - 1
    return ev >= MIN_EDGE
```

**For an AFL MVP model:** Start at 3% minimum edge. Reduce to 2% only after 300+ bets confirm positive CLV (Section 5) at that threshold. The market is moderately efficient; edges above 8% on standard lines are likely model error, not genuine signal. [Sports AI Dev, 2024; Pinnacle Betting Resources]

### 2.3 Accounting for Model Uncertainty in EV

Rather than treating model probability as a point estimate, consider a credible interval:

```python
import numpy as np
from scipy.stats import beta

# After observing W wins out of N bets in this market type
# Use a Beta posterior (conjugate to Binomial)
W, N = 42, 80
a_post, b_post = 1 + W, 1 + (N - W)
samples = beta.rvs(a_post, b_post, size=10_000)

# Expected EV distribution given model uncertainty
ev_samples = samples * decimal_odds - 1
p_positive_ev = (ev_samples > 0).mean()
expected_ev   = ev_samples.mean()
```

If the 10th percentile of EV is still positive, the edge is robust to estimation error.

---

## 3. Kelly Criterion and Variants

### 3.1 Full Kelly

Kelly (1956) showed that maximising the expected log of wealth maximises long-run growth rate. For a binary bet:

```
f* = (p * b - q) / b
   = p - q / b
```

Where:
- `p` = probability of winning (model estimate)
- `q = 1 - p` = probability of losing
- `b` = net profit per unit staked (decimal_odds - 1)

In decimal odds form:

```
f* = (model_prob * decimal_odds - 1) / (decimal_odds - 1)
   = EV / (decimal_odds - 1)
```

```python
def kelly_fraction(model_prob: float, decimal_odds: float) -> float:
    b = decimal_odds - 1
    q = 1 - model_prob
    return (model_prob * b - q) / b
```

**Theoretical property of full Kelly:** There is an X% chance that the bankroll drops to X% of its starting value at some point during the betting sequence. That is, a 10% drawdown has a ~10% probability, a 50% drawdown has ~50% probability. This is uncomfortable in practice. [Downey, 2024]

### 3.2 Why Full Kelly Is Dangerous in Practice

Three compounding problems make full Kelly impractical:

**1. Estimation error** (dominant concern)

E.O. Thorp proved that overbetting is worse than underbetting. If your true edge is `e` but you estimated `e + delta` (overstated), you are overbetting. The growth rate function is concave, so overbetting reduces long-run growth -- even if you think you have an edge, overestimating it leads to ruin faster than caution would. [Downey, 2024; Wikipedia Kelly criterion]

**2. Ruin risk from fat-tailed outcomes**

Even a small probability of catastrophic loss materially reduces optimal bet size. A 60/40 coin with an independent 1% chance of losing everything drops the optimal bet from 0.80 to 0.46. AFL models face equivalent tail risks: injury news, weather, umpiring variance.

**3. Drawdown psychology**

Full Kelly produces drawdowns of 30--50% routinely even when the model is correct. This creates behavioural risk: bettors abandon correct strategies during drawdowns.

### 3.3 Fractional Kelly

Use `f = c * f*` where `c` is a fraction:

| Fraction | Growth rate sacrifice | Drawdown reduction |
|---|---|---|
| Full (c=1.0) | 0% | Baseline |
| Half (c=0.5) | ~25% | Substantial |
| Quarter (c=0.25) | ~44% | Severe; drawdown ~ quarter |
| Eighth (c=0.125) | ~58% | Very low volatility |

**Recommendation for AFL MVP model:** Quarter Kelly (c=0.25) or Eighth Kelly (c=0.125). At this stage, model uncertainty is high; the loss in growth rate is acceptable insurance against estimation error. Once CLV is confirmed positive over 300+ bets, step up to half Kelly if desired. [Sportstrade.io, 2024; Downey, 2024]

```python
def fractional_kelly(model_prob: float, decimal_odds: float,
                     fraction: float = 0.25) -> float:
    f_star = kelly_fraction(model_prob, decimal_odds)
    return max(0.0, fraction * f_star)  # never bet negative
```

### 3.4 Capped Kelly

Apply a hard cap regardless of what Kelly suggests:

```python
MAX_BET_FRACTION = 0.03  # never bet more than 3% of bankroll

def capped_kelly(model_prob: float, decimal_odds: float,
                 fraction: float = 0.25) -> float:
    f = fractional_kelly(model_prob, decimal_odds, fraction)
    return min(f, MAX_BET_FRACTION)
```

### 3.5 Simultaneous Bets and Correlated Markets

Standard Kelly assumes bets are placed sequentially. When placing multiple bets simultaneously (multiple AFL games on a Saturday), the single-bet formula overallocates because combined exposure is higher than the sum of individual risks.

For `N` simultaneous uncorrelated bets, the portfolio Kelly fraction per bet is approximately `f_i / N` (conservative approximation). For correlated bets (e.g., same-game markets), the formula requires numerical optimisation of the log-wealth function:

```python
from scipy.optimize import minimize
import numpy as np

def portfolio_kelly(probs: list[float], odds: list[float]) -> list[float]:
    """
    Numerically solve Kelly for N simultaneous bets.
    probs: model win probabilities per bet
    odds: decimal odds per bet
    Returns optimal fractions for each bet.
    """
    n = len(probs)
    
    def neg_expected_log_wealth(f):
        # Approximation: iterate over all 2^n outcomes
        # For large N, use Monte Carlo
        total = 0
        for outcome in range(2 ** n):
            p_outcome = 1.0
            growth    = 1.0
            for i in range(n):
                won = (outcome >> i) & 1
                if won:
                    p_outcome *= probs[i]
                    growth    *= (1 + f[i] * (odds[i] - 1))
                else:
                    p_outcome *= (1 - probs[i])
                    growth    *= (1 - f[i])
            total += p_outcome * np.log(max(growth, 1e-10))
        return -total
    
    bounds   = [(0, 0.05)] * n  # cap each bet at 5%
    result   = minimize(neg_expected_log_wealth, [0.01] * n,
                        method='L-BFGS-B', bounds=bounds)
    return result.x.tolist()
```

**Practical rule:** For AFL, cap total exposure on any one round at 10% of bankroll, regardless of what individual Kelly fractions suggest. Multiple AFL games on the same round are correlated through common factors (umpiring inconsistency, weather across venues, broadcasted momentum effects). [Wikipedia Kelly criterion; Vegapit, 2024]

---

## 4. Flat Staking vs Kelly

### 4.1 The Case for Flat Staking

Flat staking (fixed unit size per bet, regardless of model confidence) outperforms Kelly in practice under two conditions:

1. **High model uncertainty:** When your probability estimates have wide confidence intervals, Kelly amplifies estimation error directly into stake size. Flat staking contains the damage.
2. **Short sample sizes:** Kelly's growth-maximising property requires the law of large numbers to operate. Over 50--200 bets, Kelly's variance can produce catastrophic drawdowns before the edge emerges.

Buchdahl (Football-Data.co.uk) has documented empirically that flat staking frequently produces better risk-adjusted returns than Kelly for model-driven bettors when the edge is uncertain -- the Kelly fraction is only optimal if your probability estimate is exact, and it never is. [Trademate Sports, 2024]

### 4.2 A Practical Heuristic

```
Stage 1 (0--300 bets): Flat staking at 1 unit per bet
Stage 2 (300+ bets, positive CLV confirmed): Quarter Kelly
Stage 3 (1000+ bets, calibrated model): Half Kelly with exposure caps
```

This ladder avoids the scenario where an overconfident model stakes heavily on early bets that happen to be data artefacts rather than genuine edge.

### 4.3 When Kelly Genuinely Wins

Kelly outperforms flat staking when:
- Edge is precisely known (e.g., mechanical arbitrage with calculable margins)
- Bet sizes vary significantly in edge magnitude (high-edge bets should scale up)
- Long-run compounding is the objective, short-term variance is irrelevant

For AFL value betting with an ML model, neither condition is cleanly satisfied in the MVP phase.

---

## 5. Closing Line Value

### 5.1 Definition and Formula

Closing Line Value (CLV) measures whether your bet was placed at better odds than where the market finally settled (the closing line). It is the primary signal for betting skill because it measures process quality (are you identifying mispriced lines?) independently of outcomes (did the bet win?).

**CLV as a percentage (probability-based, preferred):**

```
CLV% = (closing_fair_prob - bet_fair_prob) / bet_fair_prob
```

Where `bet_fair_prob` is your bet's implied probability after devigging the odds you received, and `closing_fair_prob` is the devigged closing line probability for the same outcome.

```python
def compute_clv(bet_odds: float, closing_odds: float,
                closing_opposite_odds: float) -> float:
    """
    CLV as a percentage for a two-way market.
    bet_odds: decimal odds you received
    closing_odds: final odds on your side at close
    closing_opposite_odds: final odds on the other side at close
    """
    # Devig the closing line (proportional method)
    close_implied = 1 / closing_odds
    close_opp     = 1 / closing_opposite_odds
    close_total   = close_implied + close_opp
    closing_fair  = close_implied / close_total  # devigged closing prob

    # Devig your bet (single odds; need both sides at time of bet if available)
    bet_implied   = 1 / bet_odds

    # CLV%: positive = you got better odds than the market eventually settled on
    # (note: bet_implied < closing_fair means you got longer odds = positive CLV)
    return (bet_implied - closing_fair) / closing_fair  # negative = beat the close
    # Alternatively (Buchdahl convention):
    # return (closing_odds / bet_odds) - 1  # positive = beat the close
```

The Unabated / Buchdahl convention: `CLV = (closing_devigged_odds / bet_odds) - 1`. Positive means your odds were longer than the fair closing price; you beat the market. [Buchdahl via Pinnacle Odds Dropper; Unabated, 2024]

### 5.2 Why CLV Is the Gold Standard

Profit (ROI) requires the law of large numbers to distinguish skill from luck. Win/loss records at even money need thousands of bets to demonstrate statistical significance. CLV varies in much smaller increments (~0.1 standard deviation per bet vs ~1.0 for win/loss), enabling much faster signal detection.

Buchdahl estimates that consistent positive CLV can demonstrate statistical significance in as few as 50 bets -- roughly 20--40x faster than results-based evaluation. [Buchdahl via Pinnacle Odds Dropper, 2024]

**CLV is not the same as ROI:**
- Positive CLV + negative ROI in the short run is normal and expected. Bad outcomes happen to good processes.
- Negative CLV + positive ROI is a red flag. Results-based profits without process quality usually mean luck, not skill.
- The market is a continuous evaluation mechanism. If you consistently beat the market's final estimate, you have an informational edge.

### 5.3 Sample Size Requirements for CLV

| Sample size | Signal quality |
|---|---|
| < 50 bets | CLV trend visible but not reliable |
| 50--250 bets | Meaningful directional signal; not conclusive |
| 250--500 bets | Statistical significance beginning to emerge |
| 500--1000 bets | Strong confidence in edge direction |
| > 1000 bets | Production-grade validation |

**Rolling windows:** Track CLV on a rolling 250-bet window and a rolling 1000-bet window. If the 250-bet window turns negative for three consecutive observations, treat this as an alert that market conditions may have shifted. [Sports AI Dev, 2024]

### 5.4 Practical CLV for AFL

AFL's primary closing line benchmark: Pinnacle does not offer AFL. Use Sportsbet or TAB closing lines, with the caveat that these are softer books -- beating the closing line on a soft book is a weaker signal than beating Pinnacle. If Betfair (exchange) odds are available at or near close, these are the most efficient benchmark since they are set by real money with no bookmaker margin involved.

**Log the odds you received AND the closing odds for every bet.** This is non-negotiable for CLV computation. If closing odds are not captured at time of settlement, they are often unrecoverable.

---

## 6. Calibration Evaluation

A calibrated model is one where predicted probabilities mean what they say: if the model says 70%, the team should win 70% of the time across all bets where that prediction was made. Miscalibration directly distorts EV calculations.

### 6.1 Brier Score

```
BS = (1/N) * sum((p_i - o_i)^2)
```

Where `p_i` is the predicted probability and `o_i` is the outcome (1 or 0).

- **Lower is better.**
- Baseline for a 50/50 market with naive 0.5 predictions: BS = 0.25.
- A model with BS > 0.25 on AFL head-to-head is worse than constant 50% predictions.
- The Brier score decomposes into calibration + resolution components, making it diagnostically useful.

```python
import numpy as np

def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return np.mean((probs - outcomes) ** 2)
```

[Brier Score Wikipedia; Sports AI Dev, 2024]

### 6.2 Log Loss

```
LL = -(1/N) * sum(o_i * log(p_i) + (1-o_i) * log(1-p_i))
```

Log loss penalises overconfident wrong predictions more severely than Brier score. Useful for training signal (gradient-based optimisation). More volatile than Brier score for evaluation.

```python
from sklearn.metrics import log_loss

ll = log_loss(outcomes, probs)
```

**Clip predicted probabilities to [0.01, 0.99] before computing log loss** to avoid numerical instability on extreme predictions.

### 6.3 Reliability Diagrams (Calibration Curves)

Bin predictions by decile (0--10%, 10--20%, ..., 90--100%). For each bin, compute the empirical win rate. Plot predicted probability vs actual win rate. A perfectly calibrated model lies on the `y = x` diagonal.

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

fraction_of_positives, mean_predicted = calibration_curve(
    outcomes, probs, n_bins=10, strategy='uniform'
)

plt.plot(mean_predicted, fraction_of_positives, 's-', label='Model')
plt.plot([0, 1], [0, 1], '--', label='Perfect calibration')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Reliability diagram')
```

**Red flags:**
- Convex curve below the diagonal: model is overconfident.
- Concave curve above the diagonal: model is underconfident.
- High-probability and low-probability bins collapsing toward the centre: regularisation over-smoothing.

### 6.4 Expected Calibration Error (ECE)

```
ECE = sum_bins(|bin_size/N| * |avg_pred_prob - empirical_freq|)
```

Trigger recalibration when ECE > 0.015. [Sports AI Dev, 2024]

### 6.5 Recalibration Without Retraining

When the reliability diagram shows systematic bias, apply a post-hoc calibrator to the model's raw probability outputs. Use a held-out calibration set (not the training set, not the test set):

**Platt Scaling** (logistic regression on raw logits):
```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(base_model, cv=5, method='sigmoid')
calibrated_model.fit(X_cal, y_cal)
```

Use when calibration curve has a sigmoid shape. Requires fewer calibration samples (can work with 100--200 matches). [Train in Data, 2024]

**Isotonic Regression** (non-parametric monotone mapping):
```python
calibrated_model = CalibratedClassifierCV(base_model, cv=5, method='isotonic')
calibrated_model.fit(X_cal, y_cal)
```

Use for non-sigmoid systematic bias. Prone to overfitting with small calibration sets (< 500 samples). For AFL with ~200 games per season, Platt scaling is safer. [FastML; scikit-learn documentation]

**Stand-alone recalibration on raw outputs:**
```python
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# Isotonic recalibration
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(raw_probs_cal, outcomes_cal)
calibrated_probs = iso.predict(raw_probs_test)

# Platt scaling (operate on log-odds)
lr = LogisticRegression()
lr.fit(raw_probs_cal.reshape(-1, 1), outcomes_cal)
calibrated_probs = lr.predict_proba(raw_probs_test.reshape(-1, 1))[:, 1]
```

**EV is meaningless on uncalibrated probabilities.** Run calibration checks before computing any bet-sizing metric.

---

## 7. Backtest Methodology

### 7.1 Why Standard Backtests Overfit

A 2022 Journal of Sports Analytics study found betting models without proper cross-validation overestimated accuracy by up to 15%. Standard train/test splits are inadequate for sports data because:

1. **Look-ahead bias:** Using closing odds, final team ratings, or updated statistics that would not have been available at bet placement time.
2. **Temporal leakage:** Standard k-fold shuffles time, allowing future data to inform past predictions.
3. **Multiple comparisons:** Testing many parameter combinations, then selecting the best performer, inflates apparent accuracy.

The classic sign of overfit: a strategy shows 12% higher Sharpe in backtests but deteriorates 28% in forward testing. [Oddsonnet, 2024]

### 7.2 Walk-Forward Validation

The correct methodology for time-series betting data:

```
Train: Seasons 2019-2021   | Test: 2022 (round by round)
Train: Seasons 2019-2022   | Test: 2023
Train: Seasons 2019-2023   | Test: 2024
```

Each test window only contains data that existed before it. The model is retrained (or re-validated) at the start of each test window.

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=4)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    evaluate(preds, y_test)
```

Walk-forward CV reduced overfitting in an NBA betting model study by 22%, improving live ROI from -2% to +5.4%. [Oddsonnet, 2024]

### 7.3 Look-Ahead Bias Checklist for AFL

Every feature used in the model must satisfy: "Would this number have been exactly this value at the time the bet was placed?"

| Feature | Risk | Mitigation |
|---|---|---|
| Team form (last N games) | Safe if computed only from completed rounds | Use game index, not date |
| Player availability | High risk | Only use pre-game announcements published before markets open |
| Elo / power ratings | Safe if updated only after game completion | Ensure lag of +1 round |
| Weather | Moderate | Use forecast available at game-day morning, not actuals |
| Closing odds | **Critical:** never use to predict the same market's outcome | Use only as CLV benchmark |
| Injury reports | High | Timestamp every data pull; only use data older than market open |

### 7.4 Bootstrap Confidence Intervals on ROI

Point estimate ROI is unreliable over small samples. Use bootstrap resampling:

```python
import numpy as np

def bootstrap_roi_ci(bet_returns: np.ndarray, n_boot: int = 10_000,
                     ci: float = 0.95) -> tuple:
    """
    bet_returns: array of (p&l / stake) per bet, e.g., +0.9 for a win at 1.9 odds
    Returns (lower, upper) confidence interval on ROI.
    """
    boot_means = np.array([
        np.random.choice(bet_returns, size=len(bet_returns), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return np.quantile(boot_means, alpha), np.quantile(boot_means, 1 - alpha)

# Example usage
returns = np.array([0.9, -1.0, 0.9, -1.0, 1.8, -1.0, ...])
lower, upper = bootstrap_roi_ci(returns)
# If lower > 0 at 95% CI, edge is statistically meaningful
```

---

## 8. Sample Size and Statistical Significance

### 8.1 How Many Bets to Declare an Edge Real

For a two-sided test on ROI (null: ROI = 0):

```python
from scipy.stats import ttest_1samp, sem
import numpy as np

def edge_significance(returns: np.ndarray):
    t_stat, p_value = ttest_1samp(returns, popmean=0)
    roi   = returns.mean()
    ci_95 = roi + np.array([-1.96, 1.96]) * sem(returns)
    return {'roi': roi, 'p_value': p_value, 't_stat': t_stat, 'ci_95': ci_95}
```

Approximate sample sizes needed for different win rates to reach p < 0.05 (frequentist, one-sample t-test):

| Win rate at -110 (50% implied) | Sample size needed |
|---|---|
| 53% | ~1,500 bets |
| 55% | ~600 bets |
| 58% | ~250 bets |
| 60%+ | ~150 bets |

For typical AFL model edges (2--5%), expect to need 300--1,000 bets for frequentist significance. CLV provides the same signal 20--40x faster.

### 8.2 Wilson Score Interval for Win Rate

Prefer the Wilson score interval over the naive Wald interval, especially at extreme win rates or small samples:

```python
from statsmodels.stats.proportion import proportion_confint

wins  = 55
total = 100
lower, upper = proportion_confint(wins, total, alpha=0.05, method='wilson')
# Lower bound < 0.5 at 95% CI means the edge is not yet statistically confirmed
```

### 8.3 Sequential Testing

Traditional fixed-sample hypothesis testing is inappropriate for continuously accumulating betting data (the "peeking problem" inflates false positive rates). Use a sequential approach:

**Option A: Bayesian updating** (recommended for AFL MVP)

```python
# Prior: Beta(1, 1) = uniform, no prior belief
# Update with each result: win -> +1 to alpha, loss -> +1 to beta
from scipy.stats import beta as beta_dist

alpha, b_param = 1, 1  # priors
for outcome in outcomes:  # 1 = win, 0 = loss
    alpha   += outcome
    b_param += (1 - outcome)

# Posterior probability that true win rate > 0.5 (at -110 odds)
p_positive_edge = 1 - beta_dist.cdf(0.5, alpha, b_param)
```

**Option B: Sequential probability ratio test (SPRT)**

Wald's SPRT allows for continuous monitoring with controlled Type I and Type II error rates. Suitable for A/B comparisons of two strategies. See Evan Miller's sequential AB testing for implementation details. [Evan Miller, 2024]

**Bayesian advantages for sports betting:**
- No correction needed for continuous monitoring (unlike frequentist tests)
- Posterior credible intervals have intuitive interpretation
- Naturally handles small samples and prior knowledge
- Conjugate Beta-Binomial model updates in O(1) per bet

[Sports Insights; Analytics.bet; Punter2Pro, 2024]

---

## 9. Multi-Leg Bets (Parlays/SGM)

### 9.1 The Independence Assumption Violation

Traditional parlay pricing assumes outcome independence:

```
parlay_odds = product(odds_i for each leg i)
parlay_prob = product(p_i for each leg i)
```

This is only valid if outcomes are truly independent. Within a single AFL game, outcomes are correlated -- a team's win, winning margin, and a player's disposals are all influenced by common underlying factors (team quality, matchup, conditions, game state).

Sportsbooks apply a "correlation tax" by reducing payout when legs are correlated. The house edge on SGMs is typically **15--25%**, versus **4--7%** for single bets. [OddsIndex, 2024]

### 9.2 When SGMs Can Have Genuine +EV

Positive expected value in same-game multis exists only when:

1. **Your correlation model is more accurate than the bookmaker's.** If the sportsbook uses a naive independence model and you have empirical correlation estimates, the mismatch creates an edge.
2. **The sportsbook caps the correlation discount.** Some bookmakers apply a fixed discount that does not fully reflect true correlation. If true correlation is lower than their discount, the SGM is priced in your favour.
3. **The individual legs are mispriced before the correlation adjustment.** If each leg has +EV individually, stacking them compounds edge -- but also compounds estimation error.

For AFL player props: disposals and kicks are correlated (~0.75 Pearson r). Disposals and team score are correlated but weaker (~0.35). A model that correctly prices these correlations can identify when a bookmaker's SGM offering is +EV.

### 9.3 Modelling Approach

```python
import numpy as np
from scipy.stats import multivariate_normal

# Estimate correlation between two player props from historical data
# e.g., PlayerA disposals and PlayerA kicks in same game
historical_data = ...  # shape (N_games, 2)
corr_matrix = np.corrcoef(historical_data.T)

# Simulate joint outcomes using multivariate normal (Gaussian copula)
simulations = multivariate_normal.rvs(mean=[mu1, mu2], cov=corr_matrix, size=10_000)
# Price the multi based on empirical joint probability
joint_prob = np.mean((simulations[:, 0] > threshold1) & (simulations[:, 1] > threshold2))
```

### 9.4 Practical Recommendation

For an AFL MVP model: **avoid SGMs until single-bet CLV is confirmed positive.** The complexity of correlation modelling adds a second layer of model risk on top of an unvalidated base model. Multi-leg bets are an optimisation for after the base model is proven.

---

## 10. Paper Trading Log Design

The paper trading log is the single source of truth for model evaluation. Missing fields at the time of bet placement are almost always unrecoverable later. Design for completeness from day one.

### 10.1 Required Fields

| Field | Type | Notes |
|---|---|---|
| `bet_id` | UUID | Unique identifier |
| `timestamp_placed` | ISO 8601 | Exact time bet was placed (or simulated) |
| `sport` | str | AFL |
| `market` | str | head-to-head / line / total / player prop |
| `event` | str | "Team A vs Team B, Round N, Season YYYY" |
| `round` | int | AFL round number |
| `season` | int | Season year |
| `team_bet` | str | Which team / side bet on |
| `bet_odds_decimal` | float | Odds at time of placement |
| `model_prob` | float | Model's win probability |
| `fair_prob_at_placement` | float | Devigged market probability at placement |
| `ev_at_placement` | float | `model_prob * odds - 1` |
| `kelly_fraction` | float | Recommended stake fraction |
| `stake_units` | float | Actual stake in units |
| `closing_odds_decimal` | float | Final odds before game start |
| `closing_opposite_odds` | float | Final odds for opposing outcome (for devigging) |
| `closing_fair_prob` | float | Devigged closing probability |
| `clv_pct` | float | Computed closing line value % |
| `outcome` | int | 1 = win, 0 = loss |
| `pnl_units` | float | Profit/loss in units |

### 10.2 Model Snapshot Fields (Prevent Look-Ahead Bias in Review)

| Field | Type | Notes |
|---|---|---|
| `model_version` | str | Git commit hash or version tag |
| `model_inputs_hash` | str | Hash of feature vector (for reproducibility) |
| `home_elo` | float | Home team Elo at time of prediction |
| `away_elo` | float | Away team Elo at time of prediction |
| `home_form_3` | float | Home team win rate over last 3 games |
| `away_form_3` | float | Away team win rate over last 3 games |
| `home_venue_advantage` | float | Model's venue adjustment |
| `market_open_odds` | float | Opening line (to track line movement) |
| `odds_movement_pct` | float | `(closing - placement) / placement` |

### 10.3 Decision Rationale Field

```json
{
  "decision_rationale": {
    "edge_threshold_met": true,
    "model_confidence": "high",
    "market_conditions": "early-line, pre-team-announcement",
    "flags": ["player_injury_risk_elevated", "away_game_heavy_rain_forecast"],
    "override": null
  }
}
```

Even for a mechanical model, logging the market conditions and any flags encountered is essential for post-hoc analysis of why edge degraded in certain contexts.

### 10.4 Recommended Storage

A single CSV or SQLite database with one row per bet. Add a `notes` free-text field for anything not captured in structured fields. Do not normalise prematurely -- flat is fine for hundreds to low thousands of bets.

---

## 11. Bankroll Management

### 11.1 Unit Sizing

Define 1 unit as a fixed percentage of starting bankroll (not current bankroll, until stake management is formulated):

- **Recommendation for MVP:** 1 unit = 1% of bankroll.
- **Rationale:** At 1%, a catastrophic 20-unit losing streak (plausible with a flawed model) loses 20% of bankroll -- uncomfortable but survivable.
- Do not use dollar amounts. Units as percentages keep sizing consistent with bankroll growth.

### 11.2 Exposure Caps

Individual Kelly fractions do not account for correlation between bets placed on the same day/round. Hard exposure caps prevent over-concentration:

| Cap level | Rule |
|---|---|
| Per game | Maximum 3 units across all markets in a single AFL game |
| Per round | Maximum 10 units across all games in a single AFL round |
| Per day | Maximum 5% of bankroll staked (across all concurrent bets) |
| Per week | Soft review trigger if 15 units staked in one week |

**Exposure caps matter more than Kelly precision.** Kelly tells you the optimal fraction assuming your model is correct; caps protect you when it is not. [TheSpread.com, 2024]

### 11.3 Stop-Loss Policy

A stop-loss prevents tilt-driven over-betting after a drawdown:

```
Session stop-loss:  -4 units in a single day -> pause until next round
Drawdown alert:     -15% of bankroll from peak -> mandatory model review
Drawdown hard stop: -25% of bankroll from peak -> halt betting, investigate
```

A 25% drawdown in a correctly calibrated model with 3% edge is roughly a 1-in-20 event over a full season. If it happens, the prior that the model has a real edge has been substantially weakened.

### 11.4 Bankroll Growth and Restaking

Avoid increasing stake size until:
- At least 300 bets logged
- Positive CLV at the 5% significance level
- Brier score demonstrably better than the market baseline

When scaling up, increase unit size in 25% increments maximum (e.g., 1% -> 1.25% per unit), not jumps. Scaling before edge validation is the most common route to bankroll destruction.

---

## 12. Recommended MVP Evaluation Framework

This section defines the minimum dashboard to run from day one, what to compute weekly, and what triggers a model pause.

### 12.1 Per-Bet Log (Every Bet)

Capture all fields from Section 10. Non-negotiable:
- Model probability
- Odds at placement
- Closing odds (both sides)
- Outcome

Without these, CLV is uncomputable.

### 12.2 Weekly Metrics Dashboard

| Metric | Formula | Alert threshold |
|---|---|---|
| ROI | `sum(pnl) / sum(stake)` | < -5% over 50+ bets |
| CLV% mean | `mean(clv_pct)` | < 0% over rolling 100 bets |
| CLV% std | `std(clv_pct)` | N/A (diagnostic) |
| Brier score | `mean((p - o)^2)` | > 0.25 (worse than naive baseline) |
| ECE | Calibration curve deviation | > 0.015 |
| Win rate | `wins / total_bets` | > 2 SE below implied from odds |
| Bankroll drawdown | `(peak - current) / peak` | > 15% -> alert, > 25% -> halt |
| Edge hit rate | Fraction of bets with EV > threshold | < 40% -> review threshold |

### 12.3 Monthly Calibration Audit

1. Plot reliability diagram on last N=200 bets.
2. Run Platt scaling on a held-out calibration set.
3. Compare Brier score before and after recalibration.
4. If ECE > 0.015 after recalibration, investigate feature drift.

### 12.4 Trigger: Model Pause

Pause betting and review the model if any of the following occur:

| Trigger | Description |
|---|---|
| CLV negative | Mean CLV < -1% over a rolling 100-bet window |
| Brier score degradation | BS increases > 0.03 vs prior 100-bet window |
| ECE > 0.02 | After recalibration attempt |
| Drawdown > 25% | Bankroll fell more than 25% from peak |
| Market regime change | Major AFL rule change, schedule anomaly, COVID-style disruption |
| Model confidence anomaly | Model assigns >90% or <10% probability more than 10% of the time |

### 12.5 Minimum Viable Metrics at Launch

If only three things are tracked from day one:

1. **Per-bet CLV%** -- is the model identifying mispriced lines before they close?
2. **Brier score on rolling 200 bets** -- are the probabilities calibrated?
3. **Bankroll drawdown from peak** -- is the model destroying capital?

CLV validates the model's edge. Brier score validates its calibration. Drawdown validates stake management. These three metrics, computed weekly, are sufficient to make a go/no-go decision on any model modification.

---

## References and Sources

- [Shin's method Python package (mberk/shin)](https://github.com/mberk/shin)
- [Devigging methods explained -- Bet Hero](https://betherosports.com/blog/devigging-methods-explained)
- [Devigging methods comparison -- Outlier Bet](https://help.outlier.bet/en/articles/8208129-how-to-devig-odds-comparing-the-methods)
- [Pinnacle: How to calculate Expected Value](https://www.pinnacle.com/betting-resources/en/betting-strategy/how-to-calculate-expected-value/ees2ve46tm4htt32)
- [Kelly criterion -- Wikipedia](https://en.wikipedia.org/wiki/Kelly_criterion)
- [Fractional Kelly and estimation error -- Matthew Downey](https://matthewdowney.github.io/uncertainty-kelly-criterion-optimal-bet-size.html)
- [Modified Kelly Criteria -- Chu, Wu, Swartz (SFU)](https://www.sfu.ca/~tswartz/papers/kelly.pdf)
- [Fractional Kelly bankroll management -- Sportstrade](https://www.sportstrade.io/blog-detail/141/the-fractional-kelly-bankroll-management-system.html)
- [CLV demystified -- Joseph Buchdahl via Pinnacle Odds Dropper](https://www.pinnacleoddsdropper.com/blog/closing-line-value--clv-demystified-by-expert-joseph-buchdahl)
- [Getting precise about CLV -- Unabated](https://unabated.com/articles/getting-precise-about-closing-line-value)
- [CLV and AI model performance -- Sports AI Dev](https://www.sports-ai.dev/blog/closing-line-value-and-ai-model-performance)
- [AI Model Calibration: Brier Score and Reliability -- Sports AI Dev](https://www.sports-ai.dev/blog/ai-model-calibration-brier-score)
- [Brier score -- Wikipedia](https://en.wikipedia.org/wiki/Brier_score)
- [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [Platt scaling and isotonic regression -- FastML](https://fastml.com/classifier-calibration-with-platts-scaling-and-isotonic-regression/)
- [Cross-validation for betting models -- Oddsonnet](https://oddsonnet.com/news/mastering-cross-validation-techniques-for-betting-models-avoid-overfitting-and-boost-profits)
- [Backtest overfitting -- Betting Forum](https://www.betting-forum.com/threads/the-overfitting-problem-why-backtested-betting-systems-fail-in-production.47444/)
- [Statistical significance -- Sports Insights](https://www.sportsinsights.com/sports-investing-statistical-significance/)
- [Sample size -- Punter2Pro](https://punter2pro.com/sample-size-betting-results-analysis/)
- [Sequential AB testing -- Evan Miller](https://www.evanmiller.org/sequential-ab-testing.html)
- [Bayesian sports betting -- Analytics.Bet](https://analytics.bet/bsb/)
- [SGM correlation -- Wizard of Odds](https://wizardofodds.com/article/same-game-parlays-the-mathematics-of-correlation/)
- [SGM correlation tax -- OddsIndex](https://oddsindex.com/guides/same-game-parlay-correlation)
- [Kelly criterion vs flat staking -- Trademate Sports](https://www.tradematesports.com/en/blog/betting-experts-staking-strategy-kelly-criterion-flat-staking-3)
- [Numerically solving Kelly for multiple bets -- Vegapit](https://vegapit.com/article/numerically_solve_kelly_criterion_multiple_simultaneous_bets/)
- [Bankroll management -- TheSpread.com](https://www.thespread.com/betting-guides/bankroll-management-unit-size-staking-plans-risk-limits/)
- [Designing betting systems in R -- R-bloggers](https://www.r-bloggers.com/2026/02/designing-sports-betting-systems-in-r-bayesian-probabilities-expected-value-and-kelly-logic/)
- [Statistical theory of optimal decision-making in sports betting -- PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10306238/)
