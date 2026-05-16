# Evaluation Framework

This document summarises the quantitative metrics used to evaluate the AFL betting model.
All metrics are implemented in `src/bet_advisor/eval/`.

---

## Devigging (Overround Removal)

Raw bookmaker odds imply probabilities that sum to more than 1.0 (the overround).
Devigging recovers fair probability estimates that sum to exactly 1.0.

Four methods are provided (`bet_advisor.eval.devig`):

| Method | When to use |
|--------|-------------|
| `proportional` | Balanced markets; fast, transparent |
| `power` | **Default** - AFL H2H, player props; corrects favourite-longshot bias |
| `shin` | Multi-outcome futures (Brownlow, Coleman); theoretically principled |
| `additive` | Sanity check only; can produce negative probabilities for longshots |

**Default:** `power`. Switch to `shin` for futures markets.

---

## Expected Value

`EV = model_probability * decimal_odds - 1`

A positive EV means the bet is profitable in expectation over a large sample.
EV on a single bet is not informative -- only the long-run aggregate matters.

**Minimum edge threshold** (`min_edge_threshold`): 3% by default.
Adjusts upward for high model uncertainty or thick vig (>5%).

**Uncertainty-aware EV** (`ev_with_uncertainty`): accepts a sample array
(e.g. Beta posterior draws) and returns mean, 5th-, and 95th-percentile EV.
If the 5th-percentile is positive, the edge is robust to estimation error.

---

## Kelly Criterion

Kelly (1956) maximises long-run bankroll growth by staking proportionally
to the edge and inversely to the odds.

```
f* = (model_prob * decimal_odds - 1) / (decimal_odds - 1)
```

**Do not use full Kelly with an unvalidated model.** Three safer variants:

| Function | Description |
|----------|-------------|
| `full_kelly` | Theoretical maximum; dangerous with estimation error |
| `fractional_kelly` | Scales full Kelly by a fraction (default 0.25) |
| `capped_kelly` | Fractional Kelly with a hard cap (default 5% of bankroll) |
| `portfolio_kelly` | Numerical optimisation for simultaneous bets |

**Recommended progression:**
- Stage 1 (0-300 bets): flat staking at 1% of bankroll
- Stage 2 (300+ bets, positive CLV confirmed): quarter Kelly
- Stage 3 (1000+ bets, calibrated): half Kelly with exposure caps

`stake_recommendation` is a dispatcher supporting `"flat"`, `"quarter_kelly"`,
and `"capped_kelly"` modes.

---

## Calibration

A calibrated model's predicted probability of 70% should correspond to an
empirical win rate of 70% across all bets at that predicted probability.
Miscalibration distorts EV calculations.

| Metric | Description | Alert threshold |
|--------|-------------|----------------|
| Brier score | Mean squared error: (pred - outcome)^2 | >0.25 (worse than 50% naive) |
| Log loss | Mean negative log-likelihood | Higher = worse |
| ECE | Weighted mean calibration gap across bins | >0.015 triggers recalibration |

**Reliability diagram** (`reliability_diagram_data`): bin predictions by decile,
plot predicted vs actual win rate. A perfect model lies on the diagonal.

**Recalibrators** (fit on a held-out calibration set, not training data):

- `PlattCalibrator`: logistic regression on raw probs; preferred for AFL
  (n < 500 games per season, low overfitting risk).
- `IsotonicCalibrator`: monotone non-parametric; needs 500+ samples.

---

## Closing Line Value (CLV)

CLV measures whether a bet was placed at better odds than the market's final
price.  It is the primary signal for edge identification because it measures
process quality independently of outcomes.

```
CLV = closing_fair_prob - bet_implied_prob
```

Positive CLV means the bettor received longer odds than the devigged closing
price -- they identified the mispricing before the market corrected it.

**Why CLV beats ROI as an early signal:**
CLV has much lower variance per bet than win/loss, so statistical significance
can emerge in ~50 bets vs ~1000+ bets for ROI-based evaluation.

**Aggregation** (`aggregate_clv`): mean, median, % positive, sample size,
95% Wilson confidence interval on the positive rate.

**Significance testing** (`clv_significance`): one-sample t-test (H0: CLV = 0)
plus bootstrap percentile CI on the mean.

### Reference Close (Phase 5.5)

The closing probability that CLV is measured against is produced by
`ClvReferenceResolver` in `bet_advisor.eval.clv_reference`.

**Default: multi-book consensus** across Sportsbet, TAB, Ladbrokes, Pointsbet,
and Betr (all available AU books via The Odds API). Each book's closing line is
devigged independently using the power method, then runner probabilities are
averaged across books weighted by overround (lower overround = sharper book =
higher weight).

**Optional: Betfair Exchange delayed key** for match-level cross-checks. The
free delayed app key reads `lastPriceTraded` from exchange markets with a
1-180 second delay. When matched volume exceeds the configured threshold
(default AUD 1000), the exchange closing price is used; otherwise the resolver
falls back to consensus with a warning.

**Other modes:** `sportsbet_only` (single-book baseline) and `single_book`
(any configured book) are available for debugging and comparison.

**Accuracy vs Betfair Exchange live close:** approximately 1-3 percentage points
less precise on match-odds markets. For AFL player prop markets (disposals,
tackles), the gap may be smaller because exchange prop markets are thinly
matched and carry weaker price-discovery signal.

**Interpretation threshold:** require sustained CLV > 2% across 200+ bets before
concluding the model has real edge. This accounts for the noise introduced by the
multi-book consensus reference vs the sharp exchange close.

Each settled bet records `clv_reference_source` and `clv_reference_books_used`
in the SQLite `bets` table for audit. The daily report's CLV Reference section
shows the active mode and any fallbacks.

---

## Model Pause Triggers

Pause betting and investigate the model when any of the following occur:

| Trigger | Condition |
|---------|-----------|
| Negative CLV | Mean CLV < -1% over rolling 100 bets |
| Brier degradation | Brier score increases >0.03 vs prior 100-bet window |
| ECE after recalibration | ECE > 0.02 after a Platt or isotonic recalibration attempt |
| Bankroll drawdown | Current bankroll < 75% of peak |
| Model confidence anomaly | >90% or <10% predicted probability on >10% of bets |
| Market regime change | Major AFL rule change, fixture anomaly, or broadcast disruption |

---

## Minimum Viable Metrics at Launch

If only three things are tracked from day one:

1. **Per-bet CLV** -- is the model identifying mispriced lines before they close?
2. **Brier score (rolling 200 bets)** -- are probabilities calibrated?
3. **Bankroll drawdown from peak** -- is stake management functioning?

These three metrics, computed weekly, are sufficient for a go/no-go decision
on any model modification.
