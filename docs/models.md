# Models

This document describes the models implemented in `src/bet_advisor/models/`.
For market priority rationale, see [RESEARCH.md](../RESEARCH.md) §1.
For backtest methodology, see [research/04_ev_staking_evaluation.md](../research/04_ev_staking_evaluation.md) §7.

---

## Disposals model (`models/disposals.py`)

### Purpose

Predicts the probability that a player's disposal count exceeds a given over/under line in an AFL match. Targets the AFL player disposals market -- identified in Phase 1 research as the softest AFL market with the highest credible edge potential.

H2H markets are not modelled here. They are calibration baselines, not primary betting targets.

### Features used

| Feature | Description | Why |
|---|---|---|
| `ewm_disposals_mean` | Player's ewm(span=10) disposal average over prior matches | Single most predictive feature per research |
| `ewm_disposals_std` | ewm std of player disposals | Measures consistency/volatility of role |
| `rolling_5_disposals_mean` | Mean disposals over last 5 matches | Short-term form |
| `rolling_3_disposals_mean` | Mean disposals over last 3 matches | Very short-term form |
| `opp_pressure_ewm` | Opponent team's ewm disposals-conceded rating | How much the opposing team limits opposition ball-users |
| `venue_disposal_delta` | Player's historical mean disposal delta at this venue vs their overall mean | MCG is larger; certain grounds suit certain roles |
| `indoor_venue` | Binary: venue is indoor (Marvel Stadium with roof closed) | Indoor eliminates weather effects |
| `is_home` | Binary: player's team is the home team | Home-ground familiarity and crowd effect |
| `days_since_last_match` | Days since player's last recorded match | Rest and injury recovery proxy |
| `ewm_tog_pct` | ewm(span=10) of player's time-on-ground percentage | Directly controls opportunity for disposals; heavily predictive per research |

All rolling and ewm features use `shift(1)` before the aggregation window. The current match's disposal count is never included in its own lookback. This is the primary guard against look-ahead bias in feature engineering.

The feature builder accepts an `as_of` date parameter for backtest use. Passing `as_of` restricts data to rows on or before that date before computing any rolling statistics.

### Model architecture

Two GradientBoostingRegressor models are trained in sequence:

1. **Mean GBR**: Predicts expected disposal count (regression target: actual disposals). Default hyperparameters: `n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8`.

2. **Std GBR**: Predicts conditional standard deviation by fitting on absolute residuals from the mean GBR. Default: `n_estimators=200, max_depth=3`. Std is floor-clipped at 1.0 to prevent degenerate distributions.

### Over/under probability approximation

P(disposals > line) is computed as `1 - Normal_CDF(line; mean, std)`.

**This is a known approximation.** Disposal counts are discrete non-negative integers. The true distribution has a heavier right tail than a Gaussian and is bounded below at zero. The approximation is adequate for lines in the range [10, 40] where AFL disposals markets are typically set. It degrades for extreme lines (< 5 or > 45).

A future improvement would be a Poisson or negative-binomial distributional model. Deferred until the base regression is validated.

### Calibration

After fitting, the model calibrates the over/under probabilities on a held-out slice of the training data (default: last 15% by row order, preserving temporal structure).

**PlattCalibrator is the default.** It is a logistic regression fit on raw over/under probabilities. Preferred for AFL sample sizes (< 1000 player-matches per training window) because it has fewer degrees of freedom than isotonic regression and is less prone to overfitting. See Phase 1 research §3.5.

**IsotonicCalibrator** is available via `ModelConfig(calibrator_type="isotonic")`. Recommended only once 1000+ calibration samples are available (player prop markets at scale).

If the calibration slice contains only one outcome class at the reference line (e.g. all bets are "over"), calibration is skipped and a warning is logged. Raw probabilities are returned in that case.

### When to retrain

Retrain when any of the following occur:
- New season begins (refit on all prior seasons).
- ECE exceeds 0.015 on the last 200 predictions (recalibrate first; retrain if ECE remains elevated after recalibration).
- Brier score deteriorates two consecutive monthly rolling windows.
- A major rule change affects disposal-count distributions (e.g. interchange limit changes).

The `version_hash` property returns a 16-character hex hash of the model configuration (feature spec + hyperparameters). Use this to identify which configuration produced a given prediction in the bet log. Note: the hash does not incorporate training data or date. Two models trained on different windows with the same config share the same hash.

---

## Totals model (`models/totals.py`)

Phase 6 placeholder. Raises `NotImplementedError`. See RESEARCH.md §8 for the roadmap entry.

---

## H2H Elo model (`models/h2h_elo.py`)

Phase 6 placeholder. Raises `NotImplementedError`. H2H markets are the sharpest AFL markets (Brier ~0.201 per research). This model exists as a calibration reference baseline, not as a primary betting target.
