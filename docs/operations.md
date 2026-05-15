# Operations

This page covers how to run the bet-advisor daily pipeline, what environment variables are required, what the daily report contains, and how to use the CLI commands.

See also [`docs/evaluation.md`](evaluation.md) for CLV, calibration, and edge-evaluation methodology, and [`RESEARCH.md`](../RESEARCH.md) §8 for the Phase 5 scope.

---

## Required Environment Variables

Copy `.env.example` to `.env` and fill in the values before running.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ODDS_API_KEY` | Yes | - | The Odds API v4 key |
| `ODDS_API_MONTHLY_BUDGET` | No | 500 | Total monthly request budget across all uses |
| `ODDS_API_PROJECT_SHARE` | No | 200 | Requests allocated to this project per month |
| `ODDS_API_PROJECT_TAG` | No | `bet-advisor` | Tag used for quota tracking in SQLite |
| `BANKROLL` | No | 1000.0 | Starting bankroll in currency units |
| `STAKE_MODE` | No | `flat` | Staking mode: `flat`, `quarter_kelly`, `capped_kelly` |
| `SQLITE_PATH` | No | `data/operational.db` | Path to SQLite operational store |
| `DUCKDB_PATH` | No | `data/analytics.duckdb` | Path to DuckDB analytical store |
| `REPORTS_DIR` | No | `reports` | Output directory for daily markdown reports |
| `MODEL_PATH` | No | `models/disposals_latest.joblib` | Path to fitted DisposalsModel |

---

## Running the Scheduler

The scheduler is the primary production entrypoint. It manages all odds polling and report generation.

```bash
bet-advisor schedule
```

Use `--dry-run` to inspect the planned schedule without starting:

```bash
bet-advisor schedule --dry-run
```

The scheduler registers five jobs:

| Job | Trigger | Description |
|-----|---------|-------------|
| `refresh_odds_low_freq` | Every 60 min | Fetches all AFL odds outside match windows |
| `refresh_odds_high_freq` | Every 15 min | Fetches odds in the 2-hour pre-bounce window |
| `refresh_odds_final_min` | Every 60 sec | Fetches odds in the final 30 min for tracked markets only |
| `ingest_results` | Daily 21:00 AEST | Pulls match results from Squiggle; settles pending bets |
| `daily_report` | Daily 08:00 AEST | Generates the markdown report |

All times are in `Australia/Brisbane` (AEST, UTC+10, no DST).

### Quota enforcement

Every job that touches the Odds API checks MTD usage before each call:

- If `MTD usage >= ODDS_API_PROJECT_SHARE`, the job logs a warning and skips the API call.
- Quota is tracked per-project-tag in the SQLite `quota_usage` table, so it survives restarts.
- The `refresh_odds_final_min` job only polls events with active (pending) signals to minimise API consumption near bounce.

---

## CLI Commands

### `bet-advisor recommend --round N`

Generate recommendations for AFL round N and print them to stdout.

```bash
bet-advisor recommend --round 12
bet-advisor recommend --round 12 --persist        # write to SQLite
bet-advisor recommend --round 12 --allow-untrained  # bypass trained-model guard
```

The `--persist` flag writes signals and bets to SQLite. If the model is not trained, `--persist` is blocked unless `--allow-untrained` is also passed.

**Untrained model guard:** if the model has `is_trained == False`, the engine raises `UntrainedModelError`. This prevents accidentally logging paper bets from an untrained model. Pass `--allow-untrained` to bypass the guard for smoke testing, but do not use `--persist` in that state.

### `bet-advisor report --date YYYY-MM-DD`

Render and print the daily markdown report for a given date.

```bash
bet-advisor report --date 2026-05-15
bet-advisor report --date 2026-05-15 --output-dir /tmp/reports
```

The report is written to `REPORTS_DIR/<YYYY-MM-DD>.md` and printed to stdout.

### `bet-advisor quota`

Print MTD Odds API usage vs the project budget.

```bash
bet-advisor quota
```

Output shows project tag, monthly budget, project share, MTD used, and remaining.

### `bet-advisor health`

Print model health summary from the most recent snapshot in the `model_health` table.

```bash
bet-advisor health
```

Output shows version, calibration metrics (Brier, ECE, log loss), drawdown, and trigger flags.

Trigger flags that fire:
- `ECE_HIGH` -- ECE > 0.02 post-recalibration (recalibrate the model)
- `DRAWDOWN_HIGH` -- drawdown > 25% from peak bankroll (reduce stakes)
- `CLV_NEGATIVE` -- mean CLV over rolling 100 bets is negative (review model)
- `BRIER_DETERIORATED` -- Brier score has worsened >0.02 over 60 days (retrain)

### `bet-advisor backtest`

Run the walk-forward backtest via `scripts/run_backtest.py`.

```bash
bet-advisor backtest
bet-advisor backtest --config path/to/config.json
```

### `bet-advisor schedule --dry-run`

Print the planned schedule without starting the scheduler. Useful for verifying configuration.

---

## Daily Report Contents

Reports are written to `REPORTS_DIR/<YYYY-MM-DD>.md`. Each report contains:

1. **Header** -- date, bankroll, bets today, staked today.
2. **Top Recommendations table** -- ranked by edge: event, market, runner, bookmaker, odds, model probability, devigged market probability, edge, stake (% and $), confidence tier.
3. **Recommendation Detail** -- per-recommendation block with rationale bullets, counterarguments, EV, Kelly fraction, and stake breakdown.
4. **Bankroll & P&L Snapshot** -- today/week/month/all-time bet counts, staked amounts, all-time ROI with Wilson 95% CI, mean CLV, and % positive CLV.
5. **Model Health** -- last calibration snapshot version and metrics, drawdown from peak, trigger summary.
6. **Risk Flags** -- active triggers from model health, speculative bet count warnings.
7. **API Quota** -- Odds API MTD requests used vs budget.

---

## Where Outputs Land

| Artifact | Location |
|----------|----------|
| Daily report | `REPORTS_DIR/<YYYY-MM-DD>.md` |
| SQLite store (events, odds, signals, bets, quota, model health) | `SQLITE_PATH` |
| DuckDB analytical store (player stats, features) | `DUCKDB_PATH` |
| Fitted model | `MODEL_PATH` |
| Parquet archives | `data/parquet/` |
