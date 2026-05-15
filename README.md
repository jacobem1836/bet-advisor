# bet-advisor

Stats-driven AFL betting advisor. Private research project.

This is **not** an arbitrage system (see [match-bet](https://github.com/) for that). This system builds statistical models, computes expected value against bookmaker prices, and produces paper-trading recommendations with rigorous evaluation (CLV, Brier, calibration).

**Status:** Phase 1 (research) complete. Phase 2 (data foundation) complete. Phase 3 (EV framework) complete. Phase 4 (disposals model + walk-forward backtest) complete. Phase 5 (recommendation engine + scheduler + markdown report) complete.

## What it does

Targets AFL only at MVP. Primary market: player disposals. Secondary: totals, H2H.
Reads odds from The Odds API and Betfair Exchange. Models in scikit-learn.
Stores in DuckDB + SQLite + Parquet.
Outputs daily markdown recommendations.
Tracks every prediction against the closing line.

## What it does not do

- Place bets automatically
- Cover sports other than AFL (yet)
- Recommend multis or same-game multis (yet)
- Promise edge — it runs in paper-trading mode and proves itself first

## Repo layout

See [`RESEARCH.md`](./RESEARCH.md) §7 for the structure and §8 for the roadmap.
Detailed research lives in [`research/`](./research/).

## Setup

```bash
cp .env.example .env
# fill in keys
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Philosophy

Optimise for accuracy, explainability, calibration, sustainable bankroll management.
Do not optimise for reckless strategies, fake certainty, or black-box outputs.
Probabilistic and uncertainty-aware throughout.

## Daily run

```bash
cp .env.example .env
# fill in ODDS_API_KEY and other keys
source .venv/bin/activate
bet-advisor schedule
```

Use `--dry-run` to preview the job schedule without starting:

```bash
bet-advisor schedule --dry-run
```

See [`docs/operations.md`](docs/operations.md) for the full CLI reference, environment variable table, and daily report format.

## License

Personal use. No license granted.
