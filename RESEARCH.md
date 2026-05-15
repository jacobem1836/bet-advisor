# bet-advisor – Research Synthesis

**Date:** 2026-05-15
**Status:** Phase 1 (research) complete. No code written yet.

This document synthesises five parallel research streams into a concrete recommendation for what to build, in what order, and what to deliberately *not* build. Detailed sources live in `research/01–05_*.md`.

---

## TL;DR

The brief asks for an 8-agent system covering 6 sports and many bookmakers. The research says: **reject most of that as premature**. Build a single deterministic Python pipeline on AFL + player disposals as the anchor market, with H2H/totals as secondary. LLM agents are not the right tool here — almost every component of a stats betting system is deterministic arithmetic or trained ML. The one place an LLM justifiably helps is parsing injury/team-selection news into structured signals.

The single biggest risk to the project is not modelling — it is **bookmaker account restrictions**. Sportsbet restricts winning accounts aggressively, especially on player props. Execution path must include Betfair Exchange from day one.

The single biggest *value* lever is **closing line value (CLV)**, not ROI. CLV becomes statistically meaningful in ~50 bets; ROI needs 600+. Architect everything to capture, store, and report CLV from bet #1.

---

## 1. Can this actually work?

### Honest answer

On **AFL H2H**, no — the market is sharp, and the 2018–2020 Brier analysis shows the closing odds beat every public model tested. Top public models (Massey Ratings, Squiggle aggregate) beat the close by margins of single tips across hundreds of games. The 2025–26 Squiggle leaderboard looks more optimistic but partial seasons overstate performance.

On **AFL player disposals**, plausibly yes — markets are softer, less academically studied, and disposal counts are highly autocorrelated by player role. This is the most credible MVP target.

On **AFL totals**, weak yes — weather-adjusted totals have theoretical edge (178 pts dry vs 136 wet on average) but public weather granularity is the limiting factor.

### Market priority

| Tier | Market | Reason |
|------|--------|--------|
| 1 | Player disposals (O/U) | Softest pricing, role-based autocorrelation, real edge plausible |
| 2 | Team totals | Weather and pace effects exploitable |
| 3 | Line betting | Thinner than H2H, less efficient on extreme favourites |
| 4 | H2H | Market is sharp; use as calibration benchmark, not primary edge |

**Do not start with multis/SGMs.** Bookmaker hold is 15–25% vs 4–7% on singles. Only revisit once single-bet CLV is consistently positive.

---

## 2. Architecture decision

### Rejected from the brief

- 8-agent LLM architecture. Only one role (news/injury parser) is LLM-justified.
- 6 sports at launch. AFL only.
- Multi-bookmaker normalisation layer at MVP. Sportsbet via The Odds API + Betfair Exchange is enough.
- Multi-leg / SGM builder. Deferred until single-bet edge is proven.
- Slack OS integration as a coupling point. Standalone with a clean integration surface (FastAPI or MCP) instead.
- Event-driven streaming for everything. 5-min polling is correct outside the final 60 min before bounce.

### Accepted

- Modular Python pipeline with sport adapters (so NRL/NBA can be added later without rewrite).
- Strong separation between: ingestion → storage → model → EV/stake → recommendation → evaluation.
- One LLM-backed module: news/injury parser (deterministic everywhere else).
- DuckDB for analytical/backtest layer, SQLite for operational signals + bet log, Parquet for raw odds archive.
- APScheduler for orchestration. Same pattern as Vinyl Scraper, no new infrastructure.

### High-level shape

```
   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐
   │  Ingestion   │──▶│   Storage     │──▶│   Models     │
   │ (odds, stats,│   │ (DuckDB +     │   │ (disposals,  │
   │  news, wx)   │   │  SQLite +     │   │  totals,     │
   └──────────────┘   │  Parquet)     │   │  H2H Elo)    │
                      └───────┬───────┘   └──────┬───────┘
                              │                  │
                              ▼                  ▼
                      ┌──────────────┐   ┌──────────────┐
                      │  Evaluator   │◀──│  EV + Kelly  │
                      │  (CLV, Brier,│   │  + bet log   │
                      │  calibration)│   │              │
                      └──────────────┘   └──────┬───────┘
                                                │
                                                ▼
                                        ┌──────────────┐
                                        │  Notifier    │
                                        │  (CLI/MCP/   │
                                        │   webhook)   │
                                        └──────────────┘
```

---

## 3. Recommended stack

| Layer | Choice | Why |
|-------|--------|-----|
| Language | Python 3.12+ | Ecosystem, your default |
| Odds (live) | The Odds API (paid, Standard tier ~$79/mo) | Coverage of Sportsbet AFL incl. player props, reliable, no ToS landmines |
| Odds (sharp ref) | Betfair Exchange via `betfairlightweight` | Free with AU account, sharper than fixed-odds books, closing line truth |
| Historical odds | AusSportsBetting XLSX + Betfair AU CSVs | Required for honest backtest, both free |
| Match results | AFL Tables via `pyAFL` | Free, 1897+ |
| Player stats | AFL Tables + FootyWire scrape | Free, disposals/marks/tackles etc. |
| Fixture feed | Squiggle API | Free, reliable |
| Injuries/team selection | FootyWire scrape + SEN/AFL.com.au monitoring for late mail | Final 22 lands ~6:20pm day before |
| Weather | Open-Meteo (no API key) | Free, ERA5 historical from 1940, forecast included |
| Modelling | scikit-learn (GradientBoosting + isotonic/Platt calibration), statsmodels for Elo | Boring, well-understood, debuggable |
| Storage | DuckDB (analytics), SQLite (operational), Parquet (raw) | Right-sized for ~9 games/week |
| Orchestration | APScheduler in long-running Python process | Already in your stack |
| Calibration | Platt scaling (preferred over isotonic for n < 500 games) | Smaller-sample regime |
| Bet sizing | Flat staking → quarter Kelly after 300 bets with confirmed +CLV | Robust to model uncertainty |
| Notifier | Start with CLI/Markdown report. Add MCP server or FastAPI later for Slack integration. | Avoid coupling to non-existent infra |
| LLM use | One narrow module: injury/team-news parsing → structured signal | Only justified use |

---

## 4. Reusable components from `match-bet`

Per the odds ingestion research, **`collector/odds_fetcher.py` (`OddsAPIClient`)** is essentially drop-in reusable. Port it as a library:
- `_get()`, `fetch_events()`, quota logging, `BOOKMAKERS` registry all transfer.
- Remove the H2H-only filter in `parse_back_odds()` to handle spreads/totals/props.
- Replace the `BackOdds` model with a richer relational schema (event, market, runner, price, timestamp, source).

**Do not modify match-bet.** Copy what's reusable into bet-advisor as a clean module.

---

## 5. Operational constraints (the parts most projects miss)

1. **Sportsbet restricts winners fast.** Especially on player props. Plan for Betfair Exchange execution from day one. The advisor will eventually need to recommend execution venue, not just bookmaker.
2. **Closing line capture is non-negotiable.** Every bet's CLV cannot be computed retroactively. Poll cadence in the last 30 min before bounce must be ≤ 1 min for the markets you trade.
3. **The 60-minute pre-bounce emergency sub** is unsolvable from public data alone. Mitigation: monitor club socials + SEN in the final hour. Pragmatic acceptance: this is a residual risk on player markets.
4. **Marvel Stadium is indoor when roof closed.** Weather features must zero out by venue + match.
5. **Sample size reality.** First 50–100 bets are noise. Do not adjust model based on early results. CLV is the only signal worth watching early.
6. **AU legal posture.** Private use, no auto-betting, no redistribution of bookmaker data, no scraping Sportsbet directly. Use The Odds API as the contracted intermediary.

---

## 6. Evaluation framework (day-1 metrics)

Three metrics start tracking from bet #1:

1. **CLV%** – devigged closing implied prob vs your bet's implied prob. Positive CLV consistently → real skill. Single most important number.
2. **Brier score** – per market, weekly rolling. Target < 0.25 on H2H, < 0.22 on disposals.
3. **Bankroll drawdown** – max from peak. Hard halt at 25%.

Weekly: ECE (expected calibration error), reliability diagram, ROI with Wilson CI, per-market Brier breakdown.

Monthly: full calibration audit; recalibrate (Platt) if ECE > 0.015.

Triggers for model pause: ECE > 0.02 post-recalibration, drawdown > 25%, CLV < 0 over rolling 100 bets, Brier deteriorates two consecutive months.

---

## 7. Repo layout (proposed)

```
bet-advisor/
├── RESEARCH.md                  # this file
├── research/                    # detailed research docs
├── pyproject.toml
├── README.md
├── .env.example
├── config/
│   └── venues.json              # static AFL venue metadata
├── src/bet_advisor/
│   ├── ingest/
│   │   ├── odds_api.py          # ported from match-bet
│   │   ├── betfair.py
│   │   ├── afl_tables.py
│   │   ├── footywire.py
│   │   ├── squiggle.py
│   │   ├── weather.py
│   │   └── news.py              # the one LLM-using module
│   ├── storage/
│   │   ├── duckdb_store.py      # analytics + backtest
│   │   ├── sqlite_store.py      # bets, signals, P&L
│   │   └── parquet_archive.py
│   ├── models/
│   │   ├── disposals.py         # primary MVP model
│   │   ├── totals.py
│   │   └── h2h_elo.py
│   ├── eval/
│   │   ├── devig.py             # proportional, power, Shin
│   │   ├── ev.py
│   │   ├── kelly.py
│   │   ├── calibration.py       # Brier, ECE, Platt
│   │   └── clv.py
│   ├── backtest/
│   │   └── walk_forward.py
│   ├── recommend/
│   │   └── engine.py            # combine model + EV + stake + log
│   ├── notify/
│   │   └── cli_report.py        # markdown daily card
│   ├── scheduler.py             # APScheduler entrypoint
│   └── main.py
├── notebooks/                   # exploratory only
└── tests/
```

---

## 8. Roadmap

Sequenced by dependency, not time.

**Phase 2 — Data foundation**
- Port `OddsAPIClient` from match-bet.
- Wire AFL Tables (pyAFL) + FootyWire + Squiggle + Open-Meteo + AusSportsBetting historical.
- DuckDB schema for matches, players, weather. SQLite for bets + signals.
- Backfill historical to 2009.
- Smoke test: query Sportsbet H2H odds + closing line for one upcoming round.

**Phase 3 — Devig + EV + bet log**
- Implement devig (power for H2H, Shin for props).
- EV calc, Wilson CI on win rate.
- Bet log schema with all fields the EV/staking research lists.
- CLV computation from stored closing snapshots.

**Phase 4 — Disposals model (MVP edge)**
- ewm(span=10) disposal average + opponent pressure rating + venue + home/away.
- GradientBoostingRegressor → predicted distribution → over/under prob.
- Platt calibration.
- Walk-forward backtest 2019–2024.
- Honesty check: does it produce +CLV in backtest with realistic vig?

**Phase 5 — Recommendation engine + CLI**
- Daily markdown report: event, market, model prob, devigged prob, edge, stake (flat 1u), rationale, risks.
- Bet log auto-population.
- Start paper trading.

**Phase 6 — Totals model + H2H Elo**
- Secondary markets.
- H2H Elo as calibration baseline, not primary signal.

**Phase 7 — Evaluation dashboard**
- Weekly CLV/Brier/ROI summary.
- Monthly calibration audit.
- Trigger logic for model pause.

**Phase 8 — News/injury LLM module**
- Only after Phase 5 is producing recommendations.
- Parses FootyWire/AFL.com.au/SEN into structured `{player, status, confidence, source, timestamp}`.
- Feeds into model as availability feature.

**Phase 9 — Betfair Exchange integration**
- Free streaming for closing line truth.
- Eventually for execution venue recommendation.

**Phase 10 — Integration surface**
- FastAPI `/signals/today` or MCP server.
- Slack/Discord notifier hooks into this, not the core.

**Deferred / rejected**
- Multi-leg/SGM builder.
- Multi-sport (NRL, NBA, soccer, horse racing).
- Multi-bookmaker normalisation beyond The Odds API + Betfair.
- Realtime websocket streaming outside final hour.
- 8-agent debate architecture.

---

## 9. Open questions for Jacob before Phase 2

1. **The Odds API subscription** – Standard tier (~$79 USD/mo) needed for full-round prop coverage. Confirm willingness to pay, or start on Starter tier (~$25) and limit to H2H/totals first.
2. **Betfair AU account** – do you have one? Required for the sharp closing line reference. Free but takes verification.
3. **Bankroll** – what notional bankroll do recommendations size against? Affects unit size and exposure caps, not strategy.
4. **Where do recommendations land** – daily markdown file? CLI command? Eventually Slack to your existing setup? (Doesn't block Phase 2, but shapes Phase 5.)
5. **Modelling sport scope confirmation** – AFL only until single-market +CLV is proven, then revisit. Confirm.

---

## 10. Key source documents

- `research/01_afl_modelling.md` – modelling landscape, feasibility, market priority
- `research/02_afl_data_sources.md` – every data source compared with access/cost/quality
- `research/03_odds_ingestion.md` – Sportsbet/The Odds API/Betfair, match-bet reuse
- `research/04_ev_staking_evaluation.md` – devig, EV, Kelly, calibration, backtest, CLV
- `research/05_architecture.md` – signum analysis, agent-vs-Python decision, storage/orchestration
