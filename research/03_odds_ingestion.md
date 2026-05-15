# 03 – AFL Odds Ingestion Research

**Project:** bet-advisor (stats-driven AFL betting advice)
**Date:** 2026-05-15
**Scope:** Sportsbet AU and adjacent AU bookmakers – current odds, closing line capture, data modelling

---

## 1. Sportsbet AFL Market Catalogue

### Core Markets

| Market | The Odds API key | Availability | Notes |
|---|---|---|---|
| Head-to-head (H2H) | `h2h` | Monday–Thursday before game week; sometimes earlier | Most liquid. Both teams only, no draw in AFL. Overround typically 5–8% at Sportsbet; sharper books run closer to 4–5%. |
| Handicap / Line | `spreads` | Same as H2H | Line moves significantly through week. AU books set lines in 6-point increments. |
| Totals (combined score) | `totals` | Same as H2H | Sportsbet uses points totals in half-point increments. Typical overround 6–9%. |
| Quarter / half H2H | `h2h_q1`, `h2h_q2`, etc. | Day of game, sometimes day prior | Covered by The Odds API as additional markets. |

### Player Prop Markets

Sportsbet offers an extensive player prop suite for AFL. Markets open on the day of the game or the evening before (for Friday/Saturday games). Liquidity is thin compared to H2H.

| Market | The Odds API key | Coverage |
|---|---|---|
| Player disposals over/under | `player_disposals` | Sportsbet, Ladbrokes, TAB, Pointsbet, Betr |
| Player disposals over only | `player_disposals_over` | Same as above |
| Anytime goalscorer | `player_goal_scorer_anytime` | Sportsbet, Ladbrokes, TAB |
| First goalscorer | `player_goal_scorer_first` | Same |
| Last goalscorer | `player_goal_scorer_last` | Same |
| Goals over only | `player_goals_scored_over` | Limited |
| Marks over | `player_marks_over` | Select books |
| Tackles over | `player_tackles_over` | Select books |
| Handballs over | `player_handballs_over` | Select books |
| Kicks over | `player_kicks_over` | Select books |
| AFL Fantasy Points O/U | `player_afl_fantasy_score`, `player_afl_fantasy_score_over` | Select books |
| Most Marks / Most Tackles | `player_marks_most`, `player_tackles_most` | Betfair Exchange; some books |
| Same Game Multi (SGM) | N/A – not available via API | Sportsbet-proprietary composite; not standardised |

**Critical prop market caveat:** Player prop markets load on game day. If a player is named sub or a late out, Sportsbet voids or adjusts the market. Markets are suspended close to bounce while the team sheet is confirmed. The window between team announcement (~60 minutes pre-game for AFL) and bounce is when final prop odds settle – this is the closing window to capture.

### Market Sharpness

AU fixed-odds books (Sportsbet, TAB, Bet365, Ladbrokes) are soft-to-medium sharp. Their H2H markets follow Betfair Exchange prices with a lag and a margin added. Betfair Exchange AFL MATCH_ODDS is the sharpest publicly accessible price source; exchange prices converge toward true probability faster than any AU fixed book. Research from AusSportsBetting and Betfair's own hub confirms that sharp money on exchange moves first, fixed-odds books follow.

Overround benchmarks (from community research across AU seasons):
- H2H: ~104–107% (Sportsbet), ~103–105% (TAB), ~102–104% (Bet365)
- Line/spreads: ~106–109%
- Totals: ~106–110%
- Player props (disposals O/U): ~108–115% – these are the softest markets and the highest priority for a stats-driven advisor

---

## 2. Access Methods

### 2a. The Odds API (Recommended primary path)

The Odds API ([the-odds-api.com](https://the-odds-api.com/)) is the cleanest programmatic route. match-bet already uses it. It aggregates Sportsbet, TAB, Ladbrokes, Neds, Pointsbet, Betr, Bet365, Unibet, and Betfair.

**How the credit system works:**

Each call to `/sports/{sport}/odds` costs `num_markets × num_regions` credits. A single call for `aussierules_afl` with `markets=h2h,spreads,totals` and `regions=au` costs **3 credits**. Player props add 1 credit each per market key.

Historical snapshots cost **10 credits per market per region**.

| Plan | Credits/month | Historical | Cost |
|---|---|---|---|
| Free | 500 | No | $0 |
| Starter (~$25/mo) | ~20,000 | Yes | ~$25 USD |
| Standard (~$79/mo) | ~100,000 | Yes | ~$79 USD |
| Pro (~$199/mo) | ~500,000 | Yes | ~$199 USD |

(Exact pricing: [the-odds-api.com](https://the-odds-api.com/))

**AFL-specific endpoints:**

```
GET /v4/sports/aussierules_afl/odds
  ?regions=au
  &markets=h2h,spreads,totals,player_disposals,player_goal_scorer_anytime
  &bookmakers=sportsbet,tab,ladbrokes,pointsbet,betr
  &oddsFormat=decimal
  &dateFormat=iso
```

**Latency:** The Odds API polls bookmakers and caches results. Latency between a real price change at Sportsbet and the API reflecting it is typically 1–5 minutes for featured markets. Player prop updates may lag longer. This is not suitable for live in-play capture but is fine for pre-game closing line capture with a 5-minute poll cadence.

**Historical odds:** Available from June 2020 for featured markets (5-minute snapshots from September 2022). Player prop history from May 2023. This is the most accessible historical dataset for AFL.

**Pros:**
- Already integrated in match-bet (direct library reuse, see Section 9)
- Covers all AU bookmakers in one call
- Player prop coverage for AFL is genuine and tested
- Clean JSON, decimal odds, ISO timestamps
- Historical endpoint for backtesting

**Cons:**
- Not real-time; up to 5 minutes stale on featured markets
- Credit cost climbs fast with multiple prop markets
- Historical data costs 10× the live credit rate
- No SGM access

### 2b. Sportsbet Direct – Undocumented API

Community research (particularly the [sportsbook-odds-scraper](https://github.com/declanwalpole/sportsbook-odds-scraper) project) confirms Sportsbet exposes undocumented internal API endpoints. The scraper calls these rather than parsing HTML. However:

- Sportsbet requires **a separate HTTP request per market grouping** – it is "very slow to scrape" compared to multi-bookmaker APIs
- Sportsbet operates behind Cloudflare with bot detection (TLS fingerprinting, behavioural analysis, IP reputation)
- As of 2025-2026, vanilla Playwright in headless mode is reliably detected by Cloudflare. Stealth plugins (playwright-stealth, puppeteer-stealth) were deprecated February 2025 and no longer bypass current Cloudflare versions
- Effective bypass now requires paid residential proxy rotation + anti-detect browser APIs (Camoufox, Nodriver, SeleniumBase UC Mode, or commercial products like ZenRows/ScrapFly/BrightData)
- No public documentation of endpoint structure. Reverse-engineering requires browser DevTools network tab inspection per market type

The [Apify Sportsbet scraper](https://apify.com/lexis-solutions/sportsbet-com-au-scraper) exists as a managed alternative (~$25/month + usage) but AFL player prop coverage is unconfirmed.

**Verdict:** DIY direct Sportsbet scraping is high-friction, ToS-violating (see Section 8), and unstable. Not recommended as a primary source. May be useful as a point-in-time spot check.

### 2c. PuntersEdge API

[PuntersEdge](https://puntersedge.online/api-platform) is an AU-focused aggregator built as a The Odds API alternative. Covers 9 AU bookmakers including Sportsbet, TAB, Ladbrokes, Neds, Pointsbet, BlueBet, Betfair.

| Plan | Rate limit | Cost (AUD) |
|---|---|---|
| Free | 100 req/hour | $0 |
| Pro | 1,000 req/hour | $29/month |
| Pro Plus | 5,000 req/hour | $99/month |
| Unlimited | 100,000 req/hour | $249/month |

Pro Plus includes "full price movement history per selection" – opening odds, closing odds, line movement tracking.

Player prop coverage for AFL is **not confirmed** in public documentation. The API claims "full market coverage" for AFL vs NRL but does not enumerate specific prop market keys. Requires direct verification before committing.

**Verdict:** Promising AU-native alternative. Worth testing against The Odds API for AFL prop coverage depth. If prop coverage is confirmed, Pro Plus tier at $99 AUD/month gives movement history without a separate historical endpoint cost.

### 2d. Odds-API.io

[odds-api.io](https://odds-api.io/) covers 265+ bookmakers across 34 sports with WebSocket streaming. Free tier includes 5,000 requests/hour. Sportsbet AU coverage is confirmed on their bookmaker page. Specific AFL market depth is not documented in public-facing content. A low-risk experiment given the free tier.

---

## 3. The Odds API – Current Odds, AFL Summary

Sport key: `aussierules_afl`  
Regions: `au`  
Supported AU bookmakers: sportsbet, tab, ladbrokes, neds, pointsbet, betr, bet365, unibet, betfair_ex_au  

Confirmed AFL player props: `player_disposals`, `player_disposals_over`, `player_goal_scorer_anytime`, `player_goal_scorer_first`, `player_goal_scorer_last`, `player_goals_scored_over`, `player_marks_over`, `player_tackles_over`, `player_handballs_over`, `player_kicks_over`, `player_afl_fantasy_score`, `player_marks_most`, `player_tackles_most`, `player_clearances_over`

Credit cost for a full AFL pre-game snapshot (H2H + spreads + totals + 4 prop markets, au region only): **7 credits per call**.

At a 5-minute polling cadence over a 2-hour pre-game window: 24 calls × 7 credits = **168 credits per game**. With 9 games per round, that's ~1,500 credits per round – well inside any paid plan.

Free tier (500 credits/month) is only viable for intermittent spot checks or single-market polling during finals.

---

## 4. OddsPortal, OddsChecker, AusSportsBetting

### OddsPortal

[OddsPortal](https://www.oddsportal.com/aussie-rules/australia/afl/) provides AFL historical odds with full bookmaker breakdown including Sportsbet. Displays opening price through closing price (at game time). The site is scrapeable in principle but:

- Requires JavaScript rendering (React-based SPA)
- Employs rate limiting and bot detection
- No public API; scraping violates ToS
- The [OddsHarvester](https://github.com/jordantete/OddsHarvester) Python project scrapes OddsPortal using Playwright but requires maintenance as OddsPortal updates its selectors

For historical data, OddsPortal is **less convenient than AusSportsBetting** for AU markets specifically.

### AusSportsBetting

[aussportsbetting.com/data/historical-afl-results-and-odds-data/](https://www.aussportsbetting.com/data/historical-afl-results-and-odds-data/) provides free Excel downloads of historical AFL results and bookmaker odds. The dataset is:

- Free to download (no API, manual download or `requests` + direct file URL)
- Covers Sportsbet, TAB, and other AU books
- Includes opening and closing odds for H2H markets per game
- Updated through the current season
- Labelled "for personal use only; do not redistribute"

**Limitations:** H2H only. No player props, no line, no totals. No intra-week snapshots (just open and close per game). No programmatic API – scraping the download link is technically feasible via `requests` but the ToS discourages redistribution.

**Best use:** One-time historical backtest seed. Download the Excel file, convert to CSV/Parquet, load into the database as the historical baseline for H2H closing line analysis.

### OddsChecker

[oddschecker.com/australian-rules/afl](https://www.oddschecker.com/australian-rules/afl) provides AU AFL odds comparison. UK-based; AU bookmaker coverage is present but not primary. No public API. JavaScript-rendered. Lower priority than The Odds API or AusSportsBetting for this project.

---

## 5. Betfair Exchange AU

### Why It Matters

Betfair Exchange is the sharpest publicly accessible price source for AFL in Australia. Because bettors set prices against each other (no bookmaker margin), exchange prices converge toward true probability faster than any fixed-odds book. **The closing Betfair Exchange price for MATCH_ODDS is the standard reference for CLV calculation** – it is what professional AFL bettors benchmark against.

Key properties:
- Commission on winnings: 5% standard AU rate (can reduce to ~2% at volume)
- No overround – market sums to ~100% minus commission
- Event type ID for Australian Rules: **61420**
- Exchange AFL markets include: MATCH_ODDS, HANDICAP (Asian Handicap), OVER_UNDER (total points), MATCH_ODDS_LAY, NEXT_GOAL, player-specific markets via Betfair Sportsbook (not exchange)

### API Access

The Betfair Exchange API-NG is **free for any active AU account holder**. No special developer tier required beyond obtaining an App Key through the [Developer Program](https://developer.betfair.com/).

Requirements:
- Verified Betfair AU account
- App Key (delayed key = free; live key requires active betting history)
- SSL certificates (self-signed, generated locally)

### betfairlightweight Python Library

[betfairlightweight](https://pypi.org/project/betfairlightweight/) (maintained by [betcode-org](https://github.com/betcode-org/betfair)) is the standard Python wrapper. Supports both REST and **Streaming API**.

```python
import betfairlightweight
from betfairlightweight.filters import (
    streaming_market_filter,
    streaming_market_data_filter,
)

trading = betfairlightweight.APIClient(
    username="...",
    password="...",
    app_key="...",
    certs="/path/to/certs",
)
trading.login()

# List AFL events
event_types = trading.betting.list_event_types()
# Australian Rules = event type ID 61420

# Get upcoming MATCH_ODDS markets
markets = trading.betting.list_market_catalogue(
    filter=betfairlightweight.filters.market_filter(
        event_type_ids=["61420"],
        market_types=["MATCH_ODDS"],
    ),
    market_projection=["RUNNER_DESCRIPTION", "MARKET_START_TIME"],
    max_results=50,
)

# Get current prices
books = trading.betting.list_market_book(
    market_ids=[m.market_id for m in markets],
    price_projection=betfairlightweight.filters.price_projection(
        price_data=["EX_BEST_OFFERS"]
    ),
)
```

**Streaming API** – preferred for real-time closing line capture:

```python
socket = trading.streaming.create_stream()
market_filter = streaming_market_filter(
    event_type_ids=["61420"],
    market_types=["MATCH_ODDS"],
)
data_filter = streaming_market_data_filter(
    fields=["EX_ALL_OFFERS", "EX_MARKET_DEF"],
    ladder_levels=3,
)
socket.subscribe_to_markets(
    market_filter=market_filter,
    market_data_filter=data_filter,
)
socket.start()
```

### Rate Limits

Betfair removed the per-second data request limit. The REST API weight system applies: each request must not exceed **200 weight points** across market IDs (relevant for batch market book calls). The Streaming API is preferred for continuous monitoring – no polling overhead.

The Streaming API limits subscriptions to **200 markets per connection**. A single AFL round has at most ~9 match odds markets, well within this.

Historical Data API: 100 requests per 10 seconds.

### AFL Betfair Historical Data (Free CSV Downloads)

Betfair AU publishes free CSV downloads of AFL market data through [betfair-datascientists.github.io/data/dataListing/](https://betfair-datascientists.github.io/data/dataListing/). Coverage: 2021–2026, "All Markets" and "Match Odds Markets" files. These are time-stamped exchange prices suitable for backtesting CLV models. Free, no API key required.

---

## 6. Closing Line Capture Strategy

### Why Closing Line Is the Key Data Point

The closing price (last traded / best available at market suspension before bounce) is the most informationally efficient price. It reflects all sharp money, late team news (e.g., late out), weather, and injury information. Positive Closing Line Value (CLV) – getting a better price than the market closed at – is the primary long-run indicator of a bettable edge in sports betting.

For AFL, the critical closing events:
- **Team sheet announcement:** ~60 minutes pre-bounce (Thursday evening for Thursday games, Friday 5pm for Friday games, etc.). Late outs cause significant line movement.
- **Final market move:** Betfair Exchange prices continue to move until bounce. Fixed-odds books often suspend 5–15 minutes before bounce.

### Polling Cadence

**Phase 1 – Week-long baseline** (Monday to Wednesday of game week)  
Poll once every 30–60 minutes via The Odds API. Track opening prices and mid-week movements for line markets. 1 credit per poll × 7 markets = 7 credits per poll.

**Phase 2 – Team announcement window** (Thursday or Friday, 2 hours before sheets drop)  
Increase to every 5–10 minutes. This captures movement from injury rumours and market leader moves.

**Phase 3 – Team sheet to bounce** (60 minutes pre-game to bounce)  
Poll every 2–5 minutes via The Odds API (for Sportsbet price) and **stream Betfair Exchange** continuously. Store each snapshot. This 60-minute window is where closing line is established.

**Phase 4 – Market suspended**  
Betfair Exchange continues trading until bounce whistle. Poll REST API for final book state 30 seconds before expected bounce if streaming is not running.

**What to store per snapshot:**
```
event_id, market_type, runner_name, bookmaker, decimal_odds,
implied_probability, snapshot_timestamp, source (api | betfair_stream | betfair_rest),
market_status (open | suspended | closed)
```

### Betfair Exchange as Closing Reference

Use the **last available best back price on Betfair MATCH_ODDS** at market suspension as the closing line reference price. This is standard in serious CLV analysis. For AFL disposals, the closest exchange equivalent is "Most Disposals" head-to-head markets on Betfair (event type 61420, market type "DISPOSAL_MATCH_BET" or "PLAYER_PROP" – market names vary, filter by name containing the player name).

---

## 7. Data Model

### Recommended Schema (PostgreSQL)

```sql
-- Normalised event registry
CREATE TABLE events (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id TEXT NOT NULL,          -- The Odds API event ID or Betfair market ID
    source      TEXT NOT NULL,          -- 'odds_api' | 'betfair'
    sport       TEXT NOT NULL DEFAULT 'afl',
    home_team   TEXT,
    away_team   TEXT,
    commence_at TIMESTAMPTZ NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now(),
    UNIQUE (external_id, source)
);

-- Market type per event
CREATE TABLE markets (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_id    UUID NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    market_type TEXT NOT NULL,          -- 'h2h' | 'spreads' | 'totals' | 'player_disposals' | ...
    description TEXT,                   -- e.g. player name for prop markets
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Individual runner/selection within a market
CREATE TABLE runners (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    market_id   UUID NOT NULL REFERENCES markets(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,          -- team name, player name, "Over", "Under"
    line        NUMERIC,                -- handicap or O/U line value if applicable
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Price snapshots (time-series core table)
CREATE TABLE price_snapshots (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    runner_id       UUID NOT NULL REFERENCES runners(id) ON DELETE CASCADE,
    bookmaker       TEXT NOT NULL,      -- 'sportsbet' | 'tab' | 'betfair_ex_au' | ...
    decimal_odds    NUMERIC(8,4) NOT NULL,
    implied_prob    NUMERIC(8,6) GENERATED ALWAYS AS (1.0 / decimal_odds) STORED,
    snapshot_at     TIMESTAMPTZ NOT NULL,
    source          TEXT NOT NULL,      -- 'odds_api' | 'betfair_stream' | 'betfair_rest'
    market_status   TEXT DEFAULT 'open' -- 'open' | 'suspended' | 'closed'
);

-- Closing line record (written once per market per bookmaker at suspension)
CREATE TABLE closing_prices (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    runner_id       UUID NOT NULL REFERENCES runners(id),
    bookmaker       TEXT NOT NULL,
    decimal_odds    NUMERIC(8,4) NOT NULL,
    implied_prob    NUMERIC(8,6) GENERATED ALWAYS AS (1.0 / decimal_odds) STORED,
    closed_at       TIMESTAMPTZ NOT NULL,
    source          TEXT NOT NULL,
    UNIQUE (runner_id, bookmaker)
);

-- Devigged (fair probability) per snapshot batch
CREATE TABLE fair_prices (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    market_id       UUID NOT NULL REFERENCES markets(id),
    snapshot_at     TIMESTAMPTZ NOT NULL,
    method          TEXT NOT NULL,      -- 'power' | 'multiplicative' | 'additive' | 'shin'
    runner_id       UUID NOT NULL REFERENCES runners(id),
    fair_prob       NUMERIC(8,6) NOT NULL,
    overround       NUMERIC(6,4) NOT NULL
);
```

**Indexes to add:**
```sql
CREATE INDEX ON price_snapshots (runner_id, snapshot_at);
CREATE INDEX ON price_snapshots (snapshot_at);
CREATE INDEX ON closing_prices (runner_id);
```

### Bookmaker Key Normalisation

Use The Odds API key names as canonical identifiers. They match what match-bet already uses:

| Canonical key | Display name |
|---|---|
| `sportsbet` | Sportsbet |
| `tab` | TAB |
| `ladbrokes` | Ladbrokes |
| `neds` | Neds |
| `pointsbet` | PointsBet |
| `betr` | Betr |
| `bet365` | Bet365 |
| `betfair_ex_au` | Betfair Exchange AU |
| `unibet` | Unibet |

Betfair Exchange prices are stored in decimal format already. AU fixed-odds books publish decimal.

### Devigging (Removing Overround)

To calculate fair probability from a set of raw bookmaker odds:

**Multiplicative (simplest, used for H2H):**
```python
implied_probs = [1 / o for o in odds]
total = sum(implied_probs)
fair_probs = [p / total for p in implied_probs]
```

**Power method (better for favourites):**  
Find exponent `k` such that `sum((1/o)^k) = 1.0`. Requires numerical root-finding.

**Shin method** – most theoretically grounded; use for prop markets with high margin.

For MVP, implement multiplicative for H2H/spreads/totals and flag prop market devig as a follow-on task.

---

## 8. Operational Risks – Candid Assessment

### Sportsbet ToS

Sportsbet's Feed Terms and Conditions (help centre, 20973619526925) state:

> "The Live Feed is prohibited from being communicated, resupplied, copied, stored or otherwise exploited, including for any public place or purpose other than personal viewing by the Member."

Their general Terms and Conditions prohibit automated or systematic access to the platform. Scraping Sportsbet directly – even without an account – constitutes a breach of their contractual terms if you've agreed to them (i.e. if you have a Sportsbet account and use it to access markets). Without an account, you are accessing a public-facing website but still subject to their ToS as a condition of use.

**AU Legal Exposure:**  
Web scraping is not per se illegal in Australia. The key legal vectors are:
1. **Contract law** – if you have a Sportsbet account and scrape, you breach the ToS (civil, not criminal)
2. **Copyright** – odds compilations may attract thin copyright under the *IceTV* line of cases; practically, AU courts have not enforced this against personal scrapers
3. **Computer Fraud / Unauthorised Access** – the *Criminal Code Act 1995 (Cth)* s 477.1 covers unauthorised access to computers; bypassing Cloudflare bot protection to access a system "without authorisation" is a grey area but unlikely to be pursued against a private individual doing personal research
4. **Privacy Act 1988** – not engaged if you are only capturing public prices, not personal data

**Practical risk with no bets placed:**  
Sportsbet's primary concern is protecting their prices from real-time redistribution to other operators and from arb bettors hammering their prices. A private individual capturing odds for personal analysis and not placing bets at Sportsbet is low-priority for enforcement. The main actual risk is IP banning (trivially bypassed) and account closure if your account shows unusual activity. If you do not have a Sportsbet account and are only reading the public site, there is no account to close.

**Recommended mitigation:** Use The Odds API or PuntersEdge as the primary data source. This routes through aggregators who have commercial relationships with the bookmakers (or at minimum, have accepted the risk themselves). You receive the same prices without direct ToS exposure.

### IP and Rate Limiting

The Odds API: No published rate limit beyond the credit quota. Honour the quota headers in responses (`x-requests-remaining`, `x-requests-used`) – match-bet's `OddsAPIClient._get()` already logs these.

Betfair Exchange: Streaming API is preferred. REST polling at 1-second intervals will hit weight limits; 10-second intervals are safe. Streaming is push-based and avoids the polling concern entirely.

### Account Flagging

Using the Betfair Exchange API does not flag your account for sharp-bettor restrictions. The API is Betfair's official data product and they encourage its use. Using it to place automated lay bets at high volume could trigger review, but read-only streaming is entirely clean.

---

## 9. Match-Bet Codebase Reuse

The match-bet project at `/Users/jacobmarriott/Documents/Personal/projects/match-bet/` provides three components that are directly reusable.

### Directly Reusable as a Library

**`collector/odds_fetcher.py` – `OddsAPIClient`**

The `OddsAPIClient` class is a clean httpx-based wrapper with:
- `_get()` with API key injection and quota logging
- `fetch_events(sport_key, regions, bookmaker_keys)` returning raw event dicts
- `parse_back_odds(events, sport)` returning `BackOdds` dataclass list

For bet-advisor, you need:
- The same `_get()` and `fetch_events()` logic – copy exactly or extract to shared package
- `parse_back_odds()` needs extending to handle additional market types beyond `h2h` – currently it filters `if market.get("key") != "h2h": continue` (line 132). Remove that filter and add per-market-type parsing

**`collector/bookmakers.py` – `BOOKMAKERS` registry and `Bookmaker` dataclass**

The bookmaker registry is directly reusable. The `SPORT_KEY_MAP`, `BOOKMAKERS` dict, and `get_bookmaker()` helper can be imported unchanged. The `market_url()` method on `Bookmaker` works for deep-linking to Sportsbet's AFL section.

`ACTIVE_SPORT_KEYS` already includes `"aussierules_afl"` as the second priority.

**`config.py` – Config pattern**

The `Config` dataclass / `load_config()` pattern is worth replicating. bet-advisor will need additional fields: `betfair_username`, `betfair_password`, `betfair_app_key`, `betfair_certs_path`, `closing_poll_cadence_seconds`, `puntersedge_api_key` (if used).

### Needs Rewriting / Extension

**`collector/odds_fetcher.py` – `OddsCollector` facade and `parse_back_odds()`**

`OddsCollector.fetch_all()` is hardcoded to `h2h` only. bet-advisor requires multi-market polling. The facade should be rewritten to:
1. Accept a list of market keys to fetch
2. Store snapshots to the database rather than returning a flat list
3. Track `snapshot_at` timestamps per call

**`models.py`**

`BackOdds` is too thin for bet-advisor – no timestamp, no market type beyond the implied sport context, no line value. The schema in Section 7 replaces this with proper relational models. Do not attempt to extend `BackOdds`; the new schema is structurally different.

`ArbOpportunity` and `ValueOpportunity` are not relevant to bet-advisor's advice model and should not be ported.

**Betfair streaming**

match-bet has no Betfair integration at all. This is net-new for bet-advisor. Use `betfairlightweight` directly.

### What Not to Touch

Do not modify match-bet. Extract the reusable pieces by copying files or creating a shared internal package. The two projects have different data models and the coupling is not worth the fragility.

---

## 10. Recommended Ingestion Stack for MVP

### Tier 1 – Primary Live Odds: The Odds API

Use The Odds API for all current Sportsbet and multi-bookmaker prices. Rationale:

- match-bet integration already exists and is tested
- Covers H2H, line, totals, and AFL player props from Sportsbet in a single authenticated call
- No ToS exposure
- Historical endpoint available on any paid plan for backtesting

**Start on the $25/month Starter plan.** Credit budget for weekly polling (5-minute cadence, 7 markets, 2-hour pre-game window, 9 games/round): ~13,500 credits/round. Starter plan supports ~3–4 full rounds/month before hitting quota. At $79/month Standard plan you cover a full season comfortably.

**Concrete polling schedule using APScheduler (already in Jacob's stack):**
- Monday to Wednesday: every 30 minutes, H2H + spreads + totals only (3 credits/call)
- Thursday/Friday (2 hours before expected team announcement): every 10 minutes, add prop markets (7 credits/call)
- Game day, 2 hours to bounce: every 5 minutes, all markets (7 credits/call)
- Game day, 60 min to bounce: every 2 minutes, all markets (7 credits/call)

### Tier 2 – Sharp Closing Reference: Betfair Exchange Streaming

Stream Betfair Exchange MATCH_ODDS for all AFL matches using `betfairlightweight` StreamAPI. This is free (existing Betfair account required), push-based, and provides the canonical closing line reference.

What to stream:
- Event type: `61420` (Australian Rules)
- Market types: `MATCH_ODDS`, `HANDICAP`
- Fields: `EX_ALL_OFFERS`, `EX_MARKET_DEF`
- Start stream: 24 hours before bounce; record every state change

Write each streamed price change to `price_snapshots` with `source = 'betfair_stream'`. When the market status transitions to `SUSPENDED`, write the last price to `closing_prices` with `source = 'betfair_stream'`.

### Tier 3 – Historical Backtest Seed: AusSportsBetting + Betfair CSV

**AusSportsBetting:** Download the historical AFL Excel file from [aussportsbetting.com/data/historical-afl-results-and-odds-data/](https://www.aussportsbetting.com/data/historical-afl-results-and-odds-data/) manually. Convert to CSV. Load into `events` and `closing_prices` tables for H2H historical data back to 2010.

**Betfair AU CSV:** Download free AFL Match Odds CSVs from [betfair-datascientists.github.io/data/dataListing/](https://betfair-datascientists.github.io/data/dataListing/) for 2021–2026. These provide timestamped exchange price series useful for modelling pre-game market movement.

**The Odds API historical:** Use the `/historical/sports/aussierules_afl/odds` endpoint with 5-minute snapshot navigation for player prop history from May 2023. Query on-demand for specific past games during model development. Cost: 10 credits per market per snapshot; batch efficiently.

### Do Not Build (for MVP)

- DIY Sportsbet scraper – ToS risk, Cloudflare complexity, fragile
- OddsPortal scraper – higher scraping complexity than value delivered given AusSportsBetting and The Odds API historical both cover the same ground more cleanly
- PuntersEdge integration – evaluate after confirming AFL prop coverage; a viable swap for The Odds API if prop coverage is equivalent and AU-native pricing proves more accurate

### Architecture Summary

```
APScheduler (FastAPI/Railway)
  ├── The Odds API poller (15 min → 2 min cadence pre-game)
  │     └── writes → price_snapshots (PostgreSQL/pgvector)
  │
  └── Betfair Exchange stream (betfairlightweight)
        ├── writes → price_snapshots (continuous)
        └── on SUSPENDED → closing_prices

One-off import scripts
  ├── aussportsbetting_import.py   (historical H2H seed)
  └── betfair_csv_import.py        (historical exchange prices 2021–2026)
```

PostgreSQL is the right store (already in Jacob's stack, pgvector available for potential embedding work later). TimescaleDB is optional – `price_snapshots` will grow large over a full season but standard PostgreSQL with the indexes above handles it fine for a personal project.

---

## References

- [The Odds API – AFL Odds](https://the-odds-api.com/sports/afl-odds.html)
- [The Odds API – Betting Markets](https://the-odds-api.com/sports-odds-data/betting-markets.html)
- [The Odds API – V4 Documentation](https://the-odds-api.com/liveapi/guides/v4/)
- [The Odds API – Historical Odds](https://the-odds-api.com/historical-odds-data/)
- [PuntersEdge API](https://puntersedge.online/api-platform)
- [odds-api.io – Sportsbet Coverage](https://odds-api.io/sportsbooks/sportsbetcomau)
- [Betfair AU – AFL Predictions Model](https://www.betfair.com.au/hub/sports/afl/afl-predictions-model/)
- [betcode-org/betfair – betfairlightweight](https://github.com/betcode-org/betfair)
- [Betfair Automation Hub – Python Tutorial](https://betfair-datascientists.github.io/api/apiPythontutorial/)
- [Betfair Automation Hub – AFL Player Disposal Markets](https://betfair-datascientists.github.io/modelling/AFLPlayerDisposalsFlumine/)
- [Betfair Automation Hub – CSV Data Listing](https://betfair-datascientists.github.io/data/dataListing/)
- [Betfair – Exchange API Rate Limits](https://support.developer.betfair.com/hc/en-us/articles/115003864671-What-data-request-limits-exist-on-the-Exchange-API)
- [declanwalpole/sportsbook-odds-scraper](https://github.com/declanwalpole/sportsbook-odds-scraper)
- [AusSportsBetting – Historical AFL Data](https://www.aussportsbetting.com/data/historical-afl-results-and-odds-data/)
- [Betfair AU – AwesomeBetfair Resource List](https://github.com/betfair-down-under/AwesomeBetfair)
- [Closing Line Value – OddsJam Guide](https://oddsjam.com/betting-education/closing-line-value)
- [Sportsbet Help Centre – Feed Terms and Conditions](https://helpcentre.sportsbet.com.au/hc/en-us/articles/20973619526925-Sportsbet-Feed-Terms-Conditions)
- [How to Bypass Cloudflare (2026) – Scrapfly](https://scrapfly.io/blog/posts/how-to-bypass-cloudflare-anti-scraping)
- [Apify – Sportsbet Scraper](https://apify.com/lexis-solutions/sportsbet-com-au-scraper)
