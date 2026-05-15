# AFL Data Sources Research

**Purpose:** Identify reliable, ideally free/cheap data sources for a Python AFL betting advisor targeting Sportsbet AU head-to-head and player prop markets.

**Scope:** Historical results, player stats, injuries/team selections, weather, fixtures, venue metadata, and historical odds for backtesting.

---

## 1. Match Results and Team Stats

### AFL Tables (afltables.com)

- **URL:** [https://afltables.com/afl/afl_index.html](https://afltables.com/afl/afl_index.html)
- **Access method:** Scraping (no official API)
- **Cost:** Free
- **Historical depth:** Match scores from **1897** onward; player stats complete from 1965, all modern categories from 2011
- **Update frequency:** Updated after each round, typically within 24 hours
- **ToS concerns:** No published terms of service or scraping policy found. The site is community-run with no explicit prohibition, but rate-limit your requests and cache aggressively to avoid hammering the server.
- **Data schema:** Match results (date, teams, scores, venue, crowd), player game logs (disposals, marks, tackles, goals, behinds, kicks, handballs, hit-outs, rebound 50s, inside 50s, clearances, contested possessions, SuperCoach/Fantasy scores), team season stats, ladder history, coaching records
- **Python access:** [pyAFL](https://github.com/RamParameswaran/pyAFL) (`pip install pyAFL`) wraps scraping into Player/Team/Season objects returning Pandas DataFrames. [AflTablesScraper](https://github.com/multimeric/AflTablesScraper) is an alternative. For R, [fitzRoy](https://jimmyday12.github.io/fitzRoy/) provides `fetch_player_stats(source="AFL_tables")`.
- **Data quality:** Unofficial; known minor errors exist but common ones are corrected. Not the official Champion Data feed.
- **Recommended use:** Primary source for all historical backtesting going back more than a decade. Best single historical source by depth.
- **Reliability score:** 4/5

### Squiggle API

- **URL:** [https://api.squiggle.com.au/](https://api.squiggle.com.au/)
- **Access method:** REST API (JSON/XML/CSV), no authentication required
- **Cost:** Free
- **Historical depth:** Games data back to at least 2012; tips/prediction data from when participating models launched (varies per model)
- **Update frequency:** Real-time during games (Event API); Standard API polled on demand
- **ToS concerns:** Must set a descriptive `User-Agent` header including contact email. Must cache data and not make bulk simultaneous requests. Cannot build a site that proxies the API to end users. These are reasonable constraints for a personal advisor tool.
- **Endpoints:**
  - `?q=games;year=YYYY` -- all games for a season including scores, venue, dates, team IDs
  - `?q=tips;year=YYYY` -- model predictions from 50+ community models
  - `?q=standings;year=YYYY;round=N` -- ladder at a given point
  - `?q=power;year=YYYY` -- power rankings from models
  - `?q=sources` -- list of participating prediction models
- **Data schema:** Games (game ID, team names, scores, venue, round, year, complete flag, timestamps); Tips (source model, predicted winner, margin, probability, actual winner)
- **Recommended use:** Fixture data, live scores, aggregated model tips as a feature signal. Squiggle tips are a useful ensemble "market consensus" proxy.
- **Reliability score:** 5/5

### FootyWire

- **URL:** [https://www.footywire.com/](https://www.footywire.com/)
- **Access method:** Scraping (no public API)
- **Cost:** Free
- **Historical depth:** Player stats from **2012** onward; team selections current season
- **Update frequency:** Updated within hours of match completion
- **ToS concerns:** No explicit scraping permission. JavaScript-rendered sections may require Playwright/Selenium. The AFL analytics community has scraped this site for years without documented legal action, but this is not a permission.
- **Data schema:** Advanced player stats per game (time on ground %, metres gained, intercepts, pressure acts, score involvements), team selections with positional layout, injury list, SuperCoach/DreamTeam scores, player profiles (height, weight, age, career games)
- **Python access:** fitzRoy R package wraps it; for Python, use [DataScrapeAFL](https://github.com/AaronLiftig/DataScrapeAFL) as reference, or build custom `requests` + BeautifulSoup scraper
- **Data quality:** Generally high; mirrors official AFL stats with some additional derived metrics
- **Recommended use:** Advanced player metrics not available in AFL Tables; team selection scraping pre-round
- **Reliability score:** 4/5

### AFL.com.au (Official)

- **URL:** [https://www.afl.com.au/](https://www.afl.com.au/)
- **Access method:** Scraping (site backed by Champion Data API, not publicly accessible)
- **Cost:** Free to scrape; official API requires commercial contract
- **Historical depth:** Current season and recent seasons; deep historical not exposed on the public site
- **Update frequency:** Near-real-time during games; selections published at official announcement times
- **ToS concerns:** AFL.com.au's terms of service prohibit automated scraping. The official stats backend is provided by [Champion Data](https://docs.api.afl.championdata.com/), which is a paid commercial API (pricing not public -- requires direct contact). Scraping AFL.com.au is legally risky for a commercial application; acceptable risk for personal/research use.
- **Recommended use:** Team lineups, injury list, official match results for current season cross-reference. Not the primary historical source.
- **Reliability score:** 3/5 (as a scraping target -- reliable data, uncertain legal standing)

### Kaggle / GitHub Datasets

- **Primary dataset:** [AFL Data Analysis by akareen](https://github.com/akareen/AFL-Data-Analysis) -- 5,700+ players, 682,000 rows of performance data, 15,000+ match records, 1897-2025, CSV format. Includes AusSportsBetting odds data (2009-2024).
- **Secondary:** [Kaggle AFL Database](https://www.kaggle.com/datasets/stoney71/aflstats) -- key stats game-by-game, player-by-player
- **Access method:** Direct download
- **Cost:** Free
- **Update frequency:** Irregular (contributor-driven)
- **Recommended use:** Bootstrap historical training data quickly. Validate against live scraped sources before relying on it.
- **Reliability score:** 3/5 (depends on contributor's update cadence)

---

## 2. Player-Level Statistics

### What is available historically

| Stat category | AFL Tables | FootyWire | Notes |
|---|---|---|---|
| Kicks, handballs, disposals | 1965+ | 2012+ | Core stats |
| Marks, tackles | 1965+ | 2012+ | Core stats |
| Goals, behinds | 1897+ | 2012+ | |
| Hit-outs | 1965+ | 2012+ | |
| Inside 50s, rebound 50s | 1999+ | 2012+ | |
| Clearances | 1999+ | 2012+ | |
| Contested possessions | 1999+ | 2012+ | |
| Time on ground % | Not available | 2012+ | FootyWire only |
| Metres gained | Not available | 2018+ | FootyWire only |
| Pressure acts | Not available | 2018+ | FootyWire only |
| SuperCoach score | 2012+ | 2012+ | Useful composite proxy |
| AFL Fantasy score | 2012+ | 2012+ | Slightly different weighting |

### Current-season-only data

Player prop lines (disposals over/under, goal scorer markets) are only available from bookmakers in real time -- there is no historical archive of Sportsbet's own prop lines through a free source. The Odds API covers player props from **May 2023** onward at 5-minute intervals (paid). DFS Australia ([https://dfsaustralia.com/afl-player-lines/](https://dfsaustralia.com/afl-player-lines/)) shows current-round lines but has no historical archive.

### SuperCoach/Fantasy scores as a composite signal

SuperCoach scores weight: kicks (3), handballs (1.5), marks (3), tackles (4), goals (8), behinds (0.5), hit-outs (1), frees for (1), frees against (-3). This composite is useful for a single-number player performance proxy. Available via FootyWire scraping and AFL Fantasy ([https://fantasy.afl.com.au/](https://fantasy.afl.com.au/)) -- the latter requires login.

---

## 3. Injuries and Team Selections

### AFL.com.au Injury List

- **URL:** [https://www.afl.com.au/matches/injury-list](https://www.afl.com.au/matches/injury-list)
- **Access method:** Scraping
- **Cost:** Free
- **Update frequency:** Updated when clubs submit medical updates, typically mid-week and after each match
- **Data:** Player name, injury type, expected return timeline (in weeks or "Season")
- **ToS concerns:** Same AFL.com.au scraping concerns as above
- **Reliability score:** 4/5

### FootyWire Injury List

- **URL:** [https://www.footywire.com/afl/footy/injury_list](https://www.footywire.com/afl/footy/injury_list)
- **Access method:** Scraping (clean HTML tables)
- **Cost:** Free
- **Data:** Player, injury type, return timeline -- 127+ injured players across all 18 teams mid-season
- **Update frequency:** Several times weekly during season
- **Recommended use:** More scrapeable than AFL.com.au; use as primary injury source
- **Reliability score:** 4/5

### FootyWire Team Selections

- **URL:** [https://www.footywire.com/afl/footy/afl_team_selections](https://www.footywire.com/afl/footy/afl_team_selections)
- **Access method:** Scraping (semantic HTML, player links with ID slugs)
- **Cost:** Free
- **Data:** Full positional lineup, interchange, emergencies, ins/outs for each match
- **Update frequency:** Updated at official announcement times (see schedule below)
- **Reliability score:** 4/5

### AFL Team Selection Announcement Schedule

Official timing for 2026 season:

| Match day | Extended squad (18+8) | Final 22 announcement |
|---|---|---|
| Thursday | N/A | Wednesday 6:20 PM AEST |
| Friday | N/A | Thursday 6:20 PM AEST |
| Saturday | N/A | Thursday 6:20 PM AEST |
| Sunday | Thursday 6:20 PM AEST | Friday 5:00 PM AEST |
| Monday | Thursday 6:20 PM AEST | Friday 5:00 PM AEST |

**Late substitute rule:** The emergency substitute is confirmed by each club **60 minutes before bounce**. This is the "late out" problem -- a player scratched at 60-minute-to-bounce is not in the official announcement data. The only way to catch this is:
1. Monitor club social media (Twitter/X, club website news feed) in the 60-minute window
2. Monitor AFL.com.au team lineups page for updates ([https://www.afl.com.au/matches/team-lineups](https://www.afl.com.au/matches/team-lineups))
3. Monitor SEN.com.au "ins and outs" articles ([https://www.sen.com.au/news/afl](https://www.sen.com.au/news/afl)) which often break late changes first

Sportsbet handles this by: voiding bets on players named as substitute. Bets on non-starters are voided on most markets. The 60-minute window is the critical risk period.

### Rotowire AFL Injury Report

- **URL:** [https://www.rotowire.com/afl/injury-report.php](https://www.rotowire.com/afl/injury-report.php)
- **Access method:** Scraping (BeautifulSoup-compatible HTML)
- **Cost:** Free
- **Data:** Player injury status by team, with GTD (game-time decision) / Out / Probable labels
- **Update frequency:** Updated multiple times per week during season
- **ToS concerns:** No explicit API; the Rotowire API client on PyPI exists but terms are unclear. Scraping with low frequency is standard practice in the analytics community.
- **Recommended use:** Cross-reference with FootyWire injury list; Rotowire uses slightly different status labels useful for GTD flagging
- **Reliability score:** 3/5

### Zero Hanger

- **URL:** [https://www.zerohanger.com/afl/injuries-suspensions/](https://www.zerohanger.com/afl/injuries-suspensions/)
- **Access method:** Scraping
- **Cost:** Free
- **Data:** Injuries, suspensions (MRO/Tribunal outcomes), round-by-round team line-ups
- **Update frequency:** Near-daily during season
- **Recommended use:** Suspensions tracking (MRO charges that change final 22 one or two days before game)
- **Reliability score:** 3/5

### SEN (Sports Entertainment Network)

- **URL:** [https://www.sen.com.au/news/afl](https://www.sen.com.au/news/afl)
- **Access method:** Scraping news articles; no structured API
- **Cost:** Free
- **Recommended use:** Late mail source. SEN publishes "ins and outs" articles at announcement time and often breaks emergency substitution news before official channels. Parse article headlines and body text for player names + "in/out/sub/emergency" language.
- **Reliability score:** 3/5 (unstructured text, NLP parsing required)

---

## 4. Weather Data

### Open-Meteo Historical Weather API

- **URL:** [https://open-meteo.com/en/docs/historical-weather-api](https://open-meteo.com/en/docs/historical-weather-api)
- **Access method:** REST API, no key required for non-commercial use
- **Cost:** Free for non-commercial; commercial use requires paid plan
- **Historical depth:** ERA5 reanalysis data from **1940** onward; ERA5-Land from 1950. This enables weather backfill for any historical match.
- **Update frequency:** Historical archive up to 5 days before present; not suitable for real-time
- **Rate limits:** Not explicitly stated; the service is designed for batch queries
- **Key variables:** Temperature (2m), precipitation, wind speed/direction (10m and 100m), relative humidity, cloud cover, UV index, apparent temperature
- **URL format:**
  ```
  https://archive-api.open-meteo.com/v1/archive
    ?latitude=-37.81&longitude=144.96
    &start_date=2023-04-01&end_date=2023-04-01
    &hourly=temperature_2m,precipitation,windspeed_10m,windgusts_10m,weathercode
  ```
- **Recommended use:** Primary weather source for backtesting. Query match kick-off time ± 2 hours using venue coordinates.
- **Reliability score:** 5/5

### BOM (Bureau of Meteorology) via Open-Meteo

- **URL:** [https://open-meteo.com/en/docs/bom-api](https://open-meteo.com/en/docs/bom-api)
- **Access method:** REST API via Open-Meteo wrapper
- **Cost:** Free (non-commercial)
- **Note:** BOM is currently upgrading its platform -- open data delivery is "temporarily suspended" as of May 2026. Use the ERA5 endpoint instead until BOM data resumes.
- **Recommended use:** 10-day forecasts when available; fall back to Open-Meteo global models
- **Reliability score:** 2/5 (currently unreliable due to BOM platform migration)

### OpenWeatherMap

- **URL:** [https://openweathermap.org/api/history](https://openweathermap.org/api/history)
- **Access method:** REST API, key required
- **Cost:** Historical data requires paid plan; free tier covers current + 5-day forecast only
- **Historical depth:** From 1979 onward (hourly)
- **Recommended use:** Alternative to Open-Meteo if needed; not required if Open-Meteo is sufficient
- **Reliability score:** 4/5

### Venue Weather Coordinates

| Venue | Lat | Long | Indoor/Outdoor | Notes |
|---|---|---|---|---|
| MCG, Melbourne | -37.820 | 144.984 | Outdoor | |
| Marvel Stadium (Docklands) | -37.817 | 144.947 | **Indoor (roof closed 2019-2025; partially open 2026)** | Weather irrelevant when roof closed; note 2026 policy change for select matches |
| Adelaide Oval | -34.915 | 138.596 | Outdoor | |
| Perth Stadium (Optus) | -31.951 | 115.889 | Outdoor | |
| The Gabba, Brisbane | -27.486 | 153.038 | Outdoor | |
| SCG, Sydney | -33.891 | 151.225 | Outdoor | |
| GMHBA Stadium, Geelong | -38.157 | 144.355 | Outdoor | |
| People First Stadium, Gold Coast | -28.005 | 153.426 | Outdoor | |
| York Park, Launceston | -41.440 | 147.125 | Outdoor | |
| Manuka Oval, Canberra | -35.322 | 149.131 | Outdoor | |

**Marvel Stadium roof flag is critical:** Games played with roof closed have weather completely removed as a factor. Model should zero-out weather features for roof-closed matches. For 2026, check each fixture -- the AFL are selectively opening the roof for clear night/twilight games.

---

## 5. Schedule and Fixture Data

### Squiggle API

- **Endpoint:** `https://api.squiggle.com.au/?q=games;year=2026`
- **Fields:** Game ID, round, year, date/time (UTC), home team, away team, venue, is_final flag, scores when completed
- **Cost:** Free
- **Update frequency:** Live
- **Recommended use:** Primary fixture source -- clean, programmatic, well-structured
- **Reliability score:** 5/5

### FixtureDownload.com

- **URL:** [https://fixturedownload.com/results/afl-2026](https://fixturedownload.com/results/afl-2026)
- **Access method:** Direct download (CSV, XLSX, ICS, JSON) -- no scraping needed
- **Cost:** Free
- **Data:** Date, round number, home team, away team, venue, kick-off time (local), result when completed
- **Historical depth:** Available for 2018 onward
- **Recommended use:** Backup fixture source; useful for importing into calendar for scheduling checks
- **Reliability score:** 4/5

---

## 6. Ground and Venue Metadata

The best approach is a **manually curated static reference file** since this data changes slowly (new stadium names, occasional venue additions). There is no clean, machine-readable API for venue metadata.

**Sources to compile from:**

- [Wikipedia: List of AFL grounds](https://en.wikipedia.org/wiki/List_of_Australian_Football_League_grounds) -- capacity, home teams, city, surface, opened date
- [Austadiums.com](https://www.austadiums.com/stadiums/marvel-stadium) -- roof type, dimensions, surface condition notes
- [AFL.com.au venues](https://www.afl.com.au/venues/3) -- official venue pages per ground

**Recommended schema for the static file:**

```python
{
  "venue_name": "Marvel Stadium",
  "aliases": ["Docklands Stadium", "Etihad Stadium"],
  "city": "Melbourne",
  "state": "VIC",
  "latitude": -37.817,
  "longitude": 144.947,
  "capacity": 56347,
  "surface": "grass",
  "indoor": True,
  "roof_policy_note": "Roof closed all games 2019-2025; selectively open 2026",
  "home_teams": ["Essendon", "North Melbourne", "Western Bulldogs", "Carlton", "St Kilda"],
  "dimensions_m": None  # varies by configuration
}
```

This is a one-time effort -- approximately 15 venues to document. The akareen GitHub repo or AFL Tables venue data can seed this.

---

## 7. Historical Odds for Backtesting

This is the hardest piece. Without pre-existing historical odds, any backtest against model predictions vs. "would you have beaten the market" is not honest.

### AusSportsBetting.com

- **URL:** [https://www.aussportsbetting.com/data/historical-afl-results-and-odds-data/](https://www.aussportsbetting.com/data/historical-afl-results-and-odds-data/)
- **Access method:** Direct Excel file download
- **Cost:** Free
- **Format:** XLSX
- **Historical depth:** **2009 onward** (best free historical odds source)
- **Data fields:** Date, kick-off time (local), home team, away team, venue, home score, away score, playoff flag, plus **bookmaker head-to-head odds** (closing odds from multiple bookmakers including Sportsbet, TAB, William Hill, Betway, Unibet, Bet365, Neds, Pinnacle)
- **Update frequency:** Updated each round; the akareen GitHub repo also bundles this data through 2024
- **ToS:** Personal use only; do not republish. Data may contain errors. Do not rely on for wagering decisions (ironic given the use case, but it's a standard disclaimer).
- **Coverage gap:** Odds are **head-to-head (match winner) only**. No line (handicap), no totals, no player props.
- **Recommended use:** Primary historical odds source for match winner model backtesting. Single most important free dataset for honest backtesting.
- **Reliability score:** 4/5

### Betfair Historical Data (Free CSVs via Automation Hub)

- **URL:** [https://betfair-datascientists.github.io/data/dataListing/](https://betfair-datascientists.github.io/data/dataListing/)
- **Access method:** Direct CSV download, free, no login required
- **Cost:** Free (basic tier: 1-minute interval data, last traded price, no volume)
- **Format:** CSV
- **Historical depth:** **2021-2026** (AFL specific; Betfair exchange data from 2016 on the paid portal)
- **Data types available:** "All Markets" files and "Match Odds" files per year
- **Data fields (basic free):** Market ID, selection, last traded price at 1-minute intervals, winning selection flag. No volume, no full ladder.
- **ToS:** Betfair AU/NZ data; free for personal/research use. Pro-level (50ms tick data, full ladder) requires contacting automation@betfair.com.au.
- **Recommended use:** Secondary historical odds source. Betfair exchange odds are sharper than bookmaker prices and useful for calibrating model probability vs. market implied probability. Covers 2021+.
- **Reliability score:** 4/5

### The Odds API (Historical)

- **URL:** [https://the-odds-api.com/historical-odds-data/](https://the-odds-api.com/historical-odds-data/)
- **Access method:** REST API with API key, paid subscription required for historical access
- **Cost:** Paid (tier pricing not published; requires signup)
- **Historical depth:** AFL from **June 2020** onward, at 10-minute snapshots (until Sep 2022) then 5-minute snapshots
- **Australian bookmakers included:** Yes (AU region)
- **Markets:** Head-to-head (from 2020); player props from **May 2023** onward
- **Recommended use:** If budget allows, this is the only source for historical player prop lines (disposals, goals, marks over/under). Critical if building player prop models. Cannot be replicated from free sources.
- **Reliability score:** 4/5 (paid, well-maintained API)

### OddsPortal

- **URL:** [https://www.oddsportal.com/aussie-rules/australia/afl/results/](https://www.oddsportal.com/aussie-rules/australia/afl/results/)
- **Access method:** Scraping (JavaScript-heavy, requires Playwright)
- **Cost:** Free (scraping effort required)
- **Historical depth:** **2009-2026** shown on results pages
- **Data:** Closing odds from multiple bookmakers (typically 15-30 bookmakers per match), head-to-head
- **ToS concerns:** No official scraping permission. OddsPortal's ToS prohibit automated access. Multiple open-source scrapers exist ([OddsHarvester](https://github.com/jordantete/OddsHarvester), [odds-portal-scraper](https://github.com/Mg30/odds-portal-scraper)) but these operate in a legal grey area. Enforcement history is unclear but the ToS prohibition is explicit.
- **Recommended use:** Cross-reference and validation against AusSportsBetting data. Do not use as primary source given ToS risk.
- **Reliability score:** 2/5 (data quality fine, ToS risk significant)

### Betfair Historical Data Portal (Paid)

- **URL:** [https://historicdata.betfair.com.au/](https://historicdata.betfair.com.au/)
- **Access method:** Betfair account login; files purchased per dataset
- **Cost:** Varies by tier (Basic free at 1-minute; Advanced and Pro paid, with full ladder and 1-second/50ms resolution)
- **Historical depth:** 2016 onward (all markets since APING launch)
- **Format:** TAR + bz2 compressed files; convert to CSV via Betfair Data Processor tool
- **Recommended use:** If you need pre-2021 Betfair data or need volume/ladder data, purchase directly. For MVP, the free CSVs from the Automation Hub (2021+) are sufficient.
- **Reliability score:** 5/5

---

## Summary Tables

### Source Quick Reference

| Source | Type | Cost | Historical depth | Player-level | Odds | Scraping risk |
|---|---|---|---|---|---|---|
| AFL Tables | Scrape | Free | 1897+ | Yes (1965+) | No | Low |
| Squiggle API | API | Free | 2012+ | No | No (model tips only) | None |
| FootyWire | Scrape | Free | 2012+ | Yes (advanced) | No | Medium |
| AusSportsBetting | Download | Free | 2009+ | No | H2H only | None |
| Betfair AU CSVs | Download | Free | 2021+ | No | Exchange H2H | None |
| The Odds API | API | Paid | 2020+ (AFL) | Props 2023+ | Yes (multi-book) | None |
| Open-Meteo | API | Free | 1940+ | N/A | N/A | None |
| Squiggle fixtures | API | Free | Current + history | No | No | None |
| FixtureDownload | Download | Free | 2018+ | No | No | None |
| OddsPortal | Scrape | Free | 2009+ | No | H2H | High |
| Champion Data | API | Paid (commercial) | Full | Yes (official) | No | None |
| Rotowire AFL | Scrape | Free | Current only | Injuries | No | Low |
| Zero Hanger | Scrape | Free | Current | Suspensions | No | Low |
| SEN.com.au | Scrape | Free | Current | Late mail | No | Low |

---

## 8. Recommended Data Stack for MVP

### Minimum viable set

**Phase 1: Historical backtesting foundation**

1. **AFL Tables via pyAFL** -- match results and player stats from 1897/1965+. This is the backbone of any model. `pip install pyAFL` or scrape directly. Cache everything locally in SQLite or Parquet files. Do not re-scrape the entire history each run.

2. **AusSportsBetting XLSX** -- download once, save locally. This gives you head-to-head closing odds from 2009 to present, which is essential for any honest backtest. Combined with AFL Tables results, you can immediately answer "did the model beat the closing line?"

3. **Betfair AU free CSVs (2021-2026)** -- download all years from the Automation Hub. Betfair exchange odds are sharper than bookmaker prices; this gives you a sharper benchmark for the last 5 seasons.

**Phase 2: Current season operations**

4. **Squiggle API** -- fixture data, results, and model tip ensemble. Zero setup cost, clean API. Use as primary fixture source and for `is_final` detection.

5. **FootyWire injury list + team selections scraper** -- build a simple `requests` + BeautifulSoup scraper targeting `/injury_list` and `/afl_team_selections`. Run it at official announcement times (Wed/Thu 6:20 PM, Fri 5:00 PM). This is the pre-match availability signal.

6. **Open-Meteo Historical API** -- historical weather backfill for all past games using ERA5. For current-week games, use the forecast endpoint with venue coordinates. Free, no key required, covers 1940+.

7. **Static venue metadata file** -- 15 venues, manually curated once from Wikipedia + Austadiums. Include `indoor` flag and `roof_policy` for Marvel Stadium.

**What to add later (not MVP)**

- **The Odds API (paid)** -- needed if building player prop models. Player prop lines are not available historically from any free source except what you start scraping now. The earlier you subscribe and start archiving, the better your prop backtesting data.
- **Champion Data API (commercial)** -- only relevant if this becomes a commercial product with real-money staking at scale. Pricing requires direct negotiation with Champion Data.
- **SEN/Zero Hanger late mail monitor** -- useful for production deployment; a cron job that watches for late change articles in the 60-minute pre-bounce window.

### Honest backtesting caveats

- AusSportsBetting provides **closing** bookmaker odds, not opening or line movement history. Closing line value (CLV) tests are valid but movement analysis requires The Odds API (2020+) or Betfair tick data.
- Player prop backtesting before May 2023 is not possible from any free source. AusSportsBetting does not cover props.
- Weather data via Open-Meteo is accurate for outdoor venues. Marvel Stadium games before the 2026 roof-opening policy change should have weather features zeroed out.
- AFL Tables data has minor errors; cross-reference with FootyWire for any statistical outliers before trusting as training labels.
- The akareen GitHub repo bundles AFL Tables + AusSportsBetting data into clean CSVs -- useful for bootstrap, but validate it is current before relying on it.

---

## Sources Cited

- [AFL Tables](https://afltables.com/afl/afl_index.html)
- [Squiggle API documentation](https://squiggle.com.au/the-squiggle-api/)
- [pyAFL Python library](https://github.com/RamParameswaran/pyAFL)
- [fitzRoy R package](https://jimmyday12.github.io/fitzRoy/)
- [AflTablesScraper](https://github.com/multimeric/AflTablesScraper)
- [DataScrapeAFL (FootyWire + AFL Tables)](https://github.com/AaronLiftig/DataScrapeAFL)
- [FootyWire injury list](https://www.footywire.com/afl/footy/injury_list)
- [FootyWire team selections](https://www.footywire.com/afl/footy/afl_team_selections)
- [AusSportsBetting AFL odds data](https://www.aussportsbetting.com/data/historical-afl-results-and-odds-data/)
- [Betfair Automation Hub -- AFL CSV files](https://betfair-datascientists.github.io/data/dataListing/)
- [Betfair Historical Data Portal](https://historicdata.betfair.com.au/)
- [The Odds API -- historical AFL](https://the-odds-api.com/historical-odds-data/)
- [OddsPortal AFL results](https://www.oddsportal.com/aussie-rules/australia/afl/results/)
- [AFL Data Analysis GitHub (akareen)](https://github.com/akareen/AFL-Data-Analysis)
- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- [Open-Meteo BOM model API](https://open-meteo.com/en/docs/bom-api)
- [AFL Official Injury List](https://www.afl.com.au/matches/injury-list)
- [AFL Team Lineups](https://www.afl.com.au/matches/team-lineups)
- [Rotowire AFL Injury Report](https://www.rotowire.com/afl/injury-report.php)
- [Zero Hanger injuries and suspensions](https://www.zerohanger.com/afl/injuries-suspensions/)
- [SEN AFL news](https://www.sen.com.au/news/afl)
- [DreamTeamTalk -- when are AFL teams announced?](https://dreamteamtalk.com/2023/03/14/when-are-afl-teams-announced/)
- [AFL Thursday night team announcement return](https://www.afl.com.au/news/612186/thursday-night-team-announcements-to-return)
- [FixtureDownload AFL 2026](https://fixturedownload.com/results/afl-2026)
- [Wikipedia: List of AFL grounds](https://en.wikipedia.org/wiki/List_of_Australian_Football_League_grounds)
- [Marvel Stadium roof -- Austadiums](https://www.austadiums.com/news/1687/afl-to-reopen-marvel-stadium-roof-for-select-2026-matches)
- [Champion Data AFL Data Platform](https://docs.api.afl.championdata.com/)
- [Sportsbet -- what happens when player does not start](https://helpcentre.sportsbet.com.au/hc/en-us/articles/18430823452557-What-Happens-To-My-AFL-Bet-If-My-Player-Does-Not-Start-Or-Is-Named-The-Sub)
- [DFS Australia AFL player lines](https://dfsaustralia.com/afl-player-lines/)
- [Kaggle AFL Database](https://www.kaggle.com/datasets/stoney71/aflstats)
