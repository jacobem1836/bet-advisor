# AFL Predictive Modelling for Betting: Research Findings

> Research compiled May 2026 for the bet-advisor project.
> Goal: determine whether an amateur Python model can realistically produce positive CLV on Sportsbet AFL markets.

---

## 1. Existing Open-Source AFL Models

### Squiggle (squiggle.com.au)

Squiggle is the canonical public aggregator for AFL prediction models. It hosts ~30+ models and publishes an API and leaderboard. Think of it as the AFL equivalent of FiveThirtyEight's aggregation layer.

**What it predicts:** Win probability (head-to-head), projected margin, projected ladder. No line or totals markets directly -- the API returns probabilities and margin estimates that you can convert yourself.

**API:** Public, free, no auth. Endpoints: `games`, `tips`, `standings`, `sources`, `ladder`, `power`. Returns JSON/XML/CSV. The `tips` endpoint gives per-game predictions from all registered models, making it trivially easy to pull market comparisons.

**Leaderboard benchmark (2025 full season):** [Squiggle 2025 leaderboard](https://live.squiggle.com.au/2025.html)
- Top model (Don't Blame the Data): 163/216 tips correct, 75.5% accuracy
- Matter of Stats: 161 tips, 74.5%
- Glicko Ratings: 161 tips, 74.5%
- **Punters** (bookmaker aggregate): tracked in the leaderboard as a baseline

**Leaderboard benchmark (2026, Round 10):** [Squiggle leaderboard](https://squiggle.com.au/leaderboard/)
- Wheelo Ratings: 68/82 tips, 82.9% accuracy
- Drop Kick Data: 67 tips, 81.7%
- Punters (book aggregate): 64 tips, **78.0%** -- sitting 6th

**Critical nuance:** Tips accuracy (binary correct/incorrect) is a poor discrimination metric in a sport where 70-75% of games are won by the favourite anyway. The "Bits" metric (log-score rewarding confidence) is the more honest measure of model quality. On Bits, the market aggregate (Punters) frequently outscores or ties with the top models even when it ranks lower on raw tip count.

**Source availability:** Squiggle itself is not open source, but its API is free. Individual model authors publish methodology to varying degrees.

---

### Matter of Stats (MoSHBODS)

**URL:** [matterofstats.com](http://www.matterofstats.com/)

**What it predicts:** Win probability, projected margin. Registered on Squiggle.

**Modelling approach:** MoSHBODS (MatterOfStats Scoring Shot Based Offence Defence System) -- a dual-rating system that separately tracks each team's offensive and defensive strength. Key innovations:
- Rates teams on **scoring shots** (not raw score) to reduce goal-kicking luck noise, then blends with raw score via a tuned mixing parameter
- **Venue Performance Value (VPV):** per-team, per-venue excess margin over model expectation across a rolling time window -- captures ground-specific team performance
- Era-normalisation via rolling z-scoring (allows comparisons across 130 years of VFL/AFL history)
- Decaying learning rate within season (more weight on recent games as season progresses)

**Reported accuracy:** Competitive with market. In a 2018-2020 analysis, MoSHBODS was the only public model that outperformed betting odds on both reliability and resolution components of the Brier score decomposition. However, it finished 22nd on Squiggle's tips leaderboard in 2024 -- showing that Brier score calibration and tip count are measuring different things.

**Source code:** Not publicly available. Methodology extensively documented on the blog.

---

### FiguringFooty

**URL:** [figuringfooty.com](https://www.figuringfooty.com/)

**What it predicts:** Win probability, margin. Registered on Squiggle.

**Modelling approach:** Figuring Footy Scoring Shots (FFSS) rating -- an Elo variant that uses **scoring shots as the primary signal** rather than final score. The key insight: a team that won 80-70 by kicking 8/12 from shots should be rated differently to one that kicked 8/8. Kicking inaccuracy is significantly noisy and partly regresses toward mean in subsequent games.

Three inputs to match prediction: home ground advantage (adjusted for travel distance), home team FFSS rating, away team FFSS rating.

**Source code:** Not public. Documented methodology.

---

### The Arc

**URL:** [thearcfooty.com](https://thearcfooty.com/)

**Modelling approach:** Elo variant similar in structure to FiveThirtyEight's NFL model. Standard Elo update rule with symmetric adjustment. Incorporates venue experience differential and travel differential.

**Reported Brier score:** 0.205 in 2018-2020 analysis -- slightly worse than market (0.201) but competitive with top models. Accuracy approximately 67.8%.

**Source code:** Not published.

---

### HPN Footy

**URL:** [hpnfooty.com](https://www.hpnfooty.com/)

**Modelling approach:** Notably does **not** explicitly model home ground advantage as a separate parameter. Still achieves competitive tip accuracy. This is relevant because it suggests HGA may be partially captured by other rating signals rather than requiring an independent term.

**Source code:** Not public.

---

### FootyMaths Institute (FMI)

**URL:** [footymaths.blogspot.com](https://footymaths.blogspot.com/)

**Modelling approach:** Rating-based system. Adjusts for opposition quality, including pre-season performance. Examines counterfactual scenarios (what if all close games were reversed). Methodology details are limited in public documentation.

**Source code:** Not public.

---

### FitzRoy (R package)

**URL:** [jimmyday12.github.io/fitzRoy](https://jimmyday12.github.io/fitzRoy/)

This is the **gold standard data access layer** for AFL modelling -- but it is an R package, not a model. It provides:
- Match results from 1897 (AFL Tables)
- Player statistics from 2012 (Footywire)
- Fixture data
- Squiggle API wrapper (predictions from all registered models)
- Ladder data

It is the primary data infrastructure used in published academic AFL models. There is no direct Python equivalent of the same maturity.

---

### Betfair Data Scientists AFL Tutorials

**URL:** [betfair-datascientists.github.io/modelling/AFLmodellingPython](https://betfair-datascientists.github.io/modelling/AFLmodellingPython/)

The most complete **Python AFL modelling tutorial** in public. Four-notebook series:
1. Data cleaning (AFL Tables + Footywire via FitzRoy-inspired scraping)
2. Feature creation
3. Model selection and optimisation
4. Weekly prediction pipeline

**Features used:** Elo ratings (k=30), exponentially weighted moving averages (span=10) on all player stats, disposal efficiency, Inside 50/Rebound 50 ratios, 5-game rolling form (margin + win%), current betting odds as a feature.

**Model:** Logistic regression (C=0.01, newton-cg solver, L2 regularisation), grid searched.

**Accuracy achieved:**
- Model: 67.68% correct (2018 test set), log-loss 0.5848
- Betting odds baseline: **71.21%** correct, log-loss 0.5546
- Historical odds accuracy (full dataset): 73.15%

**Honest conclusion from the tutorial itself:** The model underperforms the betting market on all metrics. It is offered as a starting point, not as a profitable system.

**GitHub repo:** [betfair-datascientists/predictive-models/afl](https://github.com/betfair-datascientists/predictive-models/tree/master/afl)

---

## 2. Feature Sets That Matter for AFL Prediction

### Validated as having real predictive signal

**Elo ratings (team strength)** -- The most consistently predictive single feature. Carries the bulk of signal in nearly all published models. The specific k-factor matters less than the basic Elo update mechanism being present.

**Home ground advantage (HGA)** -- Consistently significant in every published study. Research using Bradley-Terry models found AT_HOME coefficients of 0.15-0.62 across individual seasons, always positive and statistically significant. Kardinia Park (Geelong), Adelaide Oval (Adelaide/Port), and UTAS Stadium (Nth Melb at Launceston) show venue-specific effects independent of the generic home designation.

**Interstate travel penalty** -- Robust negative effect (coefficients -0.43 to -0.71 in Bradley-Terry analysis). Playing interstate creates measurably worse outcomes than local away games. The effect is not linear with distance -- log-distance captures it better. Best modelled as the difference in log-distance travelled by each team (both-teams-travel cases are penalised less than one-team-travels cases).

**Venue Performance Value (VPV)** -- The MoSHBODS term for per-team, per-venue historical performance excess. Captures idiosyncratic advantages beyond generic HGA (e.g., a team that genuinely plays better at a particular ground regardless of crowd effect). Validated in MoSHBODS architecture.

**Scoring shots vs raw score** -- Significant: using scoring shots rather than raw score as the Elo update signal reduces noise from kicking accuracy variance. Figuring Footy and MoSHBODS both use this. Supported by the intuition that shot generation is a more stable skill than conversion rate across games.

**Recent form (rolling N games)** -- Has signal, but the optimal window is shorter than most amateurs assume. Exponentially weighted averages with span ~10 games work in the Betfair tutorial. Simple rolling 5-game win percentage adds incremental signal. Season-start form resets matter (teams carry ratings across preseason but need early-season form corrections).

**Previous season ladder position** -- Validated in Bradley-Terry analysis as more predictive than current ladder position for early-season games. Makes sense: current ladder has high variance in rounds 1-5.

**Forward 50 zone metrics** -- Inside 50 entries, marks inside 50, goals from inside 50 chains: all significantly predictive in team-level models. Accessible in AFL Tables data.

### Features with limited or uncertain predictive signal

**Rest days / bye advantage** -- Research on NFL shows no significant advantage from bye week rest when properly controlled for team quality. AFL-specific research is sparse, but the general finding suggests rest differential adds weak signal at best. Worth including as a binary (>7 days vs normal turnaround) but don't over-weight it.

**Weather (rain, wind)** -- AFL Lab research confirms wet weather significantly reduces total scoring (median 178 points dry vs 136 in rain), and damp games vs rainy games show different effects. However, the author explicitly concludes that public weather data is "not good enough" to exploit this reliably for betting -- you'd need accurate pre-game weather forecasts plus a good model of how each team's game style is weather-affected. Worth including in a totals model but not obviously exploitable at H2H market level.

**Head-to-head historical record** -- Likely spurious once you properly control for current team ratings. Historical H2H records incorporate eras when teams were systematically different in quality. Use current Elo instead. H2H at specific venues may have marginal additional signal.

**Late-season tanking** -- Documented phenomenon: teams outside finals contention in rounds 18-22 may show reduced effort, affecting predictability. The Bradley-Terry analysis found Finals Series games are harder to predict than regular season. However, the market prices also adjust for tanking risk. This is a known-unknown that both you and the bookmaker are aware of.

**Finals form** -- Models consistently underperform in finals. This is both because: (a) the variance of closely matched top-8 teams is higher, and (b) small sample sizes mean historical finals form is noisy.

**Weather, injuries, team selection** -- Late team selection changes are the most consistently actionable signal in AFL prop markets. This is discussed further in Section 4.

---

## 3. Can Amateur Models Beat the Closing Line on AFL?

This is the central question. Here is the honest answer, supported by evidence.

### What the data shows

**The betting market is close to efficient for H2H AFL:**

The most comprehensive public analysis ([Performance of AFL Prediction Models, 2018-2020](https://rstudio-pubs-static.s3.amazonaws.com/613310_24778e5c0c78485b9f6a011198e4c51b.html)) found:

- Betting odds Brier score: **0.201** (best of all models tested)
- Best public model (aggregate): 0.202
- Best single model: 0.202
- "No-one has been able to significantly beat the betting market over the last three seasons (Massey Ratings were only more accurate by 1 tip out of 423 games)"
- The market had the **best reliability (0.0157)** and competitive resolution -- meaning it is well-calibrated AND has discrimination power

**The 2025 Squiggle leaderboard shows a more optimistic picture** -- Wheelo Ratings at 82.9% vs Punters at 78.0% -- but this is a partial-season snapshot (Round 10 of 2026, based on the leaderboard accessed), and small sample selection effects are significant. Tips accuracy also does not directly translate to CLV.

**Key insight from MatterOfStats 2024 analysis:** The gaps between models on the leaderboard are often explained by variance (luck) as much as model quality. A model that finishes 22nd in one year can plausibly be top-10 in another. The underlying model quality differences are smaller than the rankings suggest.

### Realistic CLV ceiling for AFL H2H models

For context from other sports: elite sharp bettors in well-developed markets (NFL moneyline, NBA sides) typically achieve +1% to +2% mean CLV over large samples. Props bettors may find +5% CLV in softer markets. AFL sits somewhere between these.

**No public evidence of sustained +2% CLV from an AFL model exists in published form.** The academic literature consistently shows the market is efficient against published models. If such edge existed, it would be arbed away.

**Where the market may be softer:**
- Early-week AFL lines vs closing lines (market moves during the week as sharp money comes in -- opening lines may be softer)
- Line betting (handicap markets) vs H2H: line markets have less liquidity and may be less efficiently set, particularly for large-favourite games
- Totals (over/under total score): this is a relatively thin market in AFL and weather-adjusted totals modelling has not been systematically tested in public literature
- Player props (disposals, goals): significantly less efficient than team markets -- discussed in Section 4

### CLV strategy implication

The approach that maximises chances of positive CLV is:
1. Build a model that outputs probabilities
2. Compare to **opening line** rather than close (better chance of beating it)
3. Track CLV by logging placed-odds vs closing-odds to validate edge over time
4. Accept that a positive-CLV model requires a minimum ~500 bets to distinguish skill from variance

---

## 4. Player-Level Modelling -- Disposals, Goals, SuperCoach

### Market efficiency for player props

Player prop markets (disposals, goals, marks, tackles) on Sportsbet are materially softer than team H2H markets. The reasons:
- Lower liquidity means bookmaker risk exposure is smaller, so they invest less in sharpening the line
- Higher variance per-player (one injury changes everything) makes it harder for the book to move efficiently
- More markets (30+ players per game with 4-5 markets each) than team markets (2 markets per game)
- Betfair's exchange disposals market exists but has thin liquidity, suggesting sharp money is not systematically pressing these lines

**Disposal markets specifically:** Disposal counts are the most consistent statistical measure in AFL. Average disposals per game has a reasonably high autocorrelation for midfielders (role-based stability). This makes it modelable from historical data.

**What drives disposal prediction:**
- Player role (midfielder vs forward vs defender) -- dominant feature
- Matchup: who are they being tagged by? (Tag assignments require pre-game information)
- Team game plan (how much clearance work vs spread work)
- Venue type (MCG games tend toward higher disposals due to ground size)
- Opposition pressure rating (contested possessions allowed by the opponent)

**Data availability for player props:**

Betfair Data Scientists publish a [AFL Player Disposals tutorial](https://betfair-datascientists.github.io/modelling/AFLPlayerDisposalsFlumine/) (Part 2 covers bet placement via Flumine, Part 1 covers model construction). Their pipeline:
- Exponentially weighted moving averages (span=10) on individual player disposal history
- Features include opponent quality, venue, and team form
- Execution via Betfair exchange (not Sportsbet)

**Realistic player props edge:** Higher theoretical ceiling than team H2H. But Sportsbet account restrictions are a real concern -- if you consistently win on player props, Sportsbet will restrict stake sizes. This is the dominant practical risk for any profitable player-props strategy on Sportsbet specifically.

**Goal scorer markets:** Far higher variance than disposals. A midfielder who averages 0.8 goals/game has enormous game-to-game variance. The market sets lines accordingly with larger margins. Harder to model with positive CLV.

**SuperCoach/fantasy points markets:** These aggregate multiple stat lines, which smooths variance -- but also makes the bookmaker's composite estimate harder to beat since errors in individual components can offset.

---

## 5. Python Library Landscape

There is no Python equivalent of FitzRoy. The Python AFL data ecosystem is fragmented and less mature.

### Available Python tools

**pyAFL** ([github.com/RamParameswaran/pyAFL](https://github.com/RamParameswaran/pyAFL), [PyPI](https://pypi.org/project/pyAFL/))
- Scrapes AFLTables.com
- Provides Team, Player, Season objects with Pandas DataFrames
- Includes request caching
- Last meaningful update: April 2024
- Limitation: scraping-dependent (breaks if AFLTables changes structure), player names must match exactly
- Coverage: historical results and player stats

**AFL-Data-Analysis repo** ([github.com/akareen/AFL-Data-Analysis](https://github.com/akareen/AFL-Data-Analysis))
- CSV flat files: match scores 1897-2025, player performance stats (19M data points, 682K rows), personal player info
- Historical odds data: 2009-2024 (from AusSportsBetting)
- No active code library -- just data files in CSV format
- This is probably the **fastest path to a working training set** without writing any scraper

**AFL Tables Scraper** ([github.com/multimeric/AflTablesScraper](https://github.com/multimeric/AflTablesScraper))
- Python scraper for AFL Tables, CLI + library
- More structured than pyAFL for batch data collection

**Squiggle API (direct)** ([api.squiggle.com.au](https://api.squiggle.com.au/))
- Free, no auth, returns JSON
- Endpoints: games, tips (model predictions), standings, sources, ladder, power rankings
- Use `requests` or `httpx` directly -- no wrapper needed
- This is the easiest way to get bookmaker closing line equivalents (the "Punters" source in the API)

**AFL Champion Data Python package** ([docs.api.afl.championdata.com](https://docs.api.afl.championdata.com/guides/support-packages/python-package-documentation/))
- Official AFL data platform (Champion Data is the official AFL statistician)
- Provides play-by-play, possession chains, advanced metrics not in AFLTables
- Requires credentials -- not publicly accessible. Commercial product.
- This is what professional analysts use. Not available to amateurs.

**Betfair AFL tutorials** ([betfair-datascientists.github.io](https://betfair-datascientists.github.io/modelling/AFLmodellingPython/))
- Full Python pipeline: pandas, numpy, scikit-learn
- Uses: SVM, decision trees, logistic regression, ensemble methods, discriminant analysis, Gaussian process
- Grid search for hyperparameters
- Most practical starting point for a scikit-learn pipeline

### Recommended stack for an amateur AFL model in Python

```
Data:    AFL-Data-Analysis CSV files (historical) + Squiggle API (current season + market odds)
Model:   scikit-learn (LogisticRegression or GradientBoostingClassifier)
Elo:     Manual implementation or `elo` PyPI package
Features: pandas rolling/ewm functions
Odds:    Squiggle API "Punters" source for bookmaker reference
```

---

## 6. Calibration in AFL Models

### Published calibration data

The [Performance of AFL Prediction Models (2018-2020)](https://rstudio-pubs-static.s3.amazonaws.com/613310_24778e5c0c78485b9f6a011198e4c51b.html) is the most complete public calibration analysis. Murphy decomposition of Brier score:

| Model | Brier Score | Reliability | Resolution |
|-------|-------------|-------------|------------|
| Betting Odds | **0.201** | **0.0157** | **0.0288** |
| Aggregate | 0.202 | ~0.018 | ~0.027 |
| Live Ladders | 0.202 | ~0.019 | ~0.027 |
| Squiggle | 0.202 | ~0.019 | ~0.026 |
| Matter of Stats | ~0.203 | *best reliability, best resolution* (among non-market models) |

**What this means:**
- **Reliability** (calibration): When the market says 60%, teams win ~60% of the time. The market is better calibrated than any individual public model.
- **Resolution** (discrimination): The market also better distinguishes high-confidence from low-confidence games.
- Only MoSHBODS showed superior reliability and resolution components compared to odds -- but its overall Brier score was still worse due to accuracy on 50-50 games.

**No reliability diagram from AFL models is publicly published** in any resolution comparable to what FiveThirtyEight publishes for their models. The 2018-2020 analysis includes the decomposition values but not graphical reliability diagrams.

### What calibration means for betting

A perfectly calibrated model is necessary but not sufficient for profit. Even the betting market is well-calibrated -- meaning if you built a model with the same calibration as the market, you would break even before vig (and lose after vig). To profit, you need **systematic miscalibration in the market** that your model captures, not just your own model's calibration being good.

The evidence suggests that in AFL H2H markets, systematic market miscalibration is small and inconsistent year to year. Player props markets have less evidence of systematic calibration because they haven't been rigorously studied publicly.

---

## 7. Honest Recommendation

### Can an amateur model beat the AFL H2H closing line on Sportsbet?

**Probably not consistently, and certainly not in the first season.**

The evidence is clear: the aggregate bookmaker probability in AFL H2H markets achieves Brier scores of ~0.201 -- better than every individual public model tested over 2018-2020. The published 2025 Squiggle data shows some computer models beating the book aggregate on tip count alone, but tip count is the noisiest metric and the sample is partial. When you look at log-score (Bits) and calibration, the market's edge is consistent.

The realistic CLV ceiling for a well-constructed amateur AFL H2H model is probably +0.5% to +1.5% before account restrictions. Sportsbet is known to restrict accounts that show consistent profit, which caps upside further.

### Where should you actually focus?

**Tier 1 target: Player disposals markets (best opportunity)**

This is where the market is softest and the data is most tractable:
- Disposal counts are highly autocorrelated by player role
- Public data is sufficient (AFLTables player stats)
- The bookmaker's line-setting is less sophisticated (fewer resources dedicated to 300+ weekly prop markets)
- The academic literature has not systematically studied these markets, suggesting less competition from sharp bettors
- Historical odds data exists (AFL-Data-Analysis repo has 2009-2024 odds, though it's unclear if player prop odds are included)

**Risk:** Sportsbet will restrict winning prop bettors faster than H2H bettors. Plan for this by betting across multiple accounts or shifting to Betfair exchange for disposals where possible.

**Tier 2 target: Totals (over/under total score)**

The weather-scoring connection (median 178 points dry, 136 in heavy rain) is real and statistically significant. The totals market in AFL is thinner than H2H. A model that incorporates:
- Team scoring style (scoring shots generated per game)
- Venue characteristics (MCG vs suburban grounds)
- Forecast weather (rain probability, wind speed)
- Team pace/tempo ratings

...has a theoretical edge over a bookmaker who sets totals lines with less granularity than H2H lines. This hasn't been published as validated, which is a reason for both optimism (less efficient) and caution (less validated).

**Tier 3 target: Line betting (handicap) rather than H2H**

Line betting (e.g., -14.5 for a heavy favourite) requires a good margin estimate, not just a win probability. Margin models (predicting expected winning margin) are harder to build but the market may be softer on line betting for games with extreme favourites. Your MAE target should be below 27 points (the top models' benchmark on Squiggle 2025).

### Minimum viable model architecture

For player disposals, a plausible first iteration:

1. **Data:** AFL-Data-Analysis player stats CSVs (2012-2025) for training; Squiggle API for current season game context
2. **Target variable:** Player disposals in a specific game (regression, not classification)
3. **Core features:**
   - Player's ewm(span=10) disposal average (most important feature)
   - Player's ewm(span=10) minutes/time-on-ground (proxy for role stability)
   - Opposition contested possessions allowed per game (ewm, span=10)
   - Venue (categorical -- MCG vs other)
   - Home/away status for the player's team
   - Round number (linear trend for early vs late season)
4. **Model:** GradientBoostingRegressor or XGBRegressor -- non-linear interactions matter here
5. **Calibration:** Use isotonic regression to convert regression output to well-calibrated probability (for over/under line betting)
6. **Baseline comparison:** Squiggle "Punters" source for game-level context; you'll need to build your own odds scraper for player props

For team H2H, if you proceed anyway:

1. **Elo ratings** as the primary feature (implement from scratch -- ~50 lines of Python)
2. **HGA term** (home designation + log-distance interstate travel differential)
3. **VPV-style venue adjustment** (per-team rolling mean excess margin at each venue)
4. **Scoring shots ratio** as the Elo update signal (not raw score)
5. **Logistic regression** to convert Elo diff to probability (simple, interpretable, well-calibrated)
6. **Compare to Squiggle's Punters source** game-by-game to identify where your model diverges

### What not to build first

- Do not start with a complex ML pipeline (gradient boosting, neural nets) on team H2H. The data is too sparse (~200 games/year) and the market too efficient. You'll overfit badly and produce confident-but-wrong probabilities.
- Do not invest in weather APIs yet. The public research consensus is that weather data from public sources is not granular enough to reliably exploit.
- Do not model finals. The variance is genuinely different and sample sizes are tiny.

### The account restriction problem

This is the dominant practical constraint and is not discussed in any AFL modelling tutorial. Sportsbet aggressively restricts winning accounts. For any model that generates profit, your effective bet size will be reduced to $5-10 within months. Options:
- Use Betfair exchange for execution (no account restrictions, but thin AFL liquidity)
- Spread across multiple bookmakers (Sportsbet, TAB, Pointsbet, Ladbrokes)
- Accept that H2H model validation via actual betting has a short runway on Sportsbet and plan accordingly

---

## Key Sources

- [Squiggle AFL leaderboard](https://squiggle.com.au/leaderboard/) -- all-time and per-season model rankings with Punters benchmark
- [Performance of AFL Prediction Models (2018-2020)](https://rstudio-pubs-static.s3.amazonaws.com/613310_24778e5c0c78485b9f6a011198e4c51b.html) -- Brier score analysis, calibration decomposition
- [Betfair AFL Python modelling tutorial](https://betfair-datascientists.github.io/modelling/AFLmodellingPython/) -- complete scikit-learn pipeline
- [Betfair AFL player disposals tutorial](https://betfair-datascientists.github.io/modelling/AFLPlayerDisposalsFlumine/) -- player props execution
- [AFL-Data-Analysis GitHub repo](https://github.com/akareen/AFL-Data-Analysis) -- CSV data 1897-2025, odds 2009-2024
- [pyAFL PyPI](https://pypi.org/project/pyAFL/) -- Python scraper for AFLTables
- [Squiggle API](https://api.squiggle.com.au/) -- free JSON API for games, predictions, standings
- [Matter of Stats 2024 review](http://www.matterofstats.com/mafl-stats-journal/2024/10/7/reviewing-moshplays-2024-squiggle-performance) -- honest assessment of model variance and market comparison
- [AFL Bradley-Terry analysis (arXiv 2405.12588)](https://arxiv.org/html/2405.12588v1) -- feature importance: HGA, interstate travel, forward 50 metrics
- [AFL Lab weather analysis](https://theafllab.wordpress.com/2018/08/23/environmental-factors-affecting-afl-outcomes-the-weather-part-2/) -- wet weather scoring effects
- [Figuring Footy predictions methodology](https://figuringfooty.com/the-figuring-footy-predictions/) -- scoring shots Elo system
- [The Arc ratings system](https://thearcfooty.com/) -- FiveThirtyEight-style Elo for AFL
- [HPN Footy wet weather analysis](https://www.hpnfooty.com/?p=31764) -- rain frequency by venue
