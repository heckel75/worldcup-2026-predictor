# World Cup 2026 Predictor — Project Doc

> Living document. Update at the end of every session with what actually happened, what's blocked, and what's next.

---

## 1. What we're building

A public web dashboard that forecasts every match of the 2026 FIFA World Cup and simulates the entire tournament after each result, so the forecast evolves as the tournament unfolds.

**The distinctive feature: three different views of the same match are shown side by side.**

1. A statistical model (Dixon-Coles bivariate Poisson built on Elo ratings)
2. Sportsbook odds (margin-stripped)
3. Polymarket prediction market prices

When all three agree, confidence is high. When they diverge, the dashboard surfaces the gap as a signal.

A second layer (Claude API) writes match previews and explains divergences in natural language. **This is explicitly an experiment** — we want to see whether AI commentary adds genuine value or is decorative. The forecasts themselves are not AI-generated.

### Public-facing tagline

> A World Cup forecast dashboard that combines a statistical model, sportsbook odds, and Polymarket prices to estimate every match and simulate the whole tournament. The interesting part is not just who is favored — it's where the three sources agree, where they diverge, and how those probabilities change after every result.

---

## 2. Scope decisions

### In scope (v1, before June 11)

- Elo ratings for all 48 qualified teams, updated after every match
- Dixon-Coles goal model producing W/D/L probabilities and expected scorelines
- Monte Carlo simulation of the full tournament (10,000+ runs per update)
- Polymarket Gamma API integration
- Sportsbook odds via The Odds API
- Claude-generated match previews and divergence commentary
- Static site on GitHub Pages with:
  - Tournament tree with live probabilities
  - Per-match pages
  - "What changed today" panel (movers, fresh divergences)
  - Calibration tracker
  - Methodology / About page

### Explicitly out of scope

- Live in-play predictions
- Player-level data, automated injury parsing, xG models
- Fancy ML (XGBoost, neural nets, ensembles) — Dixon-Coles + Elo is the model
- User accounts, comments, interactivity beyond browsing
- Fully automated scheduled runs (manual trigger to regenerate is fine)

### Honest limits to communicate publicly

- Good international football models hit ~55–60% match accuracy
- Many matches are genuine toss-ups; many predictions will miss
- The value is calibrated probabilities + tournament simulation, not nailing every game

---

## 3. Architecture (target state)

```
DATA LAYER
  ├─ Historical international matches (Kaggle / openfootball CSV)
  ├─ Recent qualifiers + friendlies (API-Football free tier or scraping)
  ├─ Sportsbook odds (The Odds API)
  ├─ Polymarket prices (Gamma API)
  └─ Squad/injury info (manual + Claude-assisted news parsing)

MODEL LAYER
  ├─ Elo rating system (with home advantage + margin-of-victory)
  └─ Dixon-Coles bivariate Poisson (predicts goals, derives W/D/L)

SIMULATION LAYER
  └─ Monte Carlo: simulate remaining tournament 10,000× per update
     → team-level probabilities for each round and the title

FUSION LAYER
  ├─ Model probabilities (from simulation)
  ├─ Market-implied probabilities (sportsbook)
  ├─ Polymarket-implied probabilities
  └─ Claude: written synthesis + divergence commentary

DASHBOARD (static site, regenerated after each match)
  ├─ index.html       — tournament tree + "what changed today"
  ├─ matches/*.html   — per-match preview pages
  ├─ teams/*.html     — per-team summary pages
  ├─ calibration.html — track record
  └─ about.html       — methodology
```

---

## 4. Tech stack

- **Language:** Python 3.11+
- **Core libraries:** pandas, numpy, scipy (for Dixon-Coles fitting), requests
- **AI:** anthropic (Claude API)
- **Site generation:** Jinja2 templates → static HTML
- **Hosting:** GitHub Pages
- **Repo:** public on GitHub

---

## 5. Session-by-session plan

Each session is **~1 hour of focused work** for a beginner. Do them in order. Some weeks have more sessions than fit cleanly into 10 hours — that's fine, prioritize the earlier ones.

### How to start a new chat session

Paste this doc + one line of status, e.g.:
> "Just finished Session 6, ready for Session 7. Brief blocker: my Elo rating for Argentina looks too low compared to public Elo rankings."

Then we resume.

---

### WEEK 1 — Foundations

**Goal:** clean repo, real data, working Elo rating system, ratings CSV for all 48 WC teams.

**Session 1 — Project setup ✅ DONE**
- Create local folder structure (`data/`, `src/`, `notebooks/`)
- Python venv + install pandas, numpy, requests, python-dotenv
- Initialize Git, create public GitHub repo, push initial commit
- Files created: `.gitignore`, `README.md`, `requirements.txt`

**Session 2 — Understand Elo (no code) ✅ DONE**
- Win probability formula and intuition
- Update rule, K factor, margin-of-victory multiplier
- Home advantage in international football
- Locked in: higher rating = higher win probability, expected results barely move ratings, upsets move them a lot

**Session 3 — Get the data and explore it ✅ DONE**
- Downloaded Kaggle "International football results from 1872 to 2024" CSV (actually current through March 2026)
- Placed in `data/raw/results.csv`
- Wrote `src/explore_data.py`: prints row counts, columns, date range, tournament types
- **Result:** 49,287 total rows. 49,215 played matches available for training. 72 future fixtures including all 72 WC group stage matches. Dataset current through 2026-03-31. **No supplementary data source needed for v1.**

**Session 4 — Clean and filter the data ✅ DONE**
- Wrote `src/clean_data.py`: load CSV, parse dates, filter to matches from 2018-01-01 onwards
- Split data: 7,952 played matches → `data/processed/matches_clean.csv` (training set); 72 unplayed → `data/processed/fixtures_2026.csv` (entire WC group stage)
- Standardized team names via `TEAM_NAME_MAP` (United States→USA, Korea Republic→South Korea, Czechia→Czech Republic, etc.)
- All 48 qualified teams verified present and consistently named
- **Convention locked in:** run all scripts from project root (`python src/script.py`), not from inside `src/`
- **Note for later:** internal team names use dataset conventions (Turkey, Czech Republic, Ivory Coast). When we build the dashboard in Week 5 we'll map these to official display names (Türkiye, Czechia, Côte d'Ivoire). Don't change them now — internal consistency is what matters for the model.

**Session 5 — Implement basic Elo ✅ DONE**
- Write `src/elo.py`: `EloSystem` class with `expected_score`, `update_match`, `get_rating`
- Methods to apply K differentiation by match type (friendly vs qualifier vs major tournament)
- No margin-of-victory yet — keep it simple
- Test on first 100 matches manually, sanity-check ratings

**Session 6 — Add margin-of-victory and home advantage  ✅ DONE**
- Added MoV multiplier (1.0 / 1.5 / (11+diff)/8) and 60-Elo home advantage to src/elo.py
- expected_score gained an optional home_advantage flag; update_match gained an optional neutral parameter
- Wrote src/save_wc_ratings.py to dump the 48 WC teams to data/processed/elo_ratings_2026.csv
- Compared to eloratings.net: top 2 (Spain, Argentina) match exactly; France/England/Brazil/Netherlands/Portugal under-rated; Morocco/Japan/Senegal/Australia/Nigeria over-rated
- Diagnosed as confederation-pool drift from 8-year cold start (see §6)
- Deliverable: data/processed/elo_ratings_2026.csv — committed but contains drifted ratings; Session 7 will fix

**Session 7 — Seed initial ratings from public source (← NEXT)**
- Hand-collect or scrape eloratings.net top ~80 teams as of Jan 1, 2018
- Add a seed_ratings(dict) method to EloSystem (or accept a seed dict in __init__)
- Re-run historical pass on the same 7,952 matches, starting from seeds instead of 1500
- Re-compare top 20 vs eloratings.net — expect Brazil, Netherlands, Portugal back in elite tier
- Re-save data/processed/elo_ratings_2026.csv

---

### WEEK 2 — The match prediction model

**Goal:** given two teams, output P(home win), P(draw), P(away win) and an expected scoreline distribution.

**Session 8 — Understand Poisson goals (no code)**
- Why football scores are roughly Poisson-distributed
- What "expected goals" (λ) means in this context
- Why Dixon-Coles modifies basic Poisson (handles low-scoring draws better)

**Session 9 — Compute team attack/defense strengths from Elo and history**
- Convert each team's Elo to expected goals scored and conceded
- Validate: does France's expected goals against minnows look right?

**Session 10 — Implement Dixon-Coles**
- Write `src/dixon_coles.py`
- Function: `predict_match(home_team, away_team, neutral=True) -> dict`
- Returns: P(home/draw/away), expected goals each side, P(scorelines)

**Session 11 — Fit and validate**
- Use historical data to tune the model's parameters
- Backtest on Euro 2024 + Copa America 2024 (matches the model never saw)
- **Deliverable:** model accuracy and log loss numbers we trust

**Session 12 — Buffer / refinement**
- Address whatever the backtest revealed
- Update `PROJECT.md`

---

### WEEK 3 — Tournament simulation

**Goal:** simulate the entire World Cup 10,000 times and report each team's probability of reaching each round.

**Session 13 — Encode the 2026 bracket**
- Hard-code 12 groups, all 48 teams, group-stage fixture list
- Encode tiebreaker rules (points, GD, goals, head-to-head, fair play)
- Encode knockout bracket structure including the new Round of 32

**Session 14 — Simulate one tournament**
- Function: `simulate_tournament(model, ratings) -> dict of results`
- Plays out all 104 matches given current model, returns winner + path of every team

**Session 15 — Monte Carlo loop + aggregation**
- Run simulation 10,000 times
- Aggregate per team: P(advance from group), P(reach R16), P(reach QF), ..., P(win cup)
- Save snapshots so we can compute "what changed since yesterday"

**Session 16 — Handle the Round of 32 quirk**
- 495 possible bracket configurations depending on which 8 third-place teams advance
- Implement the actual FIFA rules for which 3rd-place teams go where
- Validate against FIFA's published bracket logic

**Session 17 — Buffer / sanity checks**
- Compare title odds to bookmakers — are we wildly off anywhere?
- If yes, debug. If no, document the agreement

---

### WEEK 4 — Markets and AI commentary

**Goal:** pull sportsbook + Polymarket data, compute divergences, generate Claude commentary.

**Session 18 — The Odds API integration**
- Sign up for free tier (500 req/month)
- Pull World Cup outright winner odds + per-match odds
- Strip vig, compute fair probabilities
- Save `data/processed/sportsbook_odds.csv`

**Session 19 — Polymarket integration**
- Hit Gamma API for World Cup markets
- Match Polymarket markets to our matches
- Save `data/processed/polymarket_odds.csv`

**Session 20 — Triple layer comparison**
- For each match: compute model prob, sportsbook prob, Polymarket prob
- Compute divergence metrics
- Flag matches where divergence > threshold

**Session 21 — Claude prompts for match previews**
- Design prompt that takes numerical inputs and produces a 3-paragraph preview
- Cache aggressively (don't re-prompt on every page load)
- Handle errors gracefully

**Session 22 — Claude prompts for divergence commentary**
- Specifically for matches flagged as divergent
- Output: short paragraph naming the gap and offering plausible (not definitive) interpretations
- **Deliverable:** for any match, we can produce all three probability layers + commentary

---

### WEEK 5 — Dashboard build

**Goal:** static site live on GitHub Pages.

**Session 23 — Site skeleton with Jinja2**
- Set up `templates/` folder, `generate_site.py` script
- Output to `docs/` (GitHub Pages serves from there)
- One ugly placeholder page that builds successfully

**Session 24 — Tournament tree page**
- Visual bracket showing current title odds for each team
- Color-coded by probability tier

**Session 25 — Per-match pages**
- One HTML page per upcoming match
- Three probability bars side by side, Claude preview, divergence note

**Session 26 — "What changed today" panel**
- Diff today's snapshot vs yesterday's
- Top 5 movers, top 3 fresh divergences
- This is the habit-forming feature — get it right

**Session 27 — Calibration tracker**
- Track every prediction we've made vs what actually happened
- Reliability diagram + Brier score
- Live as soon as the tournament starts producing results

**Session 28 — Deploy to GitHub Pages**
- Configure repo to serve `docs/` as Pages site
- Register custom subdomain if desired (optional)
- Test the full regeneration flow end to end

---

### WEEK 6 — Polish and stress test

**Goal:** dashboard is robust, regeneration is one command, you trust the system.

**Session 29 — End-to-end update script**
- One script that: pulls fresh data → updates ratings → re-runs simulation → regenerates Claude commentary → rebuilds site → commits and pushes
- Logs everything, handles errors, never silently fails

**Session 30 — Pretend-tournament dry run**
- Use Euro 2024 results: pretend they happen one match at a time
- Run the update script after each "match"
- Confirm the dashboard updates correctly, calibration tracker fills in, etc.

**Session 31 — Methodology page**
- Important for credibility
- Explain Elo, Dixon-Coles, simulation, where AI fits, limits
- Link the Kaggle dataset, Polymarket API, etc.

**Session 32 — Buffer / fix what's broken**

---

### WEEK 7 — Pre-tournament (June 1–10)

**Session 33 — Final data refresh**
- Pull every match up to June 10
- Re-fit model, recompute ratings, run final pre-tournament simulation
- Take a baseline snapshot

**Session 34 — Sharing plan**
- Write launch post (Twitter / LinkedIn / wherever)
- Send to people who tested the original pitch

**Session 35 — Last-minute polish**

---

### TOURNAMENT (June 11 – July 19)

Daily ritual (~30–45 min per match day):
- Pull yesterday's results
- Run the update script
- Spot-check Claude output for any obvious mistakes
- Push the updated site
- Tweet/share the most interesting movers

Buffer for things that break.

---

## 6. Open questions and known risks

- **Recent data gap:** ✅ Resolved in Session 3. Kaggle dataset is actively maintained, current through March 31, 2026 (last FIFA international break before WC). Fresh weekly pulls during the tournament should be sufficient.
- **API rate limits during tournament:** if we hit Odds API or API-Football limits in June, may need to pay ~$25/month. Budget for this.
- **Polymarket liquidity for some markets:** smaller-team matches may have thin Polymarket markets that produce noisy probabilities. Need to flag low-liquidity warnings.
- **Claude API costs:** with 104 matches and re-generation after each, costs could add up. Cache aggressively, only regenerate when inputs actually change.
- **Calibration before tournament:** with no live results, the calibration page will be empty for week 1. Plan: backfill with backtest results from Euro 2024 to demonstrate calibration even before WC starts.
- **Confederation-pool drift in Elo:** Pure Elo from a 2018 cold start over-rates AFCON/AFC teams and under-rates CONMEBOL/some UEFA teams — intra-confederation matches dominate and inter-confederation calibration is sparse. Plan: Session 7 seeds Jan 2018 starting ratings from eloratings.net so the 8 years of data refines real seeds instead of discovering global structure from scratch.

---

## 7. Session log

> Append a one-line entry after each session.

- **Session 1 (2026-05-01):** Project setup complete. Repo live on GitHub. Folder structure, venv, .gitignore, requirements.txt, initial commit pushed.
- **Session 2 (2026-05-01):** Walked through Elo intuition. Confirmed understanding via worked examples: Argentina vs Saudi Arabia upset (~27 Elo gain) and equal-rated draw (0 change).
- **Session 3 (2026-05-01):** Kaggle dataset explored. 49,287 rows, current through 2026-03-31. 49,215 played matches for training, 72 future fixtures (entire WC group stage included). No supplementary data source needed. File: `src/explore_data.py`.
- **Session 4 (2026-05-02):** Wrote `src/clean_data.py`. Filtered to 2018+, parsed dates, standardized team names, split played matches from future fixtures. Outputs: `matches_clean.csv` (7,952 matches) and `fixtures_2026.csv` (72 WC group-stage matches). All 48 qualified teams present and verified against official sources. Convention locked in: run scripts from project root.
- **Session 5 (2026-05-03):** Basic Elo system implemented in src/elo.py. EloSystem class with expected_score, update_match, get_rating, top_n. K varies by match type (friendly 20 / qualifier 30 / major 50 / WC 60). Ran full historical pass on 7,952 matches: 282 teams rated. Top 5: Spain 1919, Morocco 1876, Argentina 1860, France 1834, Japan 1815. African and Asian teams over-rated (no MoV yet), Brazil under-rated at #20 — both expected; Session 6 fixes.
- **Session 6 (2026-05-04):** Added MoV multiplier and 60-Elo home advantage to src/elo.py. New top 5: Spain 2015, Argentina 1960, Morocco 1931, France 1925, Japan 1888. Top 2 match eloratings.net exactly; rest of table shows confederation-pool drift (Brazil at #13 vs public #5, Morocco at #3 vs public ~#12). Saved data/processed/elo_ratings_2026.csv and src/save_wc_ratings.py. Fix deferred to Session 7.
- **Session 7 (← NEXT):** Seed initial ratings from public source to correct confederation drift.

---

## 8. How to get back into a chat session

PROJECT.md is loaded automatically into every new chat via project files, so you don't need to paste it. Just:

1. Make sure the project file is up to date (replace it with your latest local version if you've edited it since the last chat)
2. **Push current code to GitHub before asking for changes** — Claude can then fetch the current state of any source file directly instead of guessing


Then open a new chat and say: "ready for Session N" (add any blockers if relevant).
Repo: https://github.com/heckel75/worldcup-2026-predictor
