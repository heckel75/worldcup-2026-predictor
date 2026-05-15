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

**Session 7 — Seed initial ratings from public source ✅ DONE**
- Hand-collect or scrape eloratings.net top ~80 teams as of Jan 1, 2018
- Add a seed_ratings(dict) method to EloSystem (or accept a seed dict in __init__)
- Re-run historical pass on the same 7,952 matches, starting from seeds instead of 1500
- Re-compare top 20 vs eloratings.net — expect Brazil, Netherlands, Portugal back in elite tier
- Re-save data/processed/elo_ratings_2026.csv

---

### WEEK 2 — The match prediction model

**Goal:** given two teams, output P(home win), P(draw), P(away win) and an expected scoreline distribution.

**Session 8 — Understand Poisson goals (no code) ✅ DONE**
- Why football scores are roughly Poisson-distributed
- What "expected goals" (λ) means in this context
- Why Dixon-Coles modifies basic Poisson (handles low-scoring draws better)

**Session 9 — Compute team attack/defense strengths from Elo and history ✅ DONE**
- Convert each team's Elo to expected goals scored and conceded
- Validate: does France's expected goals against minnows look right?

**Session 10 — Implement Dixon-Coles ✅ DONE**
- Write `src/dixon_coles.py`
- Function: `predict_match(home_team, away_team, neutral=True) -> dict`
- Returns: P(home/draw/away), expected goals each side, P(scorelines)

**Session 11 — Fit and validate ✅ DONE**
- Use historical data to tune the model's parameters
- Backtest on Euro 2024 + Copa America 2024 (matches the model never saw)
- **Deliverable:** model accuracy and log loss numbers we trust

**Session 12 — Buffer / refinement ✅ DONE**
- Address whatever the backtest revealed
- Update `PROJECT.md`

---

### WEEK 3 — Tournament simulation

**Goal:** simulate the entire World Cup 10,000 times and report each team's probability of reaching each round.

**Session 13 — Encode the 2026 bracket ✅ DONE**
- Hard-code 12 groups, all 48 teams, group-stage fixture list
- Encode tiebreaker rules (points, GD, goals, head-to-head, fair play)
- Encode knockout bracket structure including the new Round of 32

**Session 14 — Simulate one tournament ✅ DONE**
- Function: `simulate_tournament(model, ratings) -> dict of results`
- Plays out all 104 matches given current model, returns winner + path of every team

**Session 15 — Monte Carlo loop + aggregation ✅ DONE**
- Run simulation 10,000 times
- Aggregate per team: P(advance from group), P(reach R16), P(reach QF), ..., P(win cup)
- Save snapshots so we can compute "what changed since yesterday"

**Session 16 — Handle the Round of 32 quirk ✅ DONE**
- 495 possible bracket configurations depending on which 8 third-place teams advance
- Implement the actual FIFA rules for which 3rd-place teams go where
- Validate against FIFA's published bracket logic

**Session 17 — Buffer / sanity checks ✅ DONE**
- Compare title odds to bookmakers — are we wildly off anywhere?
- If yes, debug. If no, document the agreement

---

### WEEK 4 — Markets and AI commentary

**Goal:** pull sportsbook + Polymarket data, compute divergences, generate Claude commentary.

**Session 18 — The Odds API integration (← NEXT)**
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
- **Confederation-pool drift in Elo:**  ✅ Mostly resolved in Session 7 by seeding Jan 2018 ratings from eloratings.net (101 teams). Top 5 WC ratings now match public Elo order. Residual drift remains for some CAF/AFC/CONCACAF teams (Morocco, Japan, Mexico) — this is a structural limit of pure Elo with sparse inter-confederation data and won't be eliminated without architectural changes. Acceptable for v1; revisit if Dixon-Coles backtest in Week 2 shows it materially hurts predictions.
- **Model is biased** ~4pp away from draws toward away wins. Confirmed by 5-bin reliability check on 290 backtest matches. Acceptable for v1, but the divergence detector in Week 4 should not flag a 4–5pp gap on draw probabilities as a real divergence — it's our known bias. Possibly traceable to the small fitted ρ (−0.027 vs literature norm of −0.10 to −0.15); a future refinement is to fit ρ on competitive matches only.
- **Host advantage not modeled in v1:** simulate.py and monte_carlo.py use neutral=True for every match. USA / Mexico / Canada play group games at actual home venues and a 60-Elo home bump is supported by the elo.py API. Decide before Session 33 whether to enable host advantage for the 18 host group-stage matches or leave as v1 limitation and disclose it on the methodology page.
- **Model is more concentrated at the top than markets. Confirmed in Session 17 against Polymarket VWAP:** model top-2 (Spain + Argentina) = 50% of title probability vs market top-2 = 34%. Spain at 30% (market 16.5%), Argentina at 20% (market 8.6%). This is structural: pure Elo + Dixon-Coles with no injury/form/squad data assumes peak fitness every game, and small per-match over-confidence on heavy favorites compounds across the 7-match title path. The Week 4 divergence detector should treat large model > market gaps on title odds for the very top of the table as expected behavior to surface, not as candidates for "the model thinks it knows something." For per-match divergences during the tournament, the existing detector logic (offset for the ~4pp away-win bias from §6) is still the right approach — this top-tier concentration is a title-odds-specific phenomenon, not a per-match calibration issue.
---

## 7. Session log

> Append a one-line entry after each session.

- **Session 1 (2026-05-01):** Project setup complete. Repo live on GitHub. Folder structure, venv, .gitignore, requirements.txt, initial commit pushed.
- **Session 2 (2026-05-01):** Walked through Elo intuition. Confirmed understanding via worked examples: Argentina vs Saudi Arabia upset (~27 Elo gain) and equal-rated draw (0 change).
- **Session 3 (2026-05-01):** Kaggle dataset explored. 49,287 rows, current through 2026-03-31. 49,215 played matches for training, 72 future fixtures (entire WC group stage included). No supplementary data source needed. File: `src/explore_data.py`.
- **Session 4 (2026-05-02):** Wrote `src/clean_data.py`. Filtered to 2018+, parsed dates, standardized team names, split played matches from future fixtures. Outputs: `matches_clean.csv` (7,952 matches) and `fixtures_2026.csv` (72 WC group-stage matches). All 48 qualified teams present and verified against official sources. Convention locked in: run scripts from project root.
- **Session 5 (2026-05-03):** Basic Elo system implemented in src/elo.py. EloSystem class with expected_score, update_match, get_rating, top_n. K varies by match type (friendly 20 / qualifier 30 / major 50 / WC 60). Ran full historical pass on 7,952 matches: 282 teams rated. Top 5: Spain 1919, Morocco 1876, Argentina 1860, France 1834, Japan 1815. African and Asian teams over-rated (no MoV yet), Brazil under-rated at #20 — both expected; Session 6 fixes.
- **Session 6 (2026-05-04):** Added MoV multiplier and 60-Elo home advantage to src/elo.py. New top 5: Spain 2015, Argentina 1960, Morocco 1931, France 1925, Japan 1888. Top 2 match eloratings.net exactly; rest of table shows confederation-pool drift (Brazil at #13 vs public #5, Morocco at #3 vs public ~#12). Saved data/processed/elo_ratings_2026.csv and src/save_wc_ratings.py. Fix deferred to Session 7.
- **Session 7 (2026-05-06):** Seeded EloSystem from eloratings.net Jan 2018 ratings (101 teams in data/raw/elo_seeds_2018.csv). One-line change to EloSystem.__init__ adds an optional seed_ratings dict; save_wc_ratings.py loads seeds before rebuilding ratings. New WC top 5 — Spain / Argentina / France / England / Portugal — now matches public Elo order exactly. Brazil back at #7, Netherlands #10, Germany #11; Morocco dropped from #3 to #8. Residual confederation-pool drift remains for Morocco/Japan/Mexico/Algeria but is much smaller than Session 6. Absolute ratings run ~50–80 points higher than eloratings.net (different K/MoV constants — expected, doesn't affect predictions).
- **Session 8 — Understand Poisson goals (no code) (2026-05-06)** Walked through the Dixon-Coles math conceptually. Locked in: we model goals (not outcomes); each team gets a per-match λ; independent Poisson over (λ_H, λ_A) gives a scoreline probability grid from which W/D/L follows; Dixon-Coles adjusts only the (0,0), (0,1), (1,0), (1,1) cells via a single ρ parameter to fix Poisson's mild empirical miss on low-scoring draws. Time-weighting decision deferred to Session 11.
- **Session 9 — Compute team attack/defense strengths from Elo and history (2026-05-07)** Built the Elo→expected-goals mapping.
- **Session 10 — Implement Dixon-Coles (2026-05-08)** Wrote src/dixon_coles.py. predict_from_lambdas builds a Poisson scoreline grid, applies the Dixon-Coles τ correction on the four low-score cells with placeholder ρ = −0.1, renormalizes, and aggregates to W/D/L. predict_match(home, away, ratings, neutral=True) is the team-name wrapper that calls elo_to_lambdas first. Sanity block passes all four invariants (sums to 1, symmetry, mismatch, DC draw > plain Poisson draw). USA-Mexico shows Mexico-favored — known Session 7 residual drift, not a new bug. ρ-fitting and Euro 2024 / Copa 2024 backtest deferred to Session 11.
- **Session 11 (2026-05-08):** Wrote src/backtest.py. Walked Elo forward across 7,952 matches, fit Dixon-Coles ρ via MLE on 5,882 pre-2024-06 training matches → ρ̂ = -0.027 (placeholder was -0.10). Backtested on Jun-Jul 2024 (290 matches total). On the intended target (Euro 2024 + Copa América, 83 matches): acc 0.530 / log loss 0.978 / Brier 0.583. Per-tournament: Copa 0.863 log loss (21% better than random), Euro 1.051 (4% better) — gap is structural, Euro has clustered Elos and many genuine coin flips, not fixable without out-of-scope data. Predictions saved to data/processed/backtest_2024.csv for the calibration page. Next: update ρ default in dixon_coles.py, fold per-tournament reporting into backtest.py.
- **Session 12 (2026-05-08):** Wrote src/check_calibration.py for a 5-bin reliability check on the 290 backtest predictions. Home-win probabilities are excellently calibrated (max gap 4.8pp, only at top end). Draws systematically under-predicted by ~4pp; away wins over-predicted by a matching ~3pp — small enough to ship, but worth offsetting in the divergence detector in Week 4. Added per-tournament breakdown to backtest.py via a shared _metrics() helper and a "MAJORS (Euro + Copa)" combined line. Reproduces Session 11 numbers exactly. Key finding from the breakdown: full-window accuracy (0.638) is inflated by friendlies and qualifiers; on Euro 2024 specifically (the closest analog to a major tournament with clustered Elos) accuracy is 0.490 and log loss 1.051. This — not the full-window number — is what we should expect for WC matches, and is what we'll communicate on the methodology page.
- **Session 13 (2026-05-11):** Encoded the 2026 bracket in src/bracket.py — all 12 groups by FIFA letter (cross-checked against fixtures_2026.csv), the 16-match R32 with FIFA's slot syntax (1A/2C/3CEFHI), and the R16/QF/SF/final tree as match-ID source references. Tiebreaker function rank_group implements FIFA 2026's rules (h2h-first for 3+ tied, overall-only for 2-tied) with a recursive fallback for sub-ties; rank_third_place implements the separate non-h2h ranking. Sanity check passes: 48 teams match, 12 × 6 fixtures, R32 covers all winners/runners-up + 8 third-place slot families, and a hand-built 3-way h2h cycle self-test confirms order. Third-place R32 slot resolution (the 495-row Annex C table) deferred to Session 16 as planned.
- **Session 14 (2026-05-12:)** Wrote src/simulate.py with simulate_tournament(ratings, fixtures, rng) -> dict that plays all 104 matches end-to-end: group scorelines sampled from the Dixon-Coles grid, knockout winners from W/D/L with the draw mass split 50/50 (penalties ≈ coin flips at international level), and a backtracking placeholder assigning the 8 advancing third-place teams to R32 slot families — respects FIFA's group-letter constraints but not Annex C, which Session 16 swaps in. Sampling primitives tested in isolation against analytical distributions (max 0.002 error over 200k samples). Output dict includes team_furthest_round per team, ready for Session 15's Monte Carlo aggregation. Caught and fixed two latent bugs in bracket.py with the same root cause — _break_by_overall and rank_third_place were both passing too-narrow team sets to _aggregate_stats, which silently zeroed out the relevant stats; first run revealed it via Group E (Ivory Coast +6 GD finishing behind Germany +2) and Group D. Added a two-way-tie self-test alongside the existing 3-way h2h test so this can't regress. All matches use neutral=True for v1; host-country group advantage deferred. Open v1 limit: groups where two teams tie on every overall metric (e.g. Group K with Portugal & Colombia in the seed=42 run) fall back to stable sort because we don't pass FIFA world rankings — small effect, address if it shows up.
- **Session 15 (2026-05-13):** Wrote src/monte_carlo.py — wraps simulate_tournament in a 10k loop, tallies per-team furthest-round outcomes, and saves a dated snapshot CSV (data/processed/snapshots/YYYY-MM-DD.csv). Date-based seed (int(YYYYMMDD)) for daily reproducibility. Runtime ~65s for 10k sims on the dev machine. Four sanity checks built in (per-team monotonicity + exact column sums of 32/16/8/4/2/1 for the six rounds), all pass. First real title odds: Spain 28.3%, Argentina 19.8%, France 12.6%, England 5.7%, Portugal 4.6% — top 3 = 60.7% of title probability, materially more concentrated than the sportsbook consensus (~45-55%). Spain's 28% and Brazil's 3.0% will be the two headline model-vs-market divergences once Week 4 lands; Brazil's gap traces to the Session 7 confederation drift already documented in §6. Snapshot infrastructure is the foundation for the "what changed today" panel (Session 26).
- **Session 16 (2026-05-14):** Encoded FIFA's 495-row Annex C R32 third-place lookup in data/raw/r32_annex_c.csv, built from the Wikipedia transcription via src/build_annex_c.py. Validation at build-time and module-load: every row has 8 distinct qualifying group letters, every slot assignment is drawn from the qualifying set, all 495 rows respect the slot families encoded in R32_BRACKET (e.g. "1A vs 3CEFHI"), and no row would produce a same-group R32 rematch. Diagnostic finding (in src/annex_c_diagnostics.py): 0 of 495 Q-sets are forced by the constraints alone (median ~16 valid assignments, max 214) — Annex C is a genuine FIFA design choice, not derivable. Added resolve_third_place_slots(qualifying_groups) → {slot_id: group_letter} to bracket.py with module-level caching; replaced simulate._assign_thirds's backtracking placeholder with a 6-line Annex C lookup. Old-vs-new diff on 3 sample Q-sets: 10/24 slot assignments changed — placeholder was wrong on the majority of Q-sets even though constraint-respecting. Re-ran monte_carlo.py (seed 20260514): top-3 concentration 60.7% → 61.9%, Spain +1.7pp (largest move; partly confounded with seed change from 20260513 baseline, MC noise ≈±0.5pp), rest of top-5 within ±0.7pp.
- **Session 17 (2026-05-14):** Sanity-checked title odds against Polymarket VWAP (defirate aggregator). Model top-2 concentration (Spain 30%, Argentina 20% = 50% of mass) vs market top-2 (France 17.9%, Spain 16.5% = 34%). Spearman rank corr 0.87, 41 of 47 teams agree within 3pp. Concluded the gap is structural (pure Elo+DC compounding over 7 matches, no injury/form data, market tail-risk premium) and a feature for the Week 4 divergence layer, not a bug. No code changes. Added concentration limitation to §6.
---

## 8. How to get back into a chat session

PROJECT.md is loaded automatically into every new chat via project files, so you don't need to paste it. Just:

1. Make sure the project file is up to date (replace it with your latest local version if you've edited it since the last chat)
2. **Push current code to GitHub before asking for changes** — Claude can then fetch the current state of any source file directly instead of guessing
3. Open a new chat and say "ready for Session N" (add any blockers if relevant). Claude will reply with the raw-GitHub URLs of the files it needs; paste them in one block and we resume.

**Repo:** https://github.com/heckel75/worldcup-2026-predictor

## 9. Working protocol (Claude follows these every session)

**At session start.** After reading PROJECT.md and identifying the session, Claude lists every file likely to be touched and provides raw-GitHub URLs in the form `https://raw.githubusercontent.com/heckel75/worldcup-2026-predictor/main/<path>`. Then waits. No work begins until files are pasted. Asking for "one more file" mid-session is a workflow failure — Claude should over-include rather than under-include.

**One step per response.** Each reply corresponds to one discrete step: one file change, one test run, one explanation, or one decision. After each step, Claude waits for the user. Bundling steps makes course-correction harder and gets in the way of the user actually doing the step.

**At session end.** Claude proposes two one-line additions for the user to paste:
1. **Session log line** for §7 (one bullet summarizing what got built, decided, or blocked).
2. **Commit message** for the Git commit covering the session's changes — short, imperative, scoped (e.g. `Session 16: add FIFA Annex C lookup for R32 third-place slots`).