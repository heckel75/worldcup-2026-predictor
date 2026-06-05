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
- Sportsbook odds via The Odds API (two files: per-match W/D/L and per-team title)
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
  ├─ data/processed/divergence_snapshots/YYYY-MM-DD.csv  (daily, from triple_compare.py)
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

**Session 18 — The Odds API integration ✅ DONE**
- Sign up for free tier (500 req/month)
- Pull World Cup outright winner odds + per-match odds
- Strip vig, compute fair probabilities
- Save `data/processed/sportsbook_odds.csv`

**Session 19 — Polymarket integration ✅ DONE**
- Hit Gamma API for World Cup markets
- Match Polymarket markets to our matches
- Save `data/processed/polymarket_odds.csv`

**Session 20 — Triple layer comparison  ✅ DONE**
- For each match: compute model prob, sportsbook prob, Polymarket prob
- Compute divergence metrics
- Flag matches where divergence > threshold

**Session 21 — Claude prompts for match previews ✅ DONE**
- Design prompt that takes numerical inputs and produces a 3-paragraph preview
- Cache aggressively (don't re-prompt on every page load)
- Handle errors gracefully

**Session 22 — Claude prompts for divergence commentary ✅ DONE**
- Specifically for matches flagged as divergent
- Output: short paragraph naming the gap and offering plausible (not definitive) interpretations
- **Deliverable:** for any match, we can produce all three probability layers + commentary

---

### WEEK 5 — Dashboard build

**Goal:** static site live on GitHub Pages.

**Session 23 — Site skeleton with Jinja2  ✅ DONE**
- Set up `templates/` folder, `generate_site.py` script
- Output to `docs/` (GitHub Pages serves from there)
- One ugly placeholder page that builds successfully

**Session 24 — Tournament tree page  ✅ DONE**
- Visual bracket showing current title odds for each team
- Color-coded by probability tier

**Session 25 — Per-match pages  ✅ DONE**
- One HTML page per upcoming match
- Three probability bars side by side, Claude preview, divergence note

**Session 26 — "What changed today" panel  ✅ DONE**
- Diff today's snapshot vs yesterday's
- Top 5 movers, top 3 fresh divergences
- This is the habit-forming feature — get it right

**Session 27 — Calibration tracker ✅ DONE**
- Track every prediction we've made vs what actually happened
- Reliability diagram + Brier score
- Live as soon as the tournament starts producing results

**Session 28 — Deploy to GitHub Pages ✅ DONE**
- Configure repo to serve `docs/` as Pages site
- Register custom subdomain if desired (optional)
- Test the full regeneration flow end to end

---

### WEEK 6 — Polish and stress test

**Goal:** dashboard is robust, regeneration is one command, you trust the system.

**Session 29 — End-to-end update script ✅ DONE**
- One script that: pulls fresh data → updates ratings → re-runs simulation → regenerates Claude commentary → rebuilds site → commits and pushes
- Logs everything, handles errors, never silently fails

**Session 30 — Pretend-tournament dry run ✅ DONE**
- Use Euro 2024 results: pretend they happen one match at a time
- Run the update script after each "match"
- Confirm the dashboard updates correctly, calibration tracker fills in, etc.

**Session 31 — Methodology page ✅ DONE**
- Important for credibility
- Explain Elo, Dixon-Coles, simulation, where AI fits, limits
- Link the Kaggle dataset, Polymarket API, etc.

**Session 32 — Buffer / fix what's broken**

---

### WEEK 7 — Pre-tournament (June 1–10)

**Session 33 — Final data refresh ✅ DONE**
- Pull every match up to June 10
- Re-fit model, recompute ratings, run final pre-tournament simulation
- Take a baseline snapshot

**Session 34 — Sharing plan ✅ DONE**

**Session 35a — Polymarket fetcher fix (← NEXT)**
Unplanned insert from the 2026-06-05 pre-launch check. Its own session because it's
live-API debugging of unknown size and must not sit on the launch-eve critical path.
- fetch_polymarket.py returns empty across ALL market types (title/group/per-match) — see §6. Markets exist on the live site; our fetcher can't see them (slug/JSON-shape drift since Sessions 18–19).
- Re-derive current WC event slugs + market JSON shape from the live Gamma API, update the fetcher, replace the hardcoded "as of 2026-05-17" per-match string with a real live probe.
- Needs Claude Code. When starting, paste src/fetch_polymarket.py so the work order is written against the actual current code.
- While here (small, can ride along): link per-match pages so every fixture is browsable (§6) — pages exist since Session 25 but nothing links to non-divergent/non-mover ones. generate_site.py + template change.

**Session 35 — Last-minute polish + launch**
Do as close to June 11 / launch eve as possible, in one pass — all of it wants the freshest data.
- Real launch baseline (§6 "Launch baseline is provisional"): download fresh martj42 results.csv → data/raw/ → run update.py → take new baseline snapshot. No code changes. "What changed today" must diff against this June baseline, not 2026-05-31.
- Re-run src/fetch_polymarket.py (now fixed in 35a): per-match h2h markets populate polymarket_odds.csv → third bar on match pages + per-match divergence layer light up.
- Fill the {TOKENS} in launch_copy.md with refreshed numbers (gated on the baseline re-pull above).
- Build a standalone shareable launch graphic (PNG) — the Twitter/Reddit copy leans on "one striking visual" but the survival grid only lives inside the page, not as an exportable image.
- Custom domain (optional, non-blocking): shareability + credibility over the github.io path; ideally before posting widely.
- Suggested launch timing: public posts ~June 7–8, re-pulling data the morning you post — late enough for real numbers, early enough to build an audience before kickoff + first-upset amplification.

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
- **Calibration page shows raw model, published per-match bars show bias-corrected:** The calibration tracker (Session 27) is seeded from backtest_2024.csv, which holds uncorrected Dixon-Coles probabilities, while the per-match pages (Session 25) publish probs with the ~4pp draw/away bias correction applied (offsets from §6, read live from the backtest). So the reliability diagram describes the raw model, not exactly the numbers a reader sees on a match page — the correction is precisely what closes the draw under-prediction the diagram displays. Pre-tournament the page is labeled as backtest; once wc_predictions.csv fills in June (Session 29), decide whether to log raw or corrected probs there — corrected is the more honest "did our published forecast calibrate" measure and is the recommended choice. Disclose both on the methodology page (Session 31).
- **Host advantage:** ✅ Resolved in Session 33. The 9 host-team home group-stage matches (USA/Mexico/Canada ×3 each) run neutral=False with a 60-Elo home bump in simulate.py/monte_carlo.py; triple_compare.py sets USE_FIXTURE_NEUTRAL=True to match and drops host-exclusion (flags 14→26). USA advance 67.6→79.3%, Mexico +1.6pp, Canada +1.2pp. Host matches still flag as model-over-confident vs market (expected). Knockout stays neutral (host venues not guaranteed). Match pages read the neutral column for a "HOME VENUE — {host}" label and drop the old "not modeled in v1" caveat.
- **Model is more concentrated at the top than markets. Confirmed in Session 17 against Polymarket VWAP:** model top-2 (Spain + Argentina) = 50% of title probability vs market top-2 = 34%. Spain at 30% (market 16.5%), Argentina at 20% (market 8.6%). This is structural: pure Elo + Dixon-Coles with no injury/form/squad data assumes peak fitness every game, and small per-match over-confidence on heavy favorites compounds across the 7-match title path. The Week 4 divergence detector should treat large model > market gaps on title odds for the very top of the table as expected behavior to surface, not as candidates for "the model thinks it knows something." For per-match divergences during the tournament, the existing detector logic (offset for the ~4pp away-win bias from §6) is still the right approach — this top-tier concentration is a title-odds-specific phenomenon, not a per-match calibration issue.
- **Sportsbook coverage on May 17:** The Odds API returned all 50 group-stage h2h markets (plenty of books) and a populated outright winner market. We're not waiting on data availability for Week 4; we can build against real data from day one.
- **Polymarket per-match coverage:** As of 2026-05-17, Polymarket has not posted per-match h2h markets for the WC. Their coverage today is 1 title event + 12 group-winner events + 7 misc futures. Per-match markets typically appear close to kickoff and per-match-day during the tournament. src/fetch_polymarket.py writes a header-only polymarket_odds.csv so Session 20's divergence detector has a stable join schema; re-running the fetcher during the tournament will populate it. Liquidity will vary — smaller-team matches will need a low-liquidity flag when they arrive.
- **Per-match divergence-type breakdown** (Session 20 baseline, 2026-05-18): Across the 50 group-stage matches with current sportsbook coverage, model-corr-vs-book splits 25 over-concentrated / 22 under-concentrated / 3 disagree-on-favorite. The near-balanced over/under split is the per-match analog of the title-odds concentration finding above — when model and book pick the same winner, the model is roughly as often too confident as not confident enough. True disagreements on the favored team are rare (3 of 50). Sessions 21–22 will use divergence_type as a branching input so Claude commentary fits the gap shape (e.g. "the model trusts the Elo edge more than the market" vs "the market sees something Elo can't"). The 15pp flag threshold was set on signal density at this coverage level; revisit if late-posted matches change the distribution.
- **AI commentary occasionally produces imprecise comparatives.** Session 22's prompt v2 fixed structural failures (fabricated combined gaps, decimal percentages, off-list categories like tactics). Residual: vague comparisons like "roughly half" can be quantitatively off by a non-trivial factor even when the qualitative direction is right, and the categories rule occasionally leaks ("momentum"). Acceptable for v1 — the numbers themselves are right there in the data block next to the prose. Methodology page should disclose that commentary is generated by Claude and is not a substitute for reading the actual probabilities.
- **Tournament-tree page is a survival grid pre-tournament, not a drawn bracket.** R32 slots (1A, 3CEFHI, …) can't be filled until group results exist, so the index page shows per-round survival probabilities color-coded by tier. A populated knockout bracket becomes meaningful once the group stage resolves slots — revisit as an added view during the tournament.
- **"What changed today" panel and divergence-snapshot mechanism (Session 26):** `triple_compare.py` now writes a dated divergence snapshot (`data/processed/divergence_snapshots/YYYY-MM-DD.csv`) after each run, mirroring the MC snapshot pattern. `src/whats_changed.py` diffs the two newest files in each snapshot dir: title-odds movers use `p_champion`, group-advance movers use `p_advance`, fresh divergences = `flag_divergent==True` in curr but not in prev. `MIN_MOVE_PP = 0.5` suppresses MC-noise jitter (≈±0.5pp). Panel shows a pre-tournament baseline message until live results shift probabilities; fresh divergences start accumulating as new h2h markets are posted closer to kickoff.
- ✅ RESOLVED in Session 30 — **Simulation is not yet result-aware (as of Session 29):** update.py orchestrates the full pipeline, but simulate.py still plays all 104 matches from the group stage every run (Session 14 behavior). Correct pre-tournament; the moment June 11 arrives the sim must fix played results and simulate only the remainder, or the evolving forecast is wrong. Session 30's Euro-2024 dry run is where this gets forced and hardened — it's the single biggest remaining piece before the tournament. Done: simulate.py now pins played group results by scoreline and knockout results by advancing team; dry_run.py replays a full tournament and all invariants pass.
- **Dry run contaminates the real matches_clean.csv (Session 30):** dry_run.py overrides clean_data's *input* (WC_MANUAL_RESULTS → temp feed) but clean_data's *output path is not overridable* — it always writes the real data/processed/matches_clean.csv. So a replay leaves 104 fake WC results in matches_clean.csv, and any hand-run of monte_carlo.py afterward pins the entire fake tournament (observed: Portugal 100% champion). Recovery is a single `python src/clean_data.py` because the real wc_results_manual.csv is empty — clean rebuilds from results.csv + the clean feed and the fakes wash out. PROPER FIX (do before relying on the harness again, and certainly before June): either make clean_data's output path overridable so the dry run writes to a temp matches_clean.csv, or have dry_run.py rebuild matches_clean.csv from the real feed on exit. The harness claims "Ledger restored, temp feed deleted cleanly" but does NOT restore matches_clean.csv — that's the gap.
- **Manual-feed schema must be carried through clean_data explicitly (Session 30):** the historical results.csv and the hand-maintained wc_results_manual.csv have *different schemas*. The new `advanced` column was silently dropped by clean_data until fixed (always-concat, historical rows → NaN). Any future manual-feed column (referee, venue, etc.) hits the same trap: it must be explicitly added to clean_data's output or it vanishes with no error. Always re-check the matches_clean.csv header after adding a feed column.
- **Dry run writes June/July snapshots into the real snapshots dir (Session 30):** "clear stale snapshots at startup" operates on the real data/processed/snapshots/. May snapshots survive and sort before June by filename, but the dry run's synthetic-date June/July snapshots are left behind as residue and must be deleted after a run (anything dated June/July 2026 pre-tournament is fake). A WC_SNAPSHOT_DIR override to isolate the harness to a temp dir is the clean fix — deferred, low priority.
- **generate_site.py NaN calibration-bin crash:** ✅ Resolved in Session 33. `_cal_svg` uses `pd.isna()` to skip empty/NaN reliability bins before `round()`, so the site keeps building through the first live match-days.
- **Launch baseline is provisional:** The 2026-05-31 snapshot was taken on March-31 data (Kaggle martj42 not yet refreshed past the last international break). The real launch baseline is the early-June re-pull (fold into Session 35 / launch morning): download fresh martj42 results.csv → data/raw/ → run update.py → take a new baseline. No code changes — Session 33 did the structural work. "What changed today" should diff against the June baseline, not 2026-05-31.
- **Polymarket fetcher returns empty across ALL market types (2026-06-05):** Re-ran src/fetch_polymarket.py 6 days pre-kickoff. Every layer now empty — title slug `2026-fifa-world-cup-winner-595` returns HTTP 200 but parses 0 markets, all 12 group-winner slugs 200-but-empty, all 48 teams "not found"; per-match still prints the hardcoded "as of 2026-05-17" string (a static message, not a live probe). Meanwhile Polymarket's live site has a full WC section (polymarket.com/sports/world-cup/ — games/groups/winner tabs, ~38 active markets). Diagnosis: Polymarket restructured WC markets since Sessions 18–19; the fetcher's hardcoded slugs / expected JSON shape no longer match. CODE fix, not a wait. Scope: re-derive current event slugs + market JSON shape from the live API, update fetch_polymarket.py, replace the static per-match string with a real live probe. Own session (35a) before launch eve.
- **Per-match pages exist but are unreachable unless flagged (2026-06-05):** Session 25 built docs/matches/<key>.html for all 72 fixtures, but the index grid was "intentionally not linked yet" — the only paths to a match page are the "what changed today" panel and divergence callouts. A fixture that's neither a mover nor divergent has a page nothing links to. Fix: link the survival-grid matches (or add a browsable fixtures list) to their pages in generate_site.py + templates. Small change; pages already built. Folded into Session 35a.
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
- **Session 18 (2026-05-17):** Wrote src/fetch_odds.py — fetches WC 2026 odds from The Odds API, strips vig with proportional-then-average-across-books method, saves data/processed/sportsbook_odds.csv (50 matches, p_home/p_draw/p_away, 37–44 books each) and data/processed/sportsbook_outrights.csv (48 qualified teams, p_winner). Auto-discovers WC sport keys, dumps raw JSON to data/raw/odds_api/ for cache-based reprocessing, prints quota headers per call (12/500 used after dev run). Filters non-qualified teams (Italy, Denmark, Poland, Kosovo, Jamaica, Bolivia) from outrights and renormalises over the 48 qualified — without this, dead-money mass deflates qualified teams' probs ~2.5pp. Sportsbook top-2 concentration 29.0% (Spain 14.9% / France 14.2%) sits below Polymarket's 34% and well below the model's 50%, making sportsbook our most diffuse data source — a useful third leg for the Week 4 divergence detector. Argentina 8.68% matches Polymarket 8.6% almost exactly.
- **Session 19 (2026-05-17):** Wrote src/fetch_polymarket.py — hits the public Gamma API by slug (the /tags endpoint was a dead end; only one curated soccer-league tag returned, no WC). Saves polymarket_outrights.csv (48 teams) and polymarket_groups.csv (NEW: 4 teams × 12 groups = 48 rows; a third comparison layer sportsbook can't give us, directly comparable to the model's P(advance from group)). Per-match h2h markets don't exist on Polymarket yet (20 active WC markets = 1 title + 12 groups + 7 misc); they'll appear closer to kickoff and per-match-day during the tournament, so polymarket_odds.csv is written header-only for now and the script is idempotent. Title top-3 — France 17.1% / Spain 16.0% / England 10.9% — matches the DeFiRate aggregator closely. Three-way title concentration is now triangulated: sportsbook 29% top-2 < Polymarket 33% < model 50%.
- **Session 20 (2026-05-18):** Wrote src/triple_compare.py. Runs Dixon-Coles on all 72 WC fixtures, applies bias correction with offsets read live from backtest_2024.csv (home +0.6pp / draw −3.9pp / away +3.2pp — matches Session 12), left-joins sportsbook_odds.csv and polymarket_odds.csv on (home_team, away_team), and writes data/processed/triple_compare.csv with signed/max/L1 divergence metrics. First pass at 8pp flagged 28/50 matches — too noisy; raised to 15pp and added a 3-category divergence_type column (disagree_on_favorite / model_over_concentrated / model_under_concentrated) that Session 22's prompts will branch on. Across the 50 matched matches: 3 / 25 / 22 split. 14 matches flag at 15pp; 9 host-country matches get a host advantage not modeled in v1 note and are excluded from flag. neutral=True everywhere to match simulate.py; USE_FIXTURE_NEUTRAL constant lets Session 33 flip it in one line. Polymarket join is a no-op today (header-only file) and auto-populates when per-match h2h markets appear closer to kickoff.
- **Session 21 (2026-05-19):** Wrote src/generate_previews.py — Haiku 4.5 generates 3-paragraph match previews from triple_compare.csv inputs, cached as JSON per match under data/processed/previews/ keyed on SHA1 of (rounded probs at 1pp + prompt version + model name) so trivial book ticks don't trigger regeneration but real input drift does. Iterated the system prompt twice during testing: v1→v2 closed a world-knowledge leak (Mexico-as-host language appeared despite neutral=True venue), v2→v3 added an arithmetic-verification rule after a 5pp Brazil-Morocco gap was described as "less than 1pp". Clean on the four hand-checked shapes — heavy mismatch (Spain-Cape Verde, 91%), disagree-on-favorite (Ghana-Panama), big model-vs-book gap (Brazil-Morocco), no-sportsbook fallback (Mexico-Czechia). Polymarket per-match still absent and gracefully degrades. ~$0.20 for 72 previews; cache keeps re-runs free. Output is the base preview every match gets; Session 22 layers divergence_type-branched commentary on the 14 flagged matches.
- **Session 22 (2026-05-20):** Wrote src/generate_divergences.py — divergence_type-branched commentary on top of the Session 21 preview pipeline. Three buckets per match (commentary / note / skip), mutually exclusive by construction; 14 flagged matches → Claude commentaries, 9 host-excluded → deterministic one-line note records (no LLM), 49 → skip. One system prompt + three framing notes selected by divergence_type; PROMPT_VERSION + divergence_type in input_hash so category flips auto-invalidate. v1 failed on two arithmetic patterns hand-checked on the 4-match smoke test — Australia-Turkey combined two outcomes into a fabricated "31pp opposite-direction gap" and Ivory Coast-Ecuador shipped decimal percentages with a "nearly equal 29 vs 16" comparative; v2 added an explicit no-combined-gap rule, round-half-up integer-percentage rule, expanded prohibited-categories list (tactics/formations/strategy/momentum), and reshaped the disagree_on_favorite framing to "two facts, one outcome's gap". Residual: occasional imprecise comparatives ("roughly half" when actually a third) survive — documented in §6 as a methodology disclosure rather than chased with v3. ~$0.05 for 14 commentaries; sibling-file architecture means preview cache untouched. Output: data/processed/divergences/*.json, 23 files.
- **Session 23 (2026-05-21):** Scaffolded the static site. generate_site.py lives at repo root and is a pure data-consumer (reads the newest snapshot by filename, never imports the model). Jinja2 inheritance: templates/base.html is the shared shell (head/nav/footer + {% block content %}, every link {{root}}-prefixed so Session 25 subpages work without touching it); templates/index.html extends it and lists title odds. static/style.css is a deliberately plain baseline with all theming in a :root variables block for Session 24. Generator copies static/ → docs/ and writes docs/.nojekyll. Builds clean — docs/index.html renders all 48 teams ranked (Spain 30.0% → South Africa 0.0%). Added jinja2>=3.1 to requirements.txt. Convention: docs/ is 100% build output and must be committed (GitHub Pages serves from it).
- **Session 24 (2026-05-22):** Built the tournament forecast page — replaced the placeholder table with a FiveThirtyEight-style survival grid (green ramp for advance→final, gold "hero" ramp for the champion column on its own finer scale) and committed the real editorial aesthetic in style.css (Fraunces / Hanken Grotesk / Spline Sans Mono, warm-paper theme, tier classes driven by :root boundaries). generate_site.py now owns presentation: a DISPLAY_NAMES map (Türkiye/Czechia/Côte d'Ivoire/Cabo Verde), a group-letter table duplicated from bracket.py to keep the generator decoupled from the model layer, and the tier-bucketing logic. Decision: the pre-tournament page is the survival grid; a literal populated bracket is deferred until June, when group results resolve the R32 slots.
- **Session 25 (2026-05-23):** Built per-match pages — new templates/match.html (extends base.html, root="../") and a 72-fixture loop in generate_site.py writing docs/matches/<key>.html. Three stacked probability bars (model-corrected / sportsbook / Polymarket) reusing the Session 24 type system with a dedicated W/D/L colour trio; graceful degradation for no-book matchday-3 rows and the still-header-only Polymarket file. Divergence callout reads divergences/<key>.json: commentary rows get a deterministically-computed headline gap plus Claude's paragraph, host rows get a muted caveat note. Preview prose from previews/<key>.json with an explicit Claude-authored credit. Index grid intentionally not linked yet. Fixed a Windows date-format bug (%-d) and pandas NaN handling for absent markets.
- **(2026-05-23):** Switched the working model to Claude Code in VS Code for repo work, with planning chats as the design layer — see new §10. Created CLAUDE.md in repo root (imports PROJECT.md via @PROJECT.md, adds working rhythm + hard conventions). Sessions 1–25 were built in planning chats; Session 26 onward uses Claude Code.
- **Session 26 (2026-05-24):** Built the "What changed today" panel. Added daily divergence snapshots (`data/processed/divergence_snapshots/`) from `triple_compare.py`, mirroring the MC snapshot pattern, so fresh divergences can be diffed over time. New pure, self-tested `src/whats_changed.py` computes top-5 title-odds movers, top-5 group-advance movers (with a `MIN_MOVE_PP=0.5` floor so MC noise ≈±0.5pp doesn't show as movement), and top-3 fresh divergences (newly-flagged vs prior snapshot). `generate_site.py` renders the panel on `index.html` with match-page links and a pre-tournament baseline empty state; survival grid untouched.
- **Session 27 (2026-05-25):** Built the calibration tracker. New pure, self-tested src/calibration.py (pooled reliability bins, per-outcome predicted-vs-observed, multi-class Brier, accuracy) with a regression anchor reproducing Session 11's majors numbers (n=83, Brier 0.583, acc 0.530). generate_site.py renders docs/calibration.html — inline-SVG reliability diagram + per-outcome table seeded from backtest_2024.csv (majors framing primary), labeled as pre-tournament backtest. Added header-only data/processed/wc_predictions.csv as the dormant live ledger; live append deferred to Session 29.
- **(2026-05-25):** Formalized the per-session loop in §9 — explain-and-decide is now a hard checkpoint *before* any work order; close-out always ends by asking whether the way of working should change.
- **Session 28 (2026-05-26):** Deployed the site to GitHub Pages — Settings → Pages, "Deploy from a branch," main / /docs. Default URL (no custom domain for v1): https://heckel75.github.io/worldcup-2026-predictor/. Pre-deploy audit confirmed zero root-absolute links (href="/", src="/", url(/)) so the subfolder URL doesn't break asset/subpage links; .nojekyll present. Verified live: index survival grid, a match page (root="../" resolves), and calibration.html (inline-SVG reliability diagram, majors framing) all render styled. Classic branch-deploy doesn't surface in the Actions tab — the Settings → Pages "currently being built from /docs" message is the real status indicator. No code changes; docs/ rebuilt and committed under the earlier Session 28 commit. Site-build half of the loop proven end-to-end; full data→site pipeline is Session 29.
- **Session 29 (2026-05-27):** Built the end-to-end orchestrator. New update.py at repo root runs 10 stages via subprocess (odds/Polymarket SOFT, clean_data/ratings/MC/triple_compare FATAL, ledger update in-process, previews/divergences SOFT, site build FATAL), logging each stage's output to a timestamped logs/update_*.log; SOFT failures warn and keep the stale on-disk artifact, FATAL failures abort non-zero before publishing. clean_data.py now prepends data/raw/wc_results_manual.csv (the hand-maintained results feed — group + knockout rows added after each match day) before name-standardization and dedupes keeping the manual row. New self-tested src/update_ledger.py: freeze_new_forecasts logs corrected probs for fixtures within LEDGER_LOOKAHEAD_DAYS=1, frozen once and never overwritten; attach_results fills actual post-match without touching probs; played-but-never-frozen rows stay unscored. Full run exits 0 in 341s; wc_predictions.csv confirmed dormant (all fixtures ≥15 days out). .gitignore ignores logs/ but tracks the manual feed. Script deliberately does no git. Deferred to Session 30: making the simulation result-aware, and wiring the calibration page to the live ledger.
- **Session 30 (2026-05-28):** Made the simulation result-aware and built the replay dry-run harness — the biggest remaining pre-tournament piece. simulate_tournament gains known_results (group: (home,away)→(hg,ag) pinned by scoreline substitution; KO: frozenset({a,b})→winner, validated against the node's resolved teams, raises on mismatch) and return_results mode. New src/clock.py: today() returns date.today() unless WC_ASOF_DATE is set, routed through monte_carlo/triple_compare/update_ledger so the harness can inject an as-of date. wc_results_manual.csv + clean_data.py gain an `advanced` column (advancing team for KO only; blank otherwise) via always-concat so historical rows get NaN. update.py honors WC_SKIP_FETCH=1 and WC_MANUAL_RESULTS. New dry_run.py replays one fixed-seed WC (truth) day-by-day through the real update.py, freezing forecasts before each match-day and attaching results after, asserting invariants each day. Full run: ALL INVARIANTS PASSED, exit 0 — truth champion (Portugal, seed 42) ramps ~4% (groups) → 7.5% (QF) → 18% (SF) → 100% (final); 72/72 group rows frozen-and-scored in the ledger; eliminated teams collapse to 0%. Drive-by fixes caught in review: monte_carlo._load_fixtures now loads all 72 fixtures (latent bug; confirmed does NOT move baseline — Spain 28.9/Argentina 19.9/France 11.8 matches Session 16), save_wc_ratings uses bracket.GROUPS for the team list, triple_compare early-exits on empty fixtures. Found and recovered a dry-run contamination of matches_clean.csv (see §6). Deferred to Session 31: wiring calibration.html to the live ledger.
- **Session 31 (2026-05-29):** Built the methodology page. New `templates/methodology.html` (extends base.html, root="./") rendered by `generate_site.py` to `docs/methodology.html`; nav wired across index/match/calibration. Four sections: how it works (Elo→Dixon-Coles→Monte Carlo→three-way fusion + Claude layer), accuracy (majors backtest n=83: acc 0.530 / log loss 0.978 / Brier 0.583, Euro-2024 0.490 as the honest WC analog, ~55–60% ceiling), limits (top-tier concentration, ~4pp draw/away bias corrected on match pages, confederation drift, calibration-page-raw-vs-match-page-corrected disclosure, host advantage not modeled in v1 — neutral everywhere, revisit Session 33), and the AI-commentary experiment (Claude-generated, forecasts are not, judge for yourself). Backtest numbers read live from backtest_2024.csv. Builds clean, no root-absolute links, live on Pages.
- **Session 32 (2026-05-30):** Added `WC_CLEAN_OUTPUT` / `WC_SNAPSHOT_DIR` / `WC_DIVERGENCE_SNAPSHOT_DIR` env-var resolvers to `src/clock.py`; wired into `clean_data.py` (dynamic output path + schema guard), `monte_carlo.py`, `triple_compare.py`, `update.py`, and `dry_run.py` (temp-dir isolation, env cleanup in `finally`). `python dry_run.py` → ALL INVARIANTS PASSED, zero contamination of real `data/processed/` paths; `WC_SKIP_FETCH=1 python update.py` runs all 10 stages clean. Dry run surfaced a pre-existing `generate_site.py` NaN-bin crash on the first ~3 live match-days (logged in §6, fix before June 11).
- **Session 33 (2026-05-31):** Final pre-tournament refresh + host advantage + NaN-bin guard. Kaggle data unchanged (still 7,952 training matches through Mar 31 — martj42 not refreshed upstream; real refresh deferred to early June). simulate.py/monte_carlo.py apply a 60-Elo home bump to the 9 host group-stage matches (neutral=False threaded through the fixture dict); triple_compare.py USE_FIXTURE_NEUTRAL=True, host-exclusion removed (flags 14→26). generate_site.py: match pages show host-aware venue label + caveat from the neutral column, and _cal_svg guards NaN bins. Title odds steady (Spain 29.5 / Argentina 19.8 / France 12.2); USA advance 79.3%. Baseline snapshot 2026-05-31 is provisional — see §6.
- **Session 34 (2026-06-03):** Wrote launch copy (launch_copy.md) — Twitter thread, LinkedIn post, r/soccer + r/dataisbeautiful Reddit posts, private pitch-tester note. Lead hook across all channels is the divergence angle anchored on the model-vs-market title gap (Spain ~30% model vs ~16% market), chosen over title-odds and calibration framings as the most debate-generating/shareable. Channels: Twitter/LinkedIn/Reddit. Decision: write-now/post-later — all numbers left as {TOKENS} to fill from the early-June re-pull at launch (Session 35), since the May-31 baseline is still on March-31 data. Distribution plan emphasises a public return loop (model-vs-market scoreboard resolving after each match), one striking visual per platform, and riding the pre-kickoff / first-upset news cycle. Custom domain flagged as a Session 35+ shareability/credibility task, non-blocking. No code.
---

## 8. How to get back into a chat session

PROJECT.md is loaded automatically into every new chat via project files, so you don't need to paste it. Just:

1. Make sure the project file is up to date (replace it with your latest local version if you've edited it since the last chat)
2. **Push current code to GitHub before asking for changes** — Claude can then fetch the current state of any source file directly instead of guessing
3. Open a new chat and say "ready for Session N" (add any blockers if relevant). Claude will reply with the raw-GitHub URLs of the files it needs; paste them in one block and we resume.

**Repo:** https://github.com/heckel75/worldcup-2026-predictor

## 9. Per-session loop (planning chat ↔ Claude Code)

Every session follows these five steps in order. Each is its own turn; the chat waits for the user between them.

1. **User says "ready for Session N."** Claude explains the session — what it builds, what it reuses, and the real decisions to make — then asks the user to decide. Claude stops and waits. No work order yet.
2. **User answers the decisions.** Claude then sends the Claude Code work order (files to read first, changes, checks, commit message).
3. **User runs it in Claude Code and reports back.** Paste an error/output/diff only if something's off; otherwise just confirm done. Claude Code does not commit/push on its own — the user drives git after reviewing diffs.
4. **Claude sends the close-out:** the PROJECT.md updates (§5 status + NEXT pointer, any §6 risk additions, §7 log line) and the commit message.
5. **Claude asks whether the user wants any change to the way of working** before moving on.

Still in force from the old §9: one logical step per response; over-include rather than under-include files to read; in Claude Code, plan mode (Shift+Tab) is the review checkpoint.

## 10. Working model: Claude Code in VS Code + planning chat (from Session 26)

Two separate Claude surfaces with NO live link between them. The user is the
bridge; nothing syncs automatically.

- **Planning chat** (claude.ai / app): Claude runs on Anthropic's servers and has
  **no access to the repo**. Used to think through a session, decide the approach,
  and produce a written work order. It works from THIS doc, not from your files.
- **Claude Code** (VS Code extension): Claude runs on the user's machine with full
  repo access — reads/edits files in place, runs scripts, shows diffs. Does the
  actual implementation.
- **Shared memory = CLAUDE.md + PROJECT.md only.** `CLAUDE.md` in the repo root
  imports this file via `@PROJECT.md` and Claude Code auto-reads it every session.
  Keep PROJECT.md current at each session end — it is the handoff between surfaces.

### Per-session loop
1. **Plan in a chat.** Output is a work order (files, changes, checks, commit
   message), written from PROJECT.md — not from fetched code.
2. **Execute in Claude Code.** Paste the work order. Use plan mode (Shift+Tab) for
   non-trivial work; review every diff before accepting (don't auto-accept); run
   scripts from the project root; one logical change per commit with a
   `Session N: …` message; push.
3. **Close in the chat.** Paste an error/output/diff only if something's off;
   otherwise confirm. Claude gives the one-line §7 log entry.

### What this supersedes
- §8 (push before each session, paste raw GitHub URLs) is **obsolete for Claude
  Code work** — it reads the disk directly. If you ever plan in a chat and need it
  to see a file, **paste the file's contents**, not a raw URL: planning-chat
  raw-GitHub fetches proved stale/cached (they ate the start of Session 25). Don't
  relitigate fetch staleness — paste, or rely on PROJECT.md.
- §9 "one step per response" — in Claude Code, **plan mode is that review
  checkpoint**, so the same safety holds with less friction. 