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

**Session 32 — Buffer / fix what's broken ✅ DONE**

---

### WEEK 7 — Pre-tournament (June 1–10)

**Session 33 — Final data refresh ✅ DONE**
- Pull every match up to June 10
- Re-fit model, recompute ratings, run final pre-tournament simulation
- Take a baseline snapshot

**Session 34 — Sharing plan ✅ DONE**

**Session 35a — Polymarket fetcher fix ✅ DONE**
Polymarket restructured WC markets since Sessions 18–19; root cause was slug drift, not
an outage. Fixed fetch_polymarket.py: title slug world-cup-winner, group slug
world-cup-group-{letter}-winner, hybrid /public-search fallback if a seed slug misses.
Per-match h2h markets now exist for all 72 group fixtures (series soccer-fifwc/11433 —
avoid the fif-/10238 decoy series) — wired a real per-match probe: three binary Yes/No
child markets per event (home-win / draw / away-win), renormalised over the ~0.995
over-round, teams read from structured event["teams"] on name+ordering (never
abbreviation — buggy), date from eventDate. polymarket_odds.csv now populates 72/72;
triple_compare.py per-match join is live. Also linked all 72 fixtures from the index
(browsable fixtures section) — closes the unreachable-pages gap. 

**Session 35 — Last-minute polish + launch ✅ DONE — launched June 9, two days pre-kickoff. Daily tournament updates now run as a checklist, not a session — see §11.**
Do at launch-time (Tuesday's final Kaggle pull / June 11 morning), in one pass — all of it wants the freshest data. Run phase-by-phase with the planning chat; Phase 0 gates everything.

**Open the launch chat with the planning chat FIRST** ("ready for Session 35", + one line that you've pulled / are pulling the final Kaggle file and today is launch). Phase 0 has live go/no-go judgment (data cutoff, host-row alignment, whether title odds moved) that needs reading the outputs together before committing — don't run it solo and commit on the one morning it can't be redone.

**PHASE 0 — sportsbook fixes + fresh re-pull (gating; run at pull time).** Full work order below so it survives a fresh chat:
- *Fix 1 — host-match orientation in `fetch_odds.py`:* `parse_h2h_event()` writes p_home/p_away straight from the API with no orientation check; 3 host group-stage matches come back reversed vs fixture convention (Switzerland/Canada, Czech Republic/Mexico, Turkey/USA → fixtures are Canada/Mexico/USA home). Same fix validated for Polymarket in 35a: match each h2h event to a fixture on the unordered team-name pair, orient to the fixture's (home_team, away_team), swap p_home↔p_away to follow the team when orientation differs. Sportsbook h2h is 2-way (no draw) — just the two columns. Print each flipped row. Without this, triple_compare joins correctly-oriented Polymarket against reversed sportsbook on exactly the 3 host matches the launch hook leans on.
- *Fix 2 — defensive sport-key guard:* `list_world_cup_sports()` matches 6 keys; only men's `soccer_fifa_world_cup`/`_winner` return content today, but a next-cycle qualifier or club-WC key could start posting h2h mid-tournament and flow into `sportsbook_odds.csv` unfiltered (outrights already filters to `qualified`; h2h does not). After parsing h2h, drop any row whose team-pair isn't in `fixtures_2026.csv` (reuse Fix 1's matching). Report dropped rows, don't silently discard.
- *Then the real launch baseline (§6 "Launch baseline is provisional"):* download fresh martj42 `results.csv` → `data/raw/` (replace the March-31 one) → `python update.py` → take the new June baseline snapshot. Print the max match date in cleaned data + training row count — martj42 lags the maintainer's commits, so confirm what's actually in the file rather than assuming. "What changed today" must diff against this June baseline, not 2026-05-31.
- *Verify before committing (show the planning chat, then stop):* (1) the flip lines (expect 3 host matches); (2) any Fix-2 dropped rows (expect 0 today); (3) max match date + row count; (4) the 3 host rows in `triple_compare.csv` — confirm model p_home, sportsbook p_home, and p_home_poly ALL now carry the host's win prob (all three bars aligned); (5) title top-6 from the fresh snapshot (Spain/France stable → launch hook holds).

**PHASE 1+ (unlocked once Phase 0 commits; planning chat drafts off the real fresh numbers):**
- Re-run `src/fetch_polymarket.py` (fixed in 35a): per-match h2h markets populate `polymarket_odds.csv` 72/72 → third bar + per-match divergence layer light up with live launch-day numbers.
- **`docs/CNAME` survival check (NEW — domain gotcha):** setting the Custom domain in repo Settings → Pages writes a `docs/CNAME` file. `generate_site.py` copies `static/` → `docs/` and writes `docs/.nojekyll` on every build — confirm it does NOT wipe `docs/CNAME` when it regenerates (e.g. if it clears/overwrites `docs/`). If the rebuild clobbers CNAME, the custom domain silently unsets on the next `update.py` run and the site falls back to the github.io URL mid-tournament. Fix if needed: have `generate_site.py` write/preserve `docs/CNAME` (treat it like `.nojekyll` — an output the generator owns), so the domain survives every rebuild. Verify by running `python generate_site.py` after the domain is set and confirming `docs/CNAME` still contains the domain.
- **By-date schedule page (NEW — from pre-launch review):** the 35a fixtures section groups by group letter ("who's in Group H"); add a schedule grouped by date ("what's on today") — the natural daily entry point during the tournament. Pure site code, reuses 35a's match contexts / `DISPLAY_NAMES` / link targets, bucket by `date_iso` instead of group letter. Simple chronological list for now; grows a "today" anchor + today/upcoming/played split during the tournament.
- **Analytics (NEW — decided pre-launch):** add a privacy-friendly client-side analytics script (Cloudflare Web Analytics or GoatCounter — no cookie banner needed under GDPR, user is in Paris) to `base.html` `<head>` so it lands on every page. GitHub Pages gives no server logs; client-side is how we get visit counts / top pages / referrers (to see where Reddit/Twitter traffic lands). Skip Google Analytics (cookie-consent overhead, overkill).
- Fill the {TOKENS} in `launch_copy.md` with refreshed numbers (gated on the Phase 0 baseline).
- Build a standalone shareable launch graphic (PNG) — copy leans on "one striking visual" but the survival grid only lives inside the page, not as an exportable image. Wildcard on time; splittable if the session runs long. (✅ DONE in Session 35 — src/make_launch_graphic.py → docs/launch.png.)
- **Custom domain (DECIDED pre-launch — needed by launch day):** stay on GitHub Pages, point a custom domain at it (DNS record + Pages "Custom domain" setting + "Enforce HTTPS"). Static site → GitHub Pages serves it free, absorbs launch-day spikes, no deploy-flow change. Self-hosting on own server was considered (for real server logs) but deferred to post-launch — adds uptime responsibility + new deploy plumbing + TLS, all risky on the deadline; revisit as a tested migration after a successful Pages launch, keeping Pages as fallback. Buy + point the domain BEFORE posting widely (changing the URL after people share the github.io link fragments links). All site links are {{root}}-relative (verified Session 28) so the domain swap needs no code change.

**Close-out housekeeping:** record the host-orientation fix in §6 as a GENERAL triple-source concern — both `fetch_polymarket.py` (35a) and `fetch_odds.py` (35 Phase 0) now flip; any future per-match data source needs the same unordered-pair + swap-to-follow-team treatment, never trust a source's home/away slot.

---

**TOURNAMENT milestones (post-kickoff, NOT Session 35 — flagged pre-launch so they're not lost):**
- **Session 36 — outcome tracking / calibration goes live ✅ DONE (2026-06-12)** — ledger freezes market probs + actual scores; played-match pages render result + frozen forecast + verdict from the ledger; Today/Upcoming/Results schedule; index "Who's been closer" scoreboard. See §7. Note: the calibration page's raw→live-ledger switch turned out to be automatic (load_wc_predictions returns live data once any outcome exists — page flipped to "Live WC predictions" n=2 on June 12's first update); resolved same day with a MIN_LIVE_N=24 gate in generate_site.py — backtest seed renders until 24 scored ledger rows exist, then the page switches and shows the live n. Original scope: distinct from "divergence" (which means model-vs-market-vs-Polymarket disagreement *before* a match) — this is forecast-vs-reality (did what we predicted happen). Infrastructure exists: Session 27 calibration page + Session 29 `wc_predictions.csv` ledger (freezes each forecast pre-match, attaches result after). Open decision (§6): live ledger logs raw vs bias-corrected probs — doc recommends corrected as the honest "did our published forecast calibrate" measure. Per-match pages can also show "we said X%; result was Y". First real tournament-mode task. Scoped as **Session 36** (planning chat, first days of the tournament): decide what a played match's page shows — result scoreline, the three *frozen* pre-match bars read from the ledger (not recomputed probs), and a model-vs-market "who was closer" verdict (the public scoreboard hook from the launch plan). Also covers the calibration-page raw→live-ledger switch decision. Open question to verify first: what the pipeline currently produces for a played fixture's page (stays in fixtures_2026.csv / triple_compare? stale or recomputed bars?) — check after the first real match day before designing.
- **Exact-score view ✅ DONE (2026-06-14)** — top-3 scorelines + xG line (λ_H − λ_A) + 6×6 scoreline heatmap on match pages, read from the Dixon-Coles grid. triple_compare persists lambda_home/lambda_away/scoreline_grid (6×6, 6+ folded into the "5+" edge)/top_scorelines (top-3 from the full pre-fold grid); update_ledger freezes all four under the freeze-once guard (NaN for the 16 pre-existing rows — scoreline-less by design, source grids gone); generate_site renders the block as a pure consumer ([home][away] end-to-end, no transpose), played-from-ledger / upcoming-from-triple_compare per the Session 36 split, clean skip when absent. Raw-model caveat on the block + methodology line. First retrospective heatmaps appear once the next update.py freezes tomorrow's fixtures with grids. (Original scope text below retained.) **Original scope (planning chat, pure site/plumbing — do BEFORE Session 37, freeze window is perishable):** add top-3 most-likely scorelines + expected-goals line (λ_H — λ_A) + 6×6 scoreline heatmap to match pages, read from the existing Dixon-Coles scoreline grid (already computed in predict_from_lambdas; nothing persists it today). triple_compare.py persists top-N scorelines + λs as extra columns (generate_site.py stays a pure data-consumer — Session 23 convention); freeze them into wc_predictions.csv at forecast-freeze time so played-match pages keep the true pre-match scoreline picture retrospectively (freeze-once guard untouched; matches frozen before this ships can never be backfilled — divergence snapshots carry no scorelines, and post-hoc recomputes use post-match ratings). Raw-model caveat on the block + methodology page note: the scoreline grid is the uncorrected model, so its implied W/D/L won't exactly match the corrected outcome bars (~4pp on draws) — same disclosure pattern as the calibration page. No ledger scoring of scorelines, no "who's been closer" scoreboard entry (markets carry no goals info, no fair comparison). DECIDED: Claude does NOT issue a predicted score (forecasts stay non-AI). Stretch, after the visual block ships: feed top-3 scorelines + λs into the preview/divergence prompts as interpretable inputs only (prompt-version bump, expect an iteration cycle).
- **Share buttons + Open Graph cards ✅ DONE (2026-06-14)** — X / Reddit / Facebook share-intent links in the footer shell (all 76 pages), pre-fill the platform composer with the page's absolute canonical_url + URL-encoded `<title>`, no JS/SDK. OG + Twitter summary_large_image card block was already in base.html since launch (per-page og:url + meta_description, shared og:image = launch.png); share strip reuses canonical_url directly, no new generator logic. x.com/intent/tweet (skips the twitter.com 301). Parked follow-up (Option B): per-page generated OG image — a real PNG of each match's heatmap/forecast at build time so a shared match link shows that match's graph, not the shared launch.png. Expensive (headless render or matplotlib step + new build stage across 70+ pages); own session, sits next to the §5 standalone-launch-graphic wildcard.
- **Front-page redesign — divergence panel + today/results block ✅ DONE (2026-06-15)** — index is now a today-centric dashboard. Commit 1: top-divergences panel (top-5 by magnitude from triple_compare, always-populated, reuses flag_divergent + Session 22 category labels). Commit 2: results+today block (results = latest scored slate from the ledger with Session 36 "who was closer" verdicts; today = the next slate with model favourite + divergence flag), full 72-fixture listing dropped (schedule.html only), reordered to results+today → who's-been-closer → what-changed → top-divergences → survival grid. Commit 3: re-anchored the today/results split to the ledger play-frontier, not wall-clock (see §6 + §7). **NEXT: Session 38** (survival grid → populated knockout bracket — see the standalone entry below). (Original scope below retained.) **Original scope (Session 37, planning chat, anytime — pure site code):** two related index changes sharing the same files, done together. (1) **Current top divergences panel** — replace the index's 72-fixture listing (redundant now schedule.html carries the full browsable list) with the steady, always-populated divergence signal, sorted by magnitude from today's triple_compare.csv, alongside the existing movers. Open decision: does "fresh divergences" (newly-flagged vs yesterday) stay as a secondary item or get replaced? Recommendation: keep both, current-top primary. (2) **Time-anchored "results + today" block (NEW — from this request)** — a compact front-page slice so a daily visitor sees what just happened and what's on without clicking through to schedule.html: most-recent-completed-match-day results (final score + the Session 36 "who was closer" verdict) and today's / next match-day's fixtures (model favorite + divergence flag, linked to the full match pages). This block IS the replacement for the 72-fixture dump — resolves the Session 36↔37 tension (36 restored the full list, 37 removes it): the index stops being a full fixture list and becomes a today-centric dashboard, with the complete list staying on schedule.html. Data sourcing follows the §6 hard rule: played rows from the ledger (wc_predictions.csv), never triple_compare; upcoming from triple_compare / fixtures_2026.csv. "Most recent match day" = latest date with attached results, not literally yesterday (rest days exist — group→R32 gap ~June 24–27); "today" falls back to the next scheduled match day when nothing is on. All date logic through clock.today() (respects WC_ASOF_DATE so the dry run / as-of testing still works). Rest-day empty state: "No matches today — next up {date}." Real open decision is index hierarchy, not the block: the index now carries survival grid + "What changed today" + "Who's been closer" + top-divergences + results/today — recommendation is the time-anchored block near the top with the survival grid dropping down, but decide deliberately when running the session. Splittable into 37a (divergence panel) / 37b (results+today block) if it runs long, per the 35/35a precedent. Touches whats_changed.py / generate_site.py / index template only — no model changes.

**Pre-Session-38 display-polish sessions (tournament rest-day work, ~June 16–26 — DISPLAY LAYER ONLY).**

Self-contained sessions runnable on rest days between match days, in priority order. Firewall rule (non-negotiable, from the June-27 freeze): none of these may touch the model, sim, triple_compare, ledger, or update pipeline. All work is in the display/render layer where a mistake can't corrupt data or block a publish. Open a planning chat per session ("ready for Session OG", etc.); each produces its own Claude Code work order.

- **Session OG — per-page Open Graph images ✅ DONE.** At build time, render a per-match PNG of that match's forecast and point each match page's og:image at its own PNG instead of the shared launch.png, so a shared match link previews that match's graph. Reuses src/make_launch_graphic.py (extend the existing matplotlib PNG path — do NOT add a headless-browser dep), the per-page OG meta block in base.html (per-page og:url + meta_description already present since launch; only og:image flips from constant to per-page), the Session 36 played-from-ledger / upcoming-from-triple_compare split, and DISPLAY_NAMES. Scope: match pages only for v1 (index/schedule keep launch.png). Content: played → final score + frozen pre-match bars; upcoming → three corrected bars + heatmap. Hash-cache per card (like the preview cache) so daily builds only re-render changed matches. Hard guardrail: OG render is SOFT — a failed card falls back to launch.png and the site still publishes; OG generation must never abort generate_site.py or update.py. This is the firewall in miniature, do not soften it.
- **Session GRID-SORT — survival-grid sort cascade ✅ DONE.** Sort the index survival grid by round depth: P(champion) → P(final) → P(SF) → P(QF) → P(R16) → P(advance), champion key rounded to display precision (decision B — teams tied on printed title odds break by projected depth), alphabetical final tiebreak. Pure display: a sort key in generate_site.py reading per-round probs already in the newest MC snapshot. Top of the grid unchanged; 0%-champion tail now ordered by depth instead of alphabetical. No model/snapshot change.
- **Session 38-RECON — bracket reconnaissance (~June 24–25, READ-ONLY, NOT a build).** Before the groups finish, confirm what bracket.py's resolve_third_place_slots() + the R32_BRACKET encoding actually output — shape, types, slot-id strings — so Session 38's June-27 bracket build wires known data into a known shape instead of discovering the data layer under deadline. Deliverable is a written confirmation (a throwaway print script is fine; nothing committed to the pipeline). Slots are genuinely unknown until results exist, so this is pre-reading, not pre-building. De-risks the single hardest remaining session.

Parked (NOT scheduled here, deferred past the bracket per the firewall): feeding top-3 scorelines + λs into the preview/divergence prompts (the exact-score stretch) — it's the one open item that pokes the AI SOFT stages, so it waits until after Session 38.

**Survival grid → populated knockout bracket (~June 27, group stage ends) (Session 38 — NEXT):** index is a survival grid pre-tournament because R32 slots (1A, 3CEFHI…) can't be filled until groups resolve (§6). Once the group stage finishes, draw the real bracket — it becomes the natural hero view. Don't pre-build (slots unknown until results exist).

---

## 6. Open questions and known risks

- **Recent data gap:** ✅ Resolved in Session 3. Kaggle dataset is actively maintained, current through March 31, 2026 (last FIFA international break before WC). Fresh weekly pulls during the tournament should be sufficient.
- **API rate limits during tournament:** if we hit Odds API or API-Football limits in June, may need to pay ~$25/month. Budget for this.
- **Polymarket liquidity for some markets:** smaller-team matches may have thin Polymarket markets that produce noisy probabilities. Need to flag low-liquidity warnings.
- **Claude API costs:** with 104 matches and re-generation after each, costs could add up. Cache aggressively, only regenerate when inputs actually change.
- **Calibration before tournament:** with no live results, the calibration page will be empty for week 1. Plan: backfill with backtest results from Euro 2024 to demonstrate calibration even before WC starts.
- **Confederation-pool drift in Elo:**  ✅ Mostly resolved in Session 7 by seeding Jan 2018 ratings from eloratings.net (101 teams). Top 5 WC ratings now match public Elo order. Residual drift remains for some CAF/AFC/CONCACAF teams (Morocco, Japan, Mexico) — this is a structural limit of pure Elo with sparse inter-confederation data and won't be eliminated without architectural changes. Acceptable for v1; revisit if Dixon-Coles backtest in Week 2 shows it materially hurts predictions.
- **Model is biased** ~4pp away from draws toward away wins. Confirmed by 5-bin reliability check on 290 backtest matches. Acceptable for v1, but the divergence detector in Week 4 should not flag a 4–5pp gap on draw probabilities as a real divergence — it's our known bias. Possibly traceable to the small fitted ρ (−0.027 vs literature norm of −0.10 to −0.15); a future refinement is to fit ρ on competitive matches only.
- **Calibration page shows raw model, published per-match bars show bias-corrected:** The calibration tracker (Session 27) is seeded from backtest_2024.csv, which holds uncorrected Dixon-Coles probabilities, while the per-match pages (Session 25) publish probs with the ~4pp draw/away bias correction applied (offsets from §6, read live from the backtest). So the reliability diagram describes the raw model, not exactly the numbers a reader sees on a match page — the correction is precisely what closes the draw under-prediction the diagram displays. Pre-tournament the page is labeled as backtest; once wc_predictions.csv fills in June (Session 29), decide whether to log raw or corrected probs there — corrected is the more honest "did our published forecast calibrate" measure and is the recommended choice. Disclose both on the methodology page (Session 31). ✅ RESOLVED: the live ledger logs corrected probs (implemented that way in Session 29), and the page's backtest→live switch is gated at MIN_LIVE_N=24 scored predictions (Session 36 follow-up) — no open decision remains here.
- **Host advantage:** ✅ Resolved in Session 33. The 9 host-team home group-stage matches (USA/Mexico/Canada ×3 each) run neutral=False with a 60-Elo home bump in simulate.py/monte_carlo.py; triple_compare.py sets USE_FIXTURE_NEUTRAL=True to match and drops host-exclusion (flags 14→26). USA advance 67.6→79.3%, Mexico +1.6pp, Canada +1.2pp. Host matches still flag as model-over-confident vs market (expected). Knockout stays neutral (host venues not guaranteed). Match pages read the neutral column for a "HOME VENUE — {host}" label and drop the old "not modeled in v1" caveat.
- **Model is more concentrated at the top than markets. Confirmed in Session 17 against Polymarket VWAP:** model top-2 (Spain + Argentina) = 50% of title probability vs market top-2 = 34%. Spain at 30% (market 16.5%), Argentina at 20% (market 8.6%). This is structural: pure Elo + Dixon-Coles with no injury/form/squad data assumes peak fitness every game, and small per-match over-confidence on heavy favorites compounds across the 7-match title path. The Week 4 divergence detector should treat large model > market gaps on title odds for the very top of the table as expected behavior to surface, not as candidates for "the model thinks it knows something." For per-match divergences during the tournament, the existing detector logic (offset for the ~4pp away-win bias from §6) is still the right approach — this top-tier concentration is a title-odds-specific phenomenon, not a per-match calibration issue.
- **Sportsbook coverage on May 17:** The Odds API returned all 50 group-stage h2h markets (plenty of books) and a populated outright winner market. We're not waiting on data availability for Week 4; we can build against real data from day one.
- **Polymarket per-match coverage:** As of 2026-05-17, Polymarket has not posted per-match h2h markets for the WC. Their coverage today is 1 title event + 12 group-winner events + 7 misc futures. Per-match markets typically appear close to kickoff and per-match-day during the tournament. src/fetch_polymarket.py writes a header-only polymarket_odds.csv so Session 20's divergence detector has a stable join schema; re-running the fetcher during the tournament will populate it. Liquidity will vary — smaller-team matches will need a low-liquidity flag when they arrive.
- **Per-match divergence-type breakdown** (Session 20 baseline, 2026-05-18): Across the 50 group-stage matches with current sportsbook coverage, model-corr-vs-book splits 25 over-concentrated / 22 under-concentrated / 3 disagree-on-favorite. The near-balanced over/under split is the per-match analog of the title-odds concentration finding above — when model and book pick the same winner, the model is roughly as often too confident as not confident enough. True disagreements on the favored team are rare (3 of 50). Sessions 21–22 will use divergence_type as a branching input so Claude commentary fits the gap shape (e.g. "the model trusts the Elo edge more than the market" vs "the market sees something Elo can't"). The 15pp flag threshold was set on signal density at this coverage level; revisit if late-posted matches change the distribution.
- **AI commentary occasionally produces imprecise comparatives.** Session 22's prompt v2 fixed structural failures (fabricated combined gaps, decimal percentages, off-list categories like tactics). Residual: vague comparisons like "roughly half" can be quantitatively off by a non-trivial factor even when the qualitative direction is right, and the categories rule occasionally leaks ("momentum"). Acceptable for v1 — the numbers themselves are right there in the data block next to the prose. Methodology page should disclose that commentary is generated by Claude and is not a substitute for reading the actual probabilities.
- **Tournament-tree page is a survival grid pre-tournament, not a drawn bracket.** R32 slots (1A, 3CEFHI, …) can't be filled until group results exist, so the index page shows per-round survival probabilities color-coded by tier. A populated knockout bracket becomes meaningful once the group stage resolves slots — revisit as an added view during the tournament.
- ✅ RESOLVED in Session 37 (commit a9b7a8c reordered to results+today → who's-been-closer → what-changed → top-divergences → survival grid; survival grid dropped to the bottom as recommended) — **Index page is becoming multi-panel — hierarchy is an unmade decision (flagged for Session 37):** the front page now stacks survival grid + "What changed today" + "Who's been closer" scoreboard, and Session 37 adds a top-divergences panel and a time-anchored results/today block. That's five competing claims on the top of the page. During the tournament the time-anchored "what just happened / what's on today" content is the most-wanted-first; recommendation is to lead with it and drop the survival grid down. Decide deliberately in Session 37 — this is the session's real open question, not the panels themselves. Note the Session 36↔37 tension: Session 36 restored the index's full 72-fixture listing (it had silently vanished when clean_data moved played matches out), but Session 37 deliberately removes that full list in favor of the today-centric block — the complete browsable list lives on schedule.html, so removal is safe.
- **Index today/results block anchors to the ledger play-frontier, not wall-clock (Session 37):** results half = latest date_iso with results attached in wc_predictions.csv; today half = earliest date_iso with unplayed fixtures (also produces the rest-day "next up {date}" state for free). Deliberately not clock.today() for bucketing — the once-daily morning build runs in Europe while North America is still on the previous tournament day, so a viewer-local date splits a single tournament slate across two of the operator's calendar dates and rolls onto the next card at Paris-midnight. The today half originally (commit 2, a9b7a8c) bucketed on clock.today(); commit 3 (92b1943) re-anchored it to the frontier, verified clock-independent across three WC_ASOF_DATE values. clock.today() now survives for ONE cosmetic thing only — the "Today" vs "next up {date}" label — never for which matches land in which half (so the code still calls clock.today(), but only for that label; don't be misled into thinking bucketing is clock-driven). schedule.html (Session 36) buckets the same way. Don't "fix" this back to the clock. Mid-slate rule (won't trigger in the once-daily morning rhythm): today's slate stays whole in the today block, already-scored matches show scores inline rather than moving to results.
- **"Who's been closer" verdict was logically broken until June 15 (Session 37 follow-up):** the original _verdict ranked sources by their probability on the realized outcome and called the top two a wash when within 2pp — without checking whether any source actually forecast the outcome. So it slapped "too close to call" on two opposite situations: everyone-right-and-bunched (Germany 7–1 Curaçao, all sources ~92% on the home win) AND everyone-wrong-and-bunched (Côte d'Ivoire 1–0 Ecuador — all three favoured Ecuador/draw, best home-win prob was books' 26.5%, the upset came in and was mislabelled "a wash"). Because books and Polymarket track each other and sit within 2pp constantly, the wash branch fired on nearly every match — structurally biased toward "wash" regardless of forecast quality, and silently miscrediting the markets on every upset (which would have skewed the launch's headline "model vs market" scoreboard). Fixed with three ordered branches: (1) all sources missed — no source's own H/D/A argmax equals the realized outcome (the upset case, intercepted before any wash logic); (2) sources agreed — outcome was at least one source's top pick AND the top two on-outcome probs are within 2pp (now only ever means everyone-right-and-bunched); (3) {source} closest — one source separated by >2pp on the right outcome. Scoreboard credits ONLY a distinct {source} winner; "all missed" and "sources agreed" credit nobody, so "model N" now genuinely means "the model distinctly beat the markets N times." Key implementation note: "gave the outcome its highest probability" = that source's own argmax across its three probs, NOT the highest among sources on the outcome — requires reading all three of each source's frozen probs, not just the outcome column.
- **"What changed today" panel and divergence-snapshot mechanism (Session 26):** `triple_compare.py` now writes a dated divergence snapshot (`data/processed/divergence_snapshots/YYYY-MM-DD.csv`) after each run, mirroring the MC snapshot pattern. `src/whats_changed.py` diffs the two newest files in each snapshot dir: title-odds movers use `p_champion`, group-advance movers use `p_advance`, fresh divergences = `flag_divergent==True` in curr but not in prev. `MIN_MOVE_PP = 0.5` suppresses MC-noise jitter (≈±0.5pp). Panel shows a pre-tournament baseline message until live results shift probabilities; fresh divergences start accumulating as new h2h markets are posted closer to kickoff.
- ✅ RESOLVED in Session 30 — **Simulation is not yet result-aware (as of Session 29):** update.py orchestrates the full pipeline, but simulate.py still plays all 104 matches from the group stage every run (Session 14 behavior). Correct pre-tournament; the moment June 11 arrives the sim must fix played results and simulate only the remainder, or the evolving forecast is wrong. Session 30's Euro-2024 dry run is where this gets forced and hardened — it's the single biggest remaining piece before the tournament. Done: simulate.py now pins played group results by scoreline and knockout results by advancing team; dry_run.py replays a full tournament and all invariants pass.
- ✅ RESOLVED in Session 32 (WC_CLEAN_OUTPUT makes clean_data's output path overridable; dry run isolates to a temp file) — **Dry run contaminates the real matches_clean.csv (Session 30):** dry_run.py overrides clean_data's *input* (WC_MANUAL_RESULTS → temp feed) but clean_data's *output path is not overridable* — it always writes the real data/processed/matches_clean.csv. So a replay leaves 104 fake WC results in matches_clean.csv, and any hand-run of monte_carlo.py afterward pins the entire fake tournament (observed: Portugal 100% champion). Recovery is a single `python src/clean_data.py` because the real wc_results_manual.csv is empty — clean rebuilds from results.csv + the clean feed and the fakes wash out. PROPER FIX (do before relying on the harness again, and certainly before June): either make clean_data's output path overridable so the dry run writes to a temp matches_clean.csv, or have dry_run.py rebuild matches_clean.csv from the real feed on exit. The harness claims "Ledger restored, temp feed deleted cleanly" but does NOT restore matches_clean.csv — that's the gap.
- **Manual-feed schema must be carried through clean_data explicitly (Session 30):** the historical results.csv and the hand-maintained wc_results_manual.csv have *different schemas*. The new `advanced` column was silently dropped by clean_data until fixed (always-concat, historical rows → NaN). Any future manual-feed column (referee, venue, etc.) hits the same trap: it must be explicitly added to clean_data's output or it vanishes with no error. Always re-check the matches_clean.csv header after adding a feed column.
- Second trap found June 11: a sparse manual row (prefilled placeholder) was winning the dedupe *whole*, blanking fixture metadata in fixtures_2026.csv. clean_data now **column-merges**: manual values win where present (scores, `advanced`), blank manual fields fill from the matching raw row (unordered pair + date); rows with no raw counterpart (future KO rows) pass through unchanged.
- ✅ RESOLVED in Session 32 (WC_SNAPSHOT_DIR / WC_DIVERGENCE_SNAPSHOT_DIR isolate the harness to temp dirs) — **Dry run writes June/July snapshots into the real snapshots dir (Session 30):** "clear stale snapshots at startup" operates on the real data/processed/snapshots/. May snapshots survive and sort before June by filename, but the dry run's synthetic-date June/July snapshots are left behind as residue and must be deleted after a run (anything dated June/July 2026 pre-tournament is fake). A WC_SNAPSHOT_DIR override to isolate the harness to a temp dir is the clean fix — done in Session 32.
- **generate_site.py NaN calibration-bin crash:** ✅ Resolved in Session 33. `_cal_svg` uses `pd.isna()` to skip empty/NaN reliability bins before `round()`, so the site keeps building through the first live match-days.
- **Launch baseline is provisional:** The 2026-05-31 snapshot was taken on March-31 data (Kaggle martj42 not yet refreshed past the last international break). The real launch baseline is the early-June re-pull (fold into Session 35 / launch morning): download fresh martj42 results.csv → data/raw/ → run update.py → take a new baseline. No code changes — Session 33 did the structural work. "What changed today" should diff against the June baseline, not 2026-05-31.
- ✅ RESOLVED in Session 35a — **Polymarket fetcher returns empty across ALL market types (2026-06-05):** slug drift not an outage; new slugs + hybrid discovery; per-match markets now exist for all 72 group fixtures (series 11433); fetcher populates polymarket_odds.csv 72/72.
- ✅ RESOLVED in Session 35a — **Per-match pages exist but are unreachable unless flagged (2026-06-05):** new browsable fixtures section on the index links all 72 match pages.
- **Polymarket ordering and abbreviation are both unreliable (Session 35a):** abbreviation had Curaçao as `kor`; ordering disagrees with FIFA home/away on host matches. The fetcher matches on the unordered team-name pair and orients to the fixture's convention, swapping p_home↔p_away to follow the team (p_draw untouched). Any future code reading Polymarket per-match data must do the same — never trust their home/away slot.
- ✅ RESOLVED in Session 35 (Phase 0) — **Sportsbook host-match orientation (fetch_odds.py):** parse_h2h_event wrote p_home/p_away straight from the API; 3 host group matches came back reversed vs fixture convention. Fixed with the same unordered-pair match + swap-to-follow-team approach validated for Polymarket in 35a (FLIPPED: Switzerland/Canada, Czech Republic/Mexico, Turkey/USA). Fix 2 (drop any h2h pair not in fixtures_2026.csv) added as a guard; 0 dropped. GENERAL RULE now holds for both per-match sources — never trust a source's home/away slot; match on the unordered pair and orient to the fixture.
  — **and never trust a source to send each match exactly once.** June-11 incident: the Polymarket series (soccer-fifwc/11433) carried two events per fixture — one real traded market and one dead zero-volume duplicate (uniform ~33/33/33 prices); the fetcher saved 144 rows, triple_compare joined 144, and **the site published doubled listings on the morning of June 11** (hand-deduped + rebuilt as a hotfix). Fix (same evening): `fetch_polymarket.py` dedupes on the unordered pair **keeping the max-volume event** (zero-volume survivor → LOW-LIQUIDITY warning — the §6 thin-market risk, now detected automatically); `fetch_odds.py` same dedupe defensively; `triple_compare.py` dedupes on load and **hard-fails unless its row count equals the remaining unplayed fixture count** (FATAL stage — blocks a doubled site from ever publishing again; the assert is dynamic, not a fixed 72 — it was 72 on June 11 because nothing had been played yet, shrinks as matches play, and refills when knockout fixtures enter via the manual feed). Ledger freeze guard: a fixture already frozen in wc_predictions.csv is never re-frozen. Dead events may persist in the series, so dropped-row prints on Polymarket fetches are expected, not a fault.
- ✅ RESOLVED in Session 35 (Phase 0) — **docs/CNAME survival:** generate_site.py now writes docs/CNAME (CUSTOM_DOMAIN = worldcup.divergencelog.com) on every build, like .nojekyll, so the custom domain can't silently unset on rebuild. Verified CNAME survives both update.py and generate_site.py.
- **Launch baseline is now June, not provisional (Session 35):** fresh martj42 re-pull loaded results through 2026-06-08 (8,080 training matches, was 7,952); new 2026-06-09 baseline snapshot. Title top-6: Spain 28.2 / Argentina 19.7 / France 10.2 / England 5.6 / Portugal 4.5 / Brazil 4.0. Launch hook is Spain ~28% model vs ~15.5% market (~13pp), down from the old "30 vs 16." France flipped to model-under-market (book 14.5%) — secondary talking point, not the hook. "What changed today" diffs against this 2026-06-09 baseline.
- **(2026-06-11):** Tournament day 1 — morning incident + evening hardening batch. Polymarket series 11433 carried a dead zero-volume duplicate event per fixture → 144-row triple_compare → doubled site listings published; hand-hotfixed, then guards landed pre-kickoff: max-volume dedupe + LOW-LIQUIDITY warning in fetch_polymarket.py (verified live: 72/72, USA–Paraguay keeps real prices not the 33/33/33 shell), defensive dedupe in fetch_odds.py, 72-row FATAL assert in triple_compare, freeze-once-per-fixture ledger guard. Prefilled wc_results_manual.csv with all 72 group fixtures (new src/prefill_manual_results.py, idempotent); surfaced and fixed a clean_data dedupe bug (sparse manual rows blanked fixture metadata → column-merge, manual-over-raw). Docs: §11 reworded (scores-only entry, 72-row check, expected drop-prints), Session 36 (outcome tracking) scoped. Verified: 8,107 training rows (June-10 Kaggle), 72 clean fixtures, baseline holds (Spain 27.4 / Argentina 20.4 / France 9.4). June-12 update = guards' first full live validation.
- **Played fixtures leave the live pipeline by design — the frozen ledger is their single source (Session 36):** clean_data moves scored matches into matches_clean.csv, so fixtures_2026.csv and triple_compare.csv shrink as the tournament progresses (the triple_compare FATAL assert is dynamic, `len == remaining fixtures`, so this passes — June 12's "72-row" wording in §11 was already stale). Before Session 36 this silently dropped played match pages, their schedule/index links and sitemap entries. Now `wc_predictions.csv` freezes book/Polymarket probs + poly_volume + neutral_used alongside the model probs at freeze time (freeze-once guard untouched; absent markets freeze as NaN), attach_results also records the actual scoreline, and generate_site renders played matches entirely from the ledger — played status is determined ONLY by a ledger row with a result attached, never inferred from fixture dates or from presence/absence in triple_compare. **Any future consumer of played-match data must read the ledger, never triple_compare.** Rows frozen before Session 36 were backfilled from the freeze-day divergence snapshots (one-time src/backfill_ledger_markets.py, idempotent).
- **Scoreline heatmap is display-only — folded W/D/L ≠ corrected bars, by two separate mechanisms (exact-score view, 2026-06-14):** the match-page heatmap/top-3 come straight from the uncorrected Dixon-Coles grid, so their implied result split won't match the bias-corrected W/D/L bars above them — disclosed on the block and methodology page (same register as the calibration raw-vs-corrected note). Two distinct gaps fold into that one caveat: (1) the ~4pp draw/away bias correction, applied to the published bars but never to the grid; (2) a corner-cell artifact — folding the 11×11 grid to 6×6 collapses all 6+/6+ outcomes into the single g6[5][5] diagonal cell, reclassifying that off-diagonal mass as draws. (2) is negligible on real fixtures (λ ≈ 0.4–2.7 → ~6e-06 W/D/L drift on Germany–Curaçao) but can move the heatmap's visual draw rate several pp on a hypothetical goal-fest (synthetic λ 3.8 vs 3.2 → ~5pp). Harmless because the bars are never computed from the folded grid — the heatmap is purely a display. The fold itself is loss-free for total mass (sums to 1.0); only the W/D/L classification of the corner shifts. The 16 ledger rows frozen before this change stay scoreline-less permanently (no backfill — source grids gone, recompute would use post-match ratings).
- **Shared-card cache is X-side and unpurgeable (Session OG):** X caches OG cards per-URL for ~7 days and has removed its Card Validator — there is no manual purge. The compose-box preview is independently sticky and ignores reused ?v= params, so it will show a pre-OG launch.png long after the real card is correct — judge by a third-party OG debugger or the actually-posted tweet, never the composer. Per-page OG is verified correct server-side; to force a fresh card on an already-shared URL use a new query param. New URLs (knockout pages) card correctly on first share. Not a code issue — don't chase it in the build.
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
- **Session 35a (2026-06-07):** Fixed the Polymarket fetcher (dead since the May restructure). Slug swap restored title (world-cup-winner, 48 teams, Spain/France 15.7% top) + groups; new per-match probe under series soccer-fifwc/11433 populates polymarket_odds.csv 72/72 — three binary child markets per event bucketed to W/D/L, teams from structured event["teams"] (name+ordering; abbreviation is buggy: Curaçao shows kor), date from eventDate, host-match orientation flipped to fixture convention on 3 matches (Canada/Mexico/USA, probs follow the team, p_draw untouched). triple_compare join live, no longer a no-op. Linked all 72 fixtures from the index. Two code commits + doc commit.
- **Session 35 (2026-06-09):** Launch build + launched, two days pre-kickoff (WC starts June 11). Phase 0 (gating): fetch_odds.py host-orientation fix (FLIPPED Switzerland/Canada, Czech Republic/Mexico, Turkey/USA) + non-fixture h2h guard (0 dropped); generate_site.py CNAME-survival write; fresh Kaggle re-pull through 2026-06-08 (8,080 training matches) → new 2026-06-09 baseline. All 5 go/no-go checks passed; Polymarket per-match live 72/72 from the update.py run; all three title bars aligned on hosts. Phase 1: by-date schedule.html + Schedule nav; GoatCounter analytics in base.html (worldcup-divergencelog, all 76 pages); re-runnable src/make_launch_graphic.py → docs/launch.png. Custom domain worldcup.divergencelog.com live, HTTPS enforced. Launched evening of June 9 on the Spain ~28% model vs ~15.5% market (~13pp) hook. June 11 = first market refresh; tournament-mode daily updates begin once results post.
- **Session 36 (2026-06-12):** Outcome tracking live. Found the played-fixture gap first: clean_data moves scored matches out of the live pipeline, so the two June-11 match pages (and their schedule/index/sitemap links) had silently vanished from the site. Commit 1: update_ledger.py freezes p_*_book / p_*_poly / poly_volume / neutral_used alongside the corrected model probs (freeze-once untouched, NaN for absent markets, _ensure_schema header-extends old ledgers), attach_results also records actual scorelines idempotently; one-time src/backfill_ledger_markets.py filled all 8 pre-existing rows from the freeze-day divergence snapshots (verified vs 2026-06-10.csv; re-run = no-op). Commit 2: generate_site renders played matches purely from the ledger — final-score header, the three FROZEN pre-match bars, a "who was closer" verdict (highest frozen prob on the actual outcome; top-2 within 2pp = wash; <2 sources = no verdict) reused by a new index "Who's been closer" scoreboard (after 2 matches: model 2 — model beat both markets by 14pp/10pp on the day-1 home wins); schedule.html split Today/Upcoming/Results; index fixtures list back to all 72 with scores; unplayed pages verified byte-identical. Observation: calibration page auto-flipped to "Live WC predictions" (n=2) on June 12's morning update — pre-existing Session 27/29 behavior, §11 wanted that switch deliberate. Follow-up same day: live-switch gated on MIN_LIVE_N=24 scored ledger rows in generate_site.py (backtest seed until then, with a "{live_n} of 24 so far" note; live n shown in the heading once switched). Three commits: b7b82e1, 0d80b3c, 585e5d9.
- **Exact-score view (2026-06-14):** Added the scoreline view as a pre-Session-37 milestone. predict_from_lambdas already surfaced the full 11×11 post-τ grid + λs in its return dict (zero model-layer edit). triple_compare.py persists lambda_home/away, a 6×6 "5+"-folded scoreline_grid, and top_scorelines (top-3 from the full pre-fold grid — true modes, no corner artifact); fold caught an off-by-one in review (folding from index 5 re-added the exactly-5 row → grid summed 1.08; fixed to strict 6+ tail, sums to 1.0). update_ledger.py freezes all four verbatim (JSON strings opaque, lambdas rounded) under the freeze-once guard; _ensure_schema extends old ledgers, the 16 pre-existing rows stay scoreline-less by design. generate_site.py renders a hand-built inline-SVG heatmap ([home][away], no transpose, gold diagonal = draws, away=0 column brightest on Germany–Curaçao confirming orientation) + xG line + top-3, played-from-ledger / upcoming-from-triple_compare, clean skip when absent. Raw-model caveat on block + methodology page (covers both the bias correction and the corner-cell fold). No Claude-issued score; no ledger scoring of scorelines; no scoreboard entry. Files: triple_compare.py, update_ledger.py, generate_site.py, templates/match.html, templates/methodology.html, static/style.css.
- **Share buttons + OG cards (2026-06-14):** Added X/Reddit/Facebook share-intent links to the base.html footer (all 76 pages) — plain anchors, no JS, pre-fill from canonical_url + URL-encoded title; verified encoding on the accented Curaçao page (Cura%C3%A7ao, literal &→%26, separator written &amp;). OG/Twitter card block found already present since launch (kept per-page meta_description over a constant tagline — strictly better shared cards). x.com intent host (skips twitter.com 301); CNAME survives rebuild; launch.png 1200×675 renders large card with negligible letterbox. Per-page OG image (Option B) parked as a follow-up. Files: templates/base.html, static/style.css. Separate commit from the scoreline work.
- **Session 37 (2026-06-15):** Front-page redesign, pure site code. Commit 1 (7c0ec90): current-top-divergences panel on index (top-5 by magnitude from triple_compare, always-populated unlike fresh-divergences, reuses flag_divergent threshold + Session 22 category labels). Commit 2 (a9b7a8c): results+today block (most-recent scored slate from the ledger with "who was closer" verdicts; today's card with model favourite + divergence flag), dropped the full 72-fixture index listing (now schedule.html only), reordered index to results+today → who's-been-closer → what-changed → top-divergences → survival grid. Commit 3 (92b1943): re-anchored the today/results split from clock.today() to the ledger play-frontier (Europe morning builds run on the previous North-American tournament day; viewer-local dates mis-bucketed slates). Verified live: June 14 results / June 15 today split correctly.
- **Session 37 follow-up — verdict logic fix (2026-06-15, commit aff1f59):** Eyeballing the live results block surfaced that "who was closer" mislabelled the Côte d'Ivoire upset and the Germany blowout identically as "too close to call." Diagnosed as a logic bug (the wash branch ignored whether any source forecast the outcome — see §6), not wording. Rewrote _verdict to all-missed / sources-agreed / {source}-closest; scoreboard credits only distinct {source} wins. Self-tested (all-missed, wash-correct, clear-winner, <2-sources; `python generate_site.py --test`). Live verification: Côte d'Ivoire → "all sources missed", Germany → "sources agreed"; index scoreboard now model 2 after re-tally (was inflated by upset miscredits). Built + pushed same evening.
- **Session OG (2026-06-15):** Per-match Open Graph images, display-layer only (firewall held). New src/make_og_cards.py — render_og_card(m, out_path)->bool, SOFT (never raises, falls back to launch.png on any failure), imports the palette from make_launch_graphic (single source; main() untouched, constants only). generate_site.py renders a card per match page → docs/og/<key>.png and points that page's og:image/twitter:image at it on success; index/schedule/calibration/methodology keep launch.png (base.html already threaded {{ og_image }} — no template change). Upcoming card = three corrected W/D/L bars + xG line + most-likely score; played = final score + frozen pre-match bars with ✓ on the actual outcome; both 1200×675, warm-paper. SHA1 hash-cache over CARD_VERSION + played-state + score + 1pp probs; upcoming→played flip auto-invalidates. A follow-up moved the .sha1 cache sidecars out of the served tree (docs/og/ now PNG-only) into data/processed/og_cache/, gitignored. SOFT deliberate-break verified (build exits 0, falls back, logs [og] WARNING); cache-skip verified on no-data rebuild. Per-page OG verified live via third-party OG debugger (Belgium–Egypt rendered its own card). Commits: b5a9823 — both the per-match OG and the .sha1 cache relocation landed together in this single commit (not two; planning chat assumed separate hashes), e9ae969 was the subsequent site rebuild.
- **WhatsApp share button (2026-06-15):** Added a fourth share-intent anchor (WhatsApp) to the base.html footer strip alongside X/Reddit/Facebook — plain https://wa.me/?text=... link, no JS/SDK, title + absolute canonical_url combined in the single text= field (wa.me universal form: app on mobile, Web on desktop), same encoding as the existing buttons; WhatsApp pulls the per-page OG card (Session OG) into the preview. Encoding verified on the accented Curaçao page. Commit: 7845842.
- **Session GRID-SORT (2026-06-15):** Round-depth sort cascade for the index survival grid, display-layer only (firewall held). generate_site.py now orders the grid by a tuple sort over P(champion)→final→SF→QF→R16→advance with an alphabetical-on-display-name final tiebreak; the champion key is rounded to the grid's display precision (decision B) so teams showing the same title % fall through to the depth cascade, while the deeper keys stay raw floats. Reads the six per-round probs already in the newest MC snapshot — no model/sim/snapshot/pipeline change. Verified on the rebuilt docs/index.html: contender top byte-identical (Spain/Argentina/France unchanged), 0%-champion tail now orders by projected depth instead of alphabetical. Commit ad02f7b.
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

Close-out edits to PROJECT.md are made by Claude Code in-place, not hand-pasted by the user — the planning chat supplies the exact wording, Claude Code applies it, the user reviews the diff before pushing.

When a close-out fact depends on what landed on disk (swap vs no-op, which branch, commit hashes), Claude Code writes that part of the §7 line — the planning chat doesn't infer repo state it can't see.

The operator's live-site eyeball is a real checkpoint, not a formality (added Session 37). Two bugs this session were caught only because the user looked at the rendered site and pushed back, not by any test or the loop itself: the today-half wall-clock mis-bucketing, and the verdict logic mislabelling the Côte d'Ivoire upset as "too close to call." Both passed their self-tests — the tests confirmed the code did what it was written to do, not that what it was written to do was right. Corollary rule: Claude Code owns facts about repo state the planning chat can't see (swap vs no-op, which branch, commit hashes, post-rebuild numbers). The planning chat must not infer or default them — earlier this session "don't know → assume confirm" produced a false record that Claude Code had to correct from the git history. When a close-out fact depends on what landed on disk, Claude Code writes that part; the planning chat supplies wording and defers on facts.

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

---

## 11. Tournament-mode daily update (operational checklist, NOT a session)

Once matches are being played, the daily refresh is operational, not a build. It does NOT use the plan→work-order→close loop. Run it yourself in Claude Code as the checklist below. Only open a planning chat for a real decision or a break — the one known upcoming decision is the survival-grid→drawn-bracket redraw (~June 27, when the group stage resolves R32 slots). (The calibration-page switch was resolved in Session 36: automatic at MIN_LIVE_N=24 — see Notes below.)

### After each match day

1. **Fill in scores on the prefilled row in `data/raw/wc_results_manual.csv`.** All 72 group fixtures are prefilled (`src/prefill_manual_results.py`, idempotent — safe to re-run, never overwrites entered scores): group-stage entry is two numbers per match, `advanced` stays blank. **Knockout rows (from late June) are still hand-entered** and the old traps apply only to them: internal team names (Turkey, Czech Republic, Ivory Coast — copy from fixtures/site output, never type from memory), exact header, scoreline PLUS advancing team in `advanced`. Sparse KO metadata is fine — clean_data column-merges, and KO rows have no raw counterpart anyway.

2. **`python update.py`.** Does everything else: prepends the manual results, rebuilds ratings, runs the RESULT-AWARE sim (pins played matches by scoreline / KO winner, simulates only the remainder — Session 30), re-runs triple_compare, attaches results to frozen ledger forecasts + freezes the next day's, regenerates Claude previews/divergences, rebuilds the site (incl. CNAME).

3. **Check `logs/update_*.log`.** SOFT stages (odds/Polymarket/previews/divergences) warn and keep stale artifacts on failure; FATAL stages (clean_data/ratings/MC/triple_compare/site) must pass. Confirm no FATAL abort.

4. The Polymarket stage may print dropped duplicate rows (dead zero-volume events in the series) — expected operation, not a fault. Resolved markets persist in the series for now, so the fetcher's saved count can exceed the unplayed-fixture count (June 12: Polymarket saved 72, books 70, triple_compare 70 — all correct). Investigate only on LOW-LIQUIDITY warnings or a count that doesn't square with the fixtures Polymarket currently carries.

5. **Eyeball the site:** "What changed today" panel (now has real movers), calibration page, **confirm fixture listings aren't doubled (triple_compare row count = remaining unplayed fixtures with no duplicate pairs, enforced by FATAL assert — shrinks as matches are played)**, and a played match's page (Session 36): final-score header, frozen pre-match bars from the ledger, verdict line; check the index "Who's been closer" scoreboard and the schedule's Results section picked up the new results. The verdict now reads all-missed / sources-agreed / {source}-closest (Session 37 follow-up, §6), and the scoreboard advances ONLY on a distinct {source} win — so a quiet scoreboard ("model N" unchanged) on an upset-heavy day is correct, not a fault.

6. **Commit + push.** `update.py` does no git by design.

### Notes / gotchas

- **Ledger logs corrected probs** (Session 29, §6-recommended) — calibration tracking is correct from day one; no decision needed there.
- **Calibration page** shows the backtest seed until MIN_LIVE_N=24 scored ledger rows exist (generate_site.py, Session 36 follow-up), then switches to the live ledger automatically and shows the live n — expected ~June 19–20 at roughly three matches per day. No manual switch needed; raise/lower the constant in a planning chat if 24 proves wrong.
- **Survival grid → bracket** stays a survival grid until the group stage ends; redraw as a populated bracket ~June 27 (slots unknown until results exist). Planning-chat task.
- **No Kaggle re-pull needed mid-tournament** — results come in through wc_results_manual.csv. Kaggle is only the historical training base.
- **WC_SKIP_FETCH=1 python update.py** rebuilds from on-disk data without re-hitting markets — useful if the Odds API quota is tight (500/mo) or a market fetch is flaking and you just want to re-pin a result.
- **Sharing a match card:** the per-page OG image is correct server-side (Session OG); if X shows the old launch.png, that's X's ~7-day URL cache / sticky composer, not a bug — verify via an OG debugger, force-refresh with a fresh ?og= param if needed.