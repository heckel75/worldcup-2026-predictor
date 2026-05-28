"""
dry_run.py — Session 30 tournament replay harness.

Simulates one complete WC 2026 "truth" from a fixed seed, then replays it
day-by-day through update.py to verify the pipeline is result-aware.

Protocol matches the real June morning ritual:
  - Before each match day: freeze forecasts (honest, match not yet played)
  - After each match day: results go into the feed for the NEXT run

Invariants checked every iteration:
  A. Column sums: sum(p_advance)==32, p_r16==16, p_qf==8, p_sf==4,
     p_final==2, p_champion==1.
  B. Group-eliminated teams (all 6 group games complete + truth rank 4):
     p_advance == 0  and  p_champion == 0.
  C. Champion's p_champion: non-decreasing across iterations (±CHAMP_TOLERANCE
     for MC noise) and == 1.0 on the final run.
  D. What-changed movers: non-empty from iteration 1 onward (when the feed
     contains at least one day's results and two dated snapshots exist to diff).
  E. Ledger (final run): 72 group-stage rows frozen, all scored.

Run from project root:
    python dry_run.py

Side-effects that survive after the run (review before committing):
  - data/processed/snapshots/2026-06-*.csv, 2026-07-*.csv   (new dated snapshots)
  - data/processed/divergence_snapshots/2026-06-*.csv, ...  (new dated snapshots)
  - data/processed/wc_predictions.csv  RESTORED to original state after the run
  - data/processed/matches_clean.csv   reflects full synthetic tournament at end
    (regenerated fresh on any subsequent real update.py run)
  - data/tmp_dry_run_feed.csv          deleted on clean exit
  - docs/                              rebuilt to final-day state
"""
from __future__ import annotations

import csv
import datetime as dt
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from bracket import GROUPS, TEAM_TO_GROUP
from monte_carlo import EXPECTED_SUMS, STAGE_COLS
from simulate import simulate_tournament

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DRY_RUN_SEED = 42

FIXTURES_PATH    = Path("data/processed/fixtures_2026.csv")
RATINGS_PATH     = Path("data/processed/elo_ratings_2026.csv")
REAL_FEED_PATH   = Path("data/raw/wc_results_manual.csv")
TEMP_FEED_PATH   = Path("data/tmp_dry_run_feed.csv")
DIV_SNAPS_DIR    = Path("data/processed/divergence_snapshots")
LEDGER_PATH     = Path("data/processed/wc_predictions.csv")
LEDGER_BACKUP   = Path("data/processed/wc_predictions.dry_run_backup.csv")
SNAPSHOTS_DIR   = Path("data/processed/snapshots")

# Synthetic KO round dates: arbitrary post-group calendar, chronologically
# ordered by bracket depth.  Values don't need to match the real WC calendar —
# only order and uniqueness matter for snapshot naming and MC seeds.
KO_ROUND_DATES: dict[str, dt.date] = {
    "R32":                 dt.date(2026, 7, 1),
    "R16":                 dt.date(2026, 7, 5),
    "QF":                  dt.date(2026, 7, 9),
    "SF":                  dt.date(2026, 7, 12),
    "third_place_playoff": dt.date(2026, 7, 15),
    "final":               dt.date(2026, 7, 16),
}

# MC noise ±~0.4pp at p=0.20 with 10k sims + different seeds per day.
# Allow 2pp slack on the non-decreasing champion check.
CHAMP_TOLERANCE = 0.02


# ---------------------------------------------------------------------------
# Build truth
# ---------------------------------------------------------------------------

def _build_truth() -> tuple[dict, list[dict]]:
    """Simulate one tournament (DRY_RUN_SEED), return truth dict + dated schedule.

    Schedule entry keys: home, away, hg, ag, round, winner, date (datetime.date).
    """
    ratings_df = pd.read_csv(RATINGS_PATH)
    ratings = dict(zip(ratings_df["team"], ratings_df["elo"]))

    fixtures_df = pd.read_csv(FIXTURES_PATH, parse_dates=["date"])
    fixtures = fixtures_df[["home_team", "away_team"]].to_dict("records")

    # (home, away) → real fixture date
    fixture_dates: dict[tuple[str, str], dt.date] = {
        (r["home_team"], r["away_team"]): r["date"].date()
        for _, r in fixtures_df.iterrows()
    }

    rng = np.random.default_rng(DRY_RUN_SEED)
    truth = simulate_tournament(ratings, fixtures, rng, return_results=True)

    schedule: list[dict] = []
    for mr in truth["match_results"]:
        h, a, rd = mr["home"], mr["away"], mr["round"]
        d = fixture_dates[(h, a)] if rd == "group_stage" else KO_ROUND_DATES[rd]
        schedule.append({**mr, "date": d})

    schedule.sort(key=lambda r: r["date"])
    return truth, schedule


# ---------------------------------------------------------------------------
# Temp feed helpers
# ---------------------------------------------------------------------------

def _init_temp_feed() -> None:
    """Write the real feed's header into the temp feed, clearing any old content."""
    header_line = REAL_FEED_PATH.read_text(encoding="utf-8").splitlines()[0]
    TEMP_FEED_PATH.write_text(header_line + "\n", encoding="utf-8")


def _append_to_feed(matches: list[dict]) -> None:
    """Append one day's results to the temp feed.

    Group matches: real (hg, ag) scoreline.
    KO matches: 0-0 + advanced = winner  (signals "went to penalties", winner known).
    """
    with TEMP_FEED_PATH.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        for m in matches:
            if m["round"] == "group_stage":
                writer.writerow([
                    m["date"].isoformat(),
                    m["home"], m["away"],
                    m["hg"], m["ag"],
                    "",                  # advanced
                    "FIFA World Cup",
                    "N/A", "N/A",        # city, country
                    "True",
                ])
            else:
                # KO result stored as 0-0 + advanced so build_known_results
                # resolves the draw → advanced path and pins the correct winner.
                writer.writerow([
                    m["date"].isoformat(),
                    m["home"], m["away"],
                    0, 0,
                    m["winner"],         # advanced
                    "FIFA World Cup",
                    "N/A", "N/A",
                    "True",
                ])


# ---------------------------------------------------------------------------
# Run update.py
# ---------------------------------------------------------------------------

def _run_update(asof_date: dt.date) -> str:
    """Run update.py with dry-run env vars; return combined stdout+stderr."""
    env = {
        **os.environ,
        "WC_ASOF_DATE":       asof_date.isoformat(),
        "WC_SKIP_FETCH":      "1",
        "WC_MANUAL_RESULTS":  str(TEMP_FEED_PATH),
        "PYTHONIOENCODING":   "utf-8",
    }
    result = subprocess.run(
        [sys.executable, "update.py"],
        capture_output=True, text=True, encoding="utf-8", env=env,
    )
    combined = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        print("  [WARN] update.py exited non-zero:")
        for line in combined.splitlines()[-20:]:
            print(f"    {line}")
    return combined


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def _read_snapshot(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _two_newest_snapshots() -> tuple[Path | None, Path | None]:
    csvs = sorted(SNAPSHOTS_DIR.glob("*.csv"))
    if not csvs:
        return None, None
    if len(csvs) == 1:
        return None, csvs[-1]
    return csvs[-2], csvs[-1]


# ---------------------------------------------------------------------------
# Invariant helpers
# ---------------------------------------------------------------------------

def _check_column_sums(snap: pd.DataFrame, label: str) -> None:
    """Invariant A: per-round column sums must equal expected values."""
    for col, expected in EXPECTED_SUMS.items():
        got = snap[col].sum()
        assert abs(got - expected) < 1e-9, (
            f"[{label}] FAIL column-sum invariant: "
            f"{col} sums to {got:.6f}, expected {expected}"
        )


def _compute_group_eliminated(truth: dict, played_set: set[frozenset]) -> set[str]:
    """Return teams definitively eliminated from group stage.

    A team is eliminated when:
      (a) all 6 of their group's matches are in played_set, AND
      (b) they finished rank 4 in truth's group standings.
    """
    eliminated: set[str] = set()
    for letter, teams in GROUPS.items():
        # All C(4,2)=6 unordered pairs for this group
        pairs = {
            frozenset({t1, t2})
            for i, t1 in enumerate(teams)
            for t2 in teams[i + 1:]
        }
        if not pairs.issubset(played_set):
            continue  # group not yet complete
        standings = truth["group_results"][letter]
        fourth = next(s["team"] for s in standings if s["rank"] == 4)
        eliminated.add(fourth)
    return eliminated


def _check_eliminated(snap: pd.DataFrame, eliminated: set[str], label: str) -> None:
    """Invariant B: definitively eliminated teams have p_advance == p_champion == 0."""
    for team in eliminated:
        row = snap.loc[snap["team"] == team]
        if row.empty:
            continue
        p_adv  = float(row["p_advance"].iloc[0])
        p_chmp = float(row["p_champion"].iloc[0])
        assert p_adv == 0.0, (
            f"[{label}] FAIL: {team!r} is group-eliminated but p_advance={p_adv:.4f}"
        )
        assert p_chmp == 0.0, (
            f"[{label}] FAIL: {team!r} is group-eliminated but p_champion={p_chmp:.4f}"
        )


def _check_what_changed(iteration: int, label: str) -> None:
    """Invariant D: from iteration 1+, at least one stage column must move >= 0.5pp.

    Uses direct diff of the two newest snapshots without importing generate_site.py.

    Exception: once only 2 teams have p_champion > 0 (we're in the final),
    the third-place playoff result doesn't change any stage probability for any
    team (both playoff teams already hold stage level 4 regardless of who wins),
    so the invariant is vacuously unsatisfiable and is skipped.
    """
    if iteration == 0:
        return  # first run: pre-tournament baseline, movers may be identical to prev

    prev_path, curr_path = _two_newest_snapshots()
    if prev_path is None:
        return  # only one snapshot yet — can't diff

    prev = _read_snapshot(prev_path)
    curr = _read_snapshot(curr_path)

    # Skip if only 2 teams still have title chances (final phase).
    # The third-place playoff pins a result but changes no MC stage column,
    # so requiring a mover here would always fail regardless of correctness.
    if int((prev["p_champion"] > 1e-9).sum()) <= 2:
        return

    MIN_MOVE_PP = 0.5  # same floor as whats_changed.py

    for col in ("p_champion", "p_final", "p_sf", "p_qf", "p_r16", "p_advance"):
        merged = curr[["team", col]].merge(prev[["team", col]], on="team",
                                           suffixes=("_c", "_p"))
        merged["delta"] = (merged[f"{col}_c"] - merged[f"{col}_p"]).abs() * 100
        if (merged["delta"] >= MIN_MOVE_PP).any():
            return  # at least one team moved enough — invariant satisfied

    assert False, (
        f"[{label}] FAIL: iteration {iteration} has >=2 snapshots but no team's "
        f"p_champion/p_final/p_sf/p_qf/p_r16/p_advance changed by >= {MIN_MOVE_PP}pp. "
        f"Either results are not reaching the simulation or the seeds "
        f"produce pathologically similar outputs."
    )


# ---------------------------------------------------------------------------
# Ledger helpers
# ---------------------------------------------------------------------------

def _backup_ledger() -> None:
    if LEDGER_PATH.exists():
        shutil.copy2(LEDGER_PATH, LEDGER_BACKUP)


def _restore_ledger() -> None:
    if LEDGER_BACKUP.exists():
        shutil.copy2(LEDGER_BACKUP, LEDGER_PATH)
        LEDGER_BACKUP.unlink()


def _count_ledger_rows() -> tuple[int, int]:
    """Return (total_rows, scored_rows) from the current ledger."""
    if not LEDGER_PATH.exists():
        return 0, 0
    df = pd.read_csv(LEDGER_PATH)
    if df.empty:
        return 0, 0
    scored = int((~df["outcome"].isna() & (df["outcome"] != "")).sum())
    return len(df), scored


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 68)
    print("DRY RUN — World Cup 2026 tournament replay harness")
    print(f"Truth seed: {DRY_RUN_SEED}")
    print("=" * 68)

    # ---- Restore pipeline to clean pre-tournament state --------------------
    # Required if a previous dry run left fixtures_2026.csv or
    # elo_ratings_2026.csv in a corrupted state (games played moved out of
    # fixtures, fewer than 48 WC teams in ratings file).
    print("\nRestoring pipeline to pre-tournament state...")
    for stage_cmd in [
        [sys.executable, "src/clean_data.py"],
        [sys.executable, "src/save_wc_ratings.py"],
    ]:
        r = subprocess.run(
            stage_cmd, capture_output=True, text=True,
            encoding="utf-8",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        label = stage_cmd[-1]
        if r.returncode != 0:
            print(f"  WARN: {label} failed:\n{r.stderr[-500:]}")
        else:
            print(f"  OK: {label}")

    # Clear stale WC-2026 snapshots left by previous dry runs.  Without this,
    # _two_newest_snapshots() picks old pre-existing files instead of the
    # current iteration's snapshot, making invariant checks run on wrong data.
    for snap_dir in [SNAPSHOTS_DIR, DIV_SNAPS_DIR]:
        stale = [p for p in snap_dir.glob("*.csv") if p.stem >= "2026-06-01"]
        for p in stale:
            p.unlink()
        if stale:
            print(f"  Removed {len(stale)} stale snapshot(s) from {snap_dir.name}/")

    # ---- Build truth -------------------------------------------------------
    print("\nBuilding truth simulation...")
    truth, schedule = _build_truth()
    champion = truth["winner"]
    print(f"  Champion (truth): {champion}")
    print(f"  Schedule: {len(schedule)} matches across "
          f"{len({m['date'] for m in schedule})} unique dates")

    # ---- Prep --------------------------------------------------------------
    _init_temp_feed()
    _backup_ledger()

    # Group-fixture pairs for eliminated-team tracking
    fixtures_df = pd.read_csv(FIXTURES_PATH)
    group_pairs_set: set[frozenset] = {
        frozenset({r["home_team"], r["away_team"]})
        for _, r in fixtures_df.iterrows()
    }
    # Unique ordered match dates for iteration
    unique_dates = sorted({m["date"] for m in schedule})

    # Running state
    played_frozensets: set[frozenset] = set()   # group pairs played so far
    all_played: list[dict] = []                 # full history for feed re-init
    prev_champ_p: float | None = None
    champ_p_history: list[float] = []

    print(f"\n{'Date':<12}  {'Feed':>5}  {'Snap champ%':>11}  "
          f"{'Elim':>5}  {'Ledger':>13}  Notes")
    print("-" * 68)

    try:
        for iteration, asof in enumerate(unique_dates):
            # matches on this date (to append AFTER the run)
            day_matches = [m for m in schedule if m["date"] == asof]
            label = f"iter={iteration} date={asof}"

            # Count feed rows BEFORE the update (file definitely exists here).
            n_feed = sum(1 for _ in TEMP_FEED_PATH.open(encoding="utf-8")) - 1

            # ---- Run update.py (sees feed with results through asof-1) -----
            _run_update(asof)

            # Guard: if the temp feed disappeared during the update (cause is
            # unknown but reproducible), re-create it from the accumulated
            # match history so subsequent iterations continue correctly.
            if not TEMP_FEED_PATH.exists():
                print(f"  [WARN] temp feed missing after update — "
                      f"re-creating from {len(all_played)} prior entries")
                _init_temp_feed()
                if all_played:
                    _append_to_feed(all_played)

            # ---- Read latest snapshot --------------------------------------
            _, curr_path = _two_newest_snapshots()
            assert curr_path is not None, f"[{label}] No snapshot found after update"
            snap = _read_snapshot(curr_path)

            # ---- Invariant A: column sums ----------------------------------
            _check_column_sums(snap, label)

            # ---- Invariant B: eliminated teams -----------------------------
            eliminated = _compute_group_eliminated(truth, played_frozensets)
            _check_eliminated(snap, eliminated, label)

            # ---- Champion probability --------------------------------------
            champ_row = snap.loc[snap["team"] == champion, "p_champion"]
            assert not champ_row.empty, f"Champion {champion!r} missing from snapshot"
            curr_champ_p = float(champ_row.iloc[0])
            champ_p_history.append(curr_champ_p)

            # ---- Invariant C: champion monotone ----------------------------
            if prev_champ_p is not None:
                assert curr_champ_p >= prev_champ_p - CHAMP_TOLERANCE, (
                    f"[{label}] FAIL: champion p_champion dropped from "
                    f"{prev_champ_p:.3f} to {curr_champ_p:.3f} "
                    f"(beyond {CHAMP_TOLERANCE:.2f} tolerance)"
                )
            prev_champ_p = curr_champ_p

            # ---- Invariant D: what changed ---------------------------------
            _check_what_changed(iteration, label)

            # ---- Summary line ----------------------------------------------
            n_elim = len(eliminated)
            ledger_total, ledger_scored = _count_ledger_rows()
            notes = "baseline" if iteration == 0 else ""
            print(
                f"{asof!s:<12}  {n_feed:>5}  {curr_champ_p:>10.1%}  "
                f"{n_elim:>5}  {ledger_scored:>5}/{ledger_total:<5}  {notes}"
            )

            # ---- Append this day's results to feed -------------------------
            _append_to_feed(day_matches)
            all_played.extend(day_matches)
            for m in day_matches:
                if m["round"] == "group_stage":
                    played_frozensets.add(frozenset({m["home"], m["away"]}))

        # ---- Final run: close out after last results are in the feed -------
        final_asof = unique_dates[-1] + dt.timedelta(days=1)
        print(f"\n{'='*68}")
        print(f"Final run: WC_ASOF_DATE={final_asof} (attaches last day's results)")
        _run_update(final_asof)

        _, curr_path = _two_newest_snapshots()
        snap_final = _read_snapshot(curr_path)

        # Invariant A (final)
        _check_column_sums(snap_final, "final")

        # Invariant C final: champion p == 1.0, everyone else == 0.0
        final_champ_p = float(
            snap_final.loc[snap_final["team"] == champion, "p_champion"].iloc[0]
        )
        assert abs(final_champ_p - 1.0) < 1e-9, (
            f"FAIL: champion {champion!r} has p_champion={final_champ_p:.6f} "
            f"on final run, expected exactly 1.0"
        )
        non_champs = snap_final[snap_final["team"] != champion]
        bad = non_champs[non_champs["p_champion"] > 1e-9]
        assert bad.empty, (
            f"FAIL: non-champion teams have non-zero p_champion on final run: "
            f"{bad[['team','p_champion']].to_dict('records')}"
        )

        # Invariant E: ledger has 72 group rows, all scored
        ledger_total, ledger_scored = _count_ledger_rows()
        n_group = len(fixtures_df)  # 72
        assert ledger_total == n_group, (
            f"FAIL: ledger has {ledger_total} rows, expected {n_group} group matches"
        )
        assert ledger_scored == n_group, (
            f"FAIL: ledger has {ledger_scored} scored rows, expected {n_group}"
        )

        print(f"\n{'='*68}")
        print("ALL INVARIANTS PASSED")
        print(f"  Champion: {champion} (final p_champion = {final_champ_p:.1%})")
        print(f"  Ledger:   {ledger_scored}/{ledger_total} group matches frozen+scored")
        print(f"  Champion p_champion history ({len(champ_p_history)} days):")
        for i, (d, p) in enumerate(zip(unique_dates, champ_p_history)):
            print(f"    {d}  {p:.1%}")
        print(f"    {final_asof}  {final_champ_p:.1%}  (final)")

    finally:
        # Restore ledger to pre-dry-run state regardless of pass/fail
        _restore_ledger()
        # Clean up temp feed
        if TEMP_FEED_PATH.exists():
            TEMP_FEED_PATH.unlink()
        print(f"\nLedger restored to original state.  Temp feed deleted.")


if __name__ == "__main__":
    main()
