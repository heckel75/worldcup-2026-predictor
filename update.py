#!/usr/bin/env python
"""
update.py — World Cup 2026 Predictor: end-to-end update orchestrator.

Run from project root:
    python update.py

Pipeline:
  Stage  1 — fetch_odds.py           SOFT  (network; stale artifact fine)
  Stage  2 — fetch_polymarket.py     SOFT
  Stage  3 — clean_data.py           FATAL
  Stage  4 — save_wc_ratings.py      FATAL
  Stage  5 — monte_carlo.py          FATAL
  Stage  6 — triple_compare.py       FATAL
  Stage  7 — ledger update           in-process
  Stage  8 — generate_previews.py    SOFT  (Claude API; caches survive failures)
  Stage  9 — generate_divergences.py SOFT
  Stage 10 — generate_site.py        FATAL

SOFT failure: log WARNING, keep the existing cached artifact, continue.
FATAL failure: log ERROR, abort immediately, exit non-zero.

No git operations. Review docs/ and commit manually when satisfied.
"""
from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

LEDGER_LOOKAHEAD_DAYS = 1

LEDGER_PATH  = Path("data/processed/wc_predictions.csv")
FIXTURES_PATH = Path("data/processed/fixtures_2026.csv")
TRIPLE_PATH  = Path("data/processed/triple_compare.csv")
PLAYED_PATH  = Path("data/processed/matches_clean.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tee(text: str, log_fh) -> None:
    """Write text to console and log file."""
    print(text, flush=True)
    log_fh.write(text + "\n")
    log_fh.flush()


def run_stage(label: str, cmd: list[str], log_fh, fatal: bool = True) -> bool:
    """
    Run a subprocess stage, tee its output to console + log.
    Returns True on success, False on soft failure.
    Calls sys.exit(1) on fatal failure.
    """
    _tee(f"\n{'='*60}", log_fh)
    _tee(f"STAGE: {label}", log_fh)
    _tee(f"CMD:   {' '.join(cmd)}", log_fh)
    _tee(f"{'='*60}", log_fh)

    _env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=_env)

    if result.stdout:
        _tee(result.stdout.rstrip(), log_fh)
    if result.stderr:
        _tee(result.stderr.rstrip(), log_fh)

    if result.returncode == 0:
        _tee(f"[OK] {label}", log_fh)
        return True

    if fatal:
        _tee(f"[ERROR] {label} exited {result.returncode} — aborting.", log_fh)
        sys.exit(1)
    else:
        _tee(f"[WARNING] {label} exited {result.returncode} — using stale artifact.", log_fh)
        return False


def _update_ledger(log_fh) -> tuple[int, int]:
    """
    In-process ledger update. Returns (rows_frozen, results_attached).
    Reads and writes data/processed/wc_predictions.csv.
    """
    import importlib
    import importlib.util

    _tee(f"\n{'='*60}", log_fh)
    _tee("STAGE: ledger_update (in-process)", log_fh)
    _tee(f"{'='*60}", log_fh)

    try:
        import pandas as pd

        # Lazy import so we don't need update_ledger on sys.path before this point
        if "src" not in sys.path:
            sys.path.insert(0, "src")

        # Re-import to pick up any edits; use importlib to avoid cached module
        spec = importlib.util.spec_from_file_location(
            "update_ledger", Path("src/update_ledger.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        from clock import today as _clock_today  # noqa: PLC0415

        ledger_df = pd.read_csv(LEDGER_PATH)
        fixtures_df = pd.read_csv(FIXTURES_PATH, parse_dates=["date"])
        triple_df = pd.read_csv(TRIPLE_PATH, parse_dates=["date"])
        played_df = pd.read_csv(PLAYED_PATH, parse_dates=["date"])

        before_len = len(ledger_df)

        today = _clock_today()
        ledger_df = mod.freeze_new_forecasts(
            ledger_df, fixtures_df, triple_df, today, LEDGER_LOOKAHEAD_DAYS
        )
        rows_frozen = len(ledger_df) - before_len

        ledger_before_attach = ledger_df.copy()
        ledger_df = mod.attach_results(ledger_df, played_df)

        # Count how many rows got a result filled in
        if len(ledger_before_attach) > 0 and "outcome" in ledger_before_attach.columns:
            was_empty = (ledger_before_attach["outcome"].isna()
                         | (ledger_before_attach["outcome"] == ""))
            is_now_filled = (~ledger_df["outcome"].isna()
                             & (ledger_df["outcome"] != ""))
            results_attached = int((was_empty & is_now_filled).sum())
        else:
            results_attached = 0

        ledger_df.to_csv(LEDGER_PATH, index=False)

        _tee(f"Frozen: {rows_frozen} new row(s). Results attached: {results_attached}.", log_fh)
        _tee(f"[OK] ledger_update", log_fh)
        return rows_frozen, results_attached

    except Exception as exc:
        _tee(f"[WARNING] ledger_update failed: {exc}", log_fh)
        return 0, 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Reconfigure stdout/stderr to UTF-8 so Unicode in subprocess output
    # (e.g., '→' in clean_data.py) doesn't crash on Windows cp1252 consoles.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    ts_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"update_{ts_str}.log"

    py = sys.executable

    with open(log_path, "w", encoding="utf-8") as log_fh:
        started_at = dt.datetime.now()
        _tee(f"World Cup 2026 Predictor — update started {started_at.isoformat(timespec='seconds')}", log_fh)
        _tee(f"Log: {log_path}", log_fh)

        stage_results: dict[str, bool] = {}

        # --- SOFT stages: market data fetchers ---
        # Set WC_SKIP_FETCH=1 to skip network fetches (dry runs, API down).
        skip_fetch = os.environ.get("WC_SKIP_FETCH", "0").strip() == "1"
        if skip_fetch:
            _tee("WC_SKIP_FETCH=1 — skipping fetch_odds and fetch_polymarket.", log_fh)
            stage_results["fetch_odds"] = True
            stage_results["fetch_polymarket"] = True
        else:
            stage_results["fetch_odds"] = run_stage(
                "fetch_odds", [py, "src/fetch_odds.py"], log_fh, fatal=False
            )
            stage_results["fetch_polymarket"] = run_stage(
                "fetch_polymarket", [py, "src/fetch_polymarket.py"], log_fh, fatal=False
            )

        # --- FATAL stages: data pipeline ---
        run_stage("clean_data",    [py, "src/clean_data.py"],    log_fh, fatal=True)
        run_stage("save_ratings",  [py, "src/save_wc_ratings.py"], log_fh, fatal=True)
        run_stage("monte_carlo",   [py, "src/monte_carlo.py"],   log_fh, fatal=True)
        run_stage("triple_compare",[py, "src/triple_compare.py"],log_fh, fatal=True)

        # --- In-process: ledger update ---
        rows_frozen, results_attached = _update_ledger(log_fh)

        # --- SOFT stages: Claude commentary (cached; API may be unavailable) ---
        stage_results["generate_previews"] = run_stage(
            "generate_previews", [py, "src/generate_previews.py"], log_fh, fatal=False
        )
        stage_results["generate_divergences"] = run_stage(
            "generate_divergences", [py, "src/generate_divergences.py"], log_fh, fatal=False
        )

        # --- FATAL: site rebuild ---
        run_stage("generate_site", [py, "generate_site.py"], log_fh, fatal=True)

        # --- Summary ---
        finished_at = dt.datetime.now()
        elapsed = (finished_at - started_at).total_seconds()

        _tee(f"\n{'='*60}", log_fh)
        _tee(f"=== Update complete: {finished_at.strftime('%Y-%m-%d %H:%M:%S')} "
             f"({elapsed:.0f}s) ===", log_fh)

        soft_labels = {
            "fetch_odds":           "fetch_odds",
            "fetch_polymarket":     "fetch_polymarket",
            "generate_previews":    "generate_previews",
            "generate_divergences": "generate_divergences",
        }
        for key, label in soft_labels.items():
            status = "OK" if stage_results.get(key, False) else "SOFT FAIL (stale artifact used)"
            _tee(f"  {label:<24} {status}", log_fh)

        if rows_frozen > 0 or results_attached > 0:
            _tee(f"  Ledger: {rows_frozen} new row(s) frozen, "
                 f"{results_attached} result(s) attached", log_fh)
        else:
            _tee(f"  Ledger: no changes (all fixtures still weeks out, or no new results)", log_fh)

        _tee(f"\n  → Open docs/index.html to review the \"What changed today\" panel", log_fh)
        _tee(f"  → Review docs/ and commit manually when satisfied", log_fh)
        _tee(f"\nLog: {log_path}", log_fh)


if __name__ == "__main__":
    main()
