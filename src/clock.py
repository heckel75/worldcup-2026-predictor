"""
src/clock.py — injectable "today" for the update pipeline.

Set env var WC_ASOF_DATE=YYYY-MM-DD to freeze the pipeline date for
dry-run replays (Session 30) or any re-run of a past date.
Leave it unset for normal daily operation.

Every module that needs today's date for snapshot naming, MC seeding, or
ledger lookahead should call clock.today() instead of datetime.date.today().
True wall-clock uses (log file timestamps in update.py) are exempt.
"""
import datetime as dt
import os


def today() -> dt.date:
    """Return today's date, or the date in WC_ASOF_DATE if set."""
    raw = os.environ.get("WC_ASOF_DATE", "").strip()
    if raw:
        return dt.date.fromisoformat(raw)
    return dt.date.today()


def clean_output_path() -> str:
    """Path for matches_clean.csv; WC_CLEAN_OUTPUT overrides for dry runs."""
    raw = os.environ.get("WC_CLEAN_OUTPUT", "").strip()
    return raw if raw else "data/processed/matches_clean.csv"


def snapshot_dir() -> str:
    """Snapshot directory for MC results; WC_SNAPSHOT_DIR overrides for dry runs."""
    raw = os.environ.get("WC_SNAPSHOT_DIR", "").strip()
    return raw if raw else "data/processed/snapshots"


def divergence_snapshot_dir() -> str:
    """Snapshot directory for divergence data; WC_DIVERGENCE_SNAPSHOT_DIR overrides."""
    raw = os.environ.get("WC_DIVERGENCE_SNAPSHOT_DIR", "").strip()
    return raw if raw else "data/processed/divergence_snapshots"
