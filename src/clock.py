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
