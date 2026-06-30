"""Prefill the WC manual results feed with the 72 group-stage fixtures.

This script is safe to re-run and preserves any existing manual row for a fixture.
Blank score/advanced cells are written only for fixtures that are missing in the
manual file.
"""

import pandas as pd
from pathlib import Path

RAW_MANUAL_PATH = Path("data/raw/wc_results_manual.csv")
FIXTURES_PATH = Path("data/processed/fixtures_2026.csv")
RAW_RESULTS_PATH = Path("data/raw/results.csv")


def normalize_team(name: str, name_map: dict) -> str:
    if pd.isna(name):
        return name
    return name_map.get(name, name)


def main():
    if not FIXTURES_PATH.exists():
        raise FileNotFoundError(f"Missing fixtures file: {FIXTURES_PATH}")

    fixtures = pd.read_csv(FIXTURES_PATH, dtype=str)
    if fixtures.empty:
        raise ValueError(f"No fixtures found in {FIXTURES_PATH}")

    if RAW_MANUAL_PATH.exists():
        manual = pd.read_csv(RAW_MANUAL_PATH, dtype=str)
        # Ensure the free-text full-result column exists on older feeds (blank
        # default, never overwritten — preserved like every other column).
        if "result_note" not in manual.columns:
            manual["result_note"] = ""
        existing_rows = len(manual)
    else:
        manual = pd.DataFrame(columns=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "advanced",
            "tournament",
            "city",
            "country",
            "neutral",
            "result_note",
        ])
        existing_rows = 0

    manual_cols = list(manual.columns)
    required_cols = ["date", "home_team", "away_team", "home_score", "away_score", "advanced"]
    for col in required_cols:
        if col not in manual_cols:
            raise ValueError(f"Expected manual file to contain column '{col}', got {manual_cols}")

    raw_name_map = {}
    raw_results_lookup = {}
    if RAW_RESULTS_PATH.exists():
        raw_results = pd.read_csv(RAW_RESULTS_PATH, dtype=str)
        try:
            import clean_data
            raw_name_map = getattr(clean_data, "TEAM_NAME_MAP", {})
        except ImportError:
            raw_name_map = {}

        for _, row in raw_results.iterrows():
            home = normalize_team(row.get("home_team"), raw_name_map)
            away = normalize_team(row.get("away_team"), raw_name_map)
            if pd.notna(row.get("date")) and pd.notna(home) and pd.notna(away):
                key = (row["date"], frozenset({home, away}))
                raw_results_lookup.setdefault(key, []).append(row.to_dict())

    manual_index = {
        (row["date"], row["home_team"], row["away_team"]): idx
        for idx, row in manual.iterrows()
        if pd.notna(row.get("date")) and pd.notna(row.get("home_team")) and pd.notna(row.get("away_team"))
    }

    new_rows = []
    preserved = 0
    for _, fixture in fixtures.iterrows():
        key = (fixture["date"], fixture["home_team"], fixture["away_team"])
        raw_entries = raw_results_lookup.get((fixture["date"], frozenset({fixture["home_team"], fixture["away_team"]})), [])
        raw_data = None
        if raw_entries:
            raw_data = next(
                (row for row in raw_entries if row.get("home_team") == fixture["home_team"] and row.get("away_team") == fixture["away_team"]),
                raw_entries[0],
            )

        if key in manual_index:
            preserved += 1
            idx = manual_index[key]
            row = manual.loc[idx].copy()
            if raw_data is not None:
                for col in manual_cols:
                    manual_val = row.get(col)
                    raw_val = raw_data.get(col)
                    if manual_val in [None, ""] or pd.isna(manual_val):
                        if raw_val is not None and raw_val != "":
                            row[col] = raw_val
                manual.loc[idx] = row
            continue

        row = {col: "" for col in manual_cols}
        for col in manual_cols:
            if col in fixture.index and pd.notna(fixture[col]):
                row[col] = fixture[col]
            elif raw_data is not None and raw_data.get(col) is not None and raw_data.get(col) != "":
                row[col] = raw_data.get(col)

        row["home_score"] = ""
        row["away_score"] = ""
        row["advanced"] = ""
        new_rows.append(row)

    if new_rows:
        new_df = pd.DataFrame(new_rows, columns=manual_cols)
        manual = pd.concat([manual, new_df], ignore_index=True)

    if new_rows:
        new_df = pd.DataFrame(new_rows, columns=manual_cols)
        manual = pd.concat([manual, new_df], ignore_index=True)

    manual.to_csv(RAW_MANUAL_PATH, index=False)

    print(f"Prefilled {len(new_rows)} fixture rows into {RAW_MANUAL_PATH}")
    print(f"Preserved {preserved} existing manual row(s)")
    print(f"Total rows now in file: {len(manual)}")


if __name__ == "__main__":
    main()
