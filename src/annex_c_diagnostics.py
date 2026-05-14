"""Diagnostic: for each of the 495 Q-sets, count valid assignments under the
slot-family + no-rematch constraints alone, to show why FIFA needed to publish
a table rather than have it be derived from the constraints.

Counts get split into:
  - unique:  exactly one valid assignment (table is forced)
  - multi:   2+ valid assignments (FIFA chose one for balance reasons)
"""
from __future__ import annotations

import csv
from itertools import permutations
from pathlib import Path

SLOT_FAMILIES = {
    "1A": set("CEFHI"),
    "1B": set("EFGIJ"),
    "1D": set("BEFIJ"),
    "1E": set("ABCDF"),
    "1G": set("AEHIJ"),
    "1I": set("CDFGH"),
    "1K": set("DEIJL"),
    "1L": set("EHIJK"),
}
SLOT_ORDER = ["1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L"]


def count_valid_assignments(qual: list[str]) -> int:
    """Brute-force count of bijections Q → SLOT_ORDER satisfying both constraints."""
    n = 0
    for perm in permutations(qual):
        ok = True
        for slot_id, third in zip(SLOT_ORDER, perm):
            if third not in SLOT_FAMILIES[slot_id]:
                ok = False
                break
            if third == slot_id[1]:
                ok = False
                break
        if ok:
            n += 1
    return n


def main() -> None:
    rows = list(csv.DictReader(Path("data/raw/r32_annex_c.csv").open()))
    counts = {}
    fifa_choices_match = 0
    for row in rows:
        qual = [row[f"Q{i}"] for i in range(1, 9)]
        fifa = [row[f"slot_{s}"] for s in SLOT_ORDER]
        n = count_valid_assignments(qual)
        counts[n] = counts.get(n, 0) + 1
        # sanity: FIFA's assignment must itself be valid (we already enforced
        # this when building, but cross-check from the CSV)
        for slot_id, third in zip(SLOT_ORDER, fifa):
            assert third in SLOT_FAMILIES[slot_id]
            assert third != slot_id[1]
        fifa_choices_match += 1

    print(f"Rows checked: {fifa_choices_match}")
    print("Valid-assignment count distribution across the 495 Q-sets:")
    for k in sorted(counts):
        print(f"  {k:>3} valid assignment(s): {counts[k]:>4} Q-sets")
    unique = counts.get(1, 0)
    multi = sum(v for k, v in counts.items() if k > 1)
    print(f"\nUnique (table forced):   {unique} / 495")
    print(f"Multiple (FIFA chose):   {multi} / 495")


if __name__ == "__main__":
    main()