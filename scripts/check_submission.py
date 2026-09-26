#!/usr/bin/env python3
"""Local mirror of the official submission rules (use utils/validate_submission.py
from the challenge kit when available; this is the fallback).

  python scripts/check_submission.py --out output --test-dir dataset/test
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def read_ids(path: Path) -> set[str]:
    with open(path, encoding="utf-8") as fh:
        r = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(r)
        k = header.index("entity_id")
        return {row[k] for row in r}


def check(path: Path, col: str, s1: set[str], s23: set[str]) -> tuple[list[str], dict[str, set[str]]]:
    errs, seen, lists = [], set(), {}
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        if header != ["source1_entity_id", col]:
            errs.append(f"{path.name}: header {header}")
        for n, line in enumerate(fh, 2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                errs.append(f"{path.name}:{n}: expected 2 tab-separated fields"); continue
            s, ids = parts
            if s in seen:
                errs.append(f"{path.name}:{n}: duplicate row {s}")
            seen.add(s)
            items = [x for x in ids.split(",") if x] if ids else []
            if len(items) != len(set(items)):
                errs.append(f"{path.name}:{n}: duplicate ids in list")
            bad = [x for x in items if x not in s23]
            if bad:
                errs.append(f"{path.name}:{n}: unknown / non-S2-S3 ids {bad[:3]}")
            lists[s] = set(items)
            if len(errs) > 20:
                break
    missing = s1 - seen
    if missing:
        errs.append(f"{path.name}: {len(missing)} test S1 ids missing, e.g. {sorted(missing)[:3]}")
    extra = seen - s1
    if extra:
        errs.append(f"{path.name}: {len(extra)} rows are not test S1 ids")
    return errs, lists


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output")
    ap.add_argument("--test-dir", required=True)
    a = ap.parse_args()
    t = Path(a.test_dir)
    s1 = read_ids(t / "test_source1.tsv")
    s23 = read_ids(t / "test_source2.tsv") | read_ids(t / "test_source3.tsv")
    e1, match = check(Path(a.out) / "matching_results.tsv", "matched_entity_ids", s1, s23)
    e2, cand = check(Path(a.out) / "candidate_pairs.tsv", "candidate_entity_ids", s1, s23)
    not_sub = sum(1 for s, m in match.items() if not m <= cand.get(s, set()))
    errs = e1 + e2 + ([f"{not_sub} S1 rows have matches that are not in candidate_pairs"] if not_sub else [])
    for e in errs:
        print("FAIL:", e)
    if errs:
        return 1
    n_c = sum(len(v) for v in cand.values())
    n_m = sum(len(v) for v in match.values())
    print(f"PASS: {len(s1):,} S1 rows | {n_c / len(s1):.2f} candidates/S1 | {n_m / len(s1):.3f} matches/S1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
