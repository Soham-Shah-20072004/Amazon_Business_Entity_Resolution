"""Dataset I/O helpers.

All challenge files are TAB-separated. Every read uses ``sep="\\t"`` and
``dtype=str`` so IDs / PIN codes are never coerced to numbers and parsing is
identical on every machine.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import AppConfig

TSV_SEP = "\t"


# --------------------------------------------------------------------------
# low-level reading
# --------------------------------------------------------------------------

def read_tsv(
    path: str | Path,
    expected_columns: Optional[List[str]] = None,
    strict_columns: bool = False,
) -> pd.DataFrame:
    """Read a challenge TSV with ``sep="\\t"`` and ``dtype=str``.

    Empty cells become "" (never NaN) so downstream string code is total.
    Raises FileNotFoundError if missing, ValueError on column mismatch when
    strict_columns=True. Uses the C engine (fast on 5M-row tables).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TSV not found: {path}")
    df = pd.read_csv(
        path,
        sep=TSV_SEP,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    # Defensive: strip a UTF-8 BOM from the first column name if present.
    df.columns = [c.replace("\ufeff", "") for c in df.columns]
    # Ensure every value is a str (paranoia for mixed-type edge cases).
    for col in df.columns:
        df[col] = df[col].astype(str)
    if expected_columns is not None and strict_columns:
        missing = [c for c in expected_columns if c not in df.columns]
        if missing:
            raise ValueError(f"{path}: missing expected columns {missing}")
    return df


def try_read_tsv(
    path: str | Path,
    expected_columns: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """read_tsv that returns None (instead of raising) when the file is absent."""
    path = Path(path)
    if not path.exists():
        return None
    return read_tsv(path, expected_columns=expected_columns, strict_columns=False)


# --------------------------------------------------------------------------
# challenge layout
# --------------------------------------------------------------------------

def load_training_tables(cfg: AppConfig) -> Dict[str, Optional[pd.DataFrame]]:
    """Load train_source1/2/3 + ground truth (None for any missing file)."""
    cols = cfg.columns
    src_cols = [cols["entity_id"], cols["business_name"], cols["business_address"], cols["country"]]
    gt_cols = [cols["gt_source1"], cols["gt_matched"]]
    paths = cfg.train_paths()
    return {
        "train_s1": try_read_tsv(paths["train_source1"], src_cols),
        "train_s2": try_read_tsv(paths["train_source2"], src_cols),
        "train_s3": try_read_tsv(paths["train_source3"], src_cols),
        "train_gt": try_read_tsv(paths["train_ground_truth"], gt_cols),
    }


def load_test_tables(cfg: AppConfig) -> Dict[str, Optional[pd.DataFrame]]:
    """Load test_source1/2/3 (None for any missing file)."""
    cols = cfg.columns
    src_cols = [cols["entity_id"], cols["business_name"], cols["business_address"], cols["country"]]
    paths = cfg.test_paths()
    return {
        "test_s1": try_read_tsv(paths["test_source1"], src_cols),
        "test_s2": try_read_tsv(paths["test_source2"], src_cols),
        "test_s3": try_read_tsv(paths["test_source3"], src_cols),
    }


# --------------------------------------------------------------------------
# ground truth helpers (scale-safe: no 7.6M pair materialization)
# --------------------------------------------------------------------------

def parse_matched_list(cell: str) -> List[str]:
    """Parse a `matched_entity_ids` cell into a clean ID list.

    Empty/blank cells -> []. Whitespace is stripped; empty fragments dropped.
    """
    if cell is None:
        return []
    text = str(cell).strip()
    if not text:
        return []
    return [frag.strip() for frag in text.split(",") if frag.strip()]


def ground_truth_stats(
    gt_df: Optional[pd.DataFrame], cols: Dict[str, str]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Vectorized per-S1 match counts (one C-speed pass, no pair expansion).

    Returns (gt_stats, summary) where gt_stats has one row per S1 with
    ``n_matches, n_s2_matches, n_s3_matches`` + singleton/pattern flags, and
    summary holds corpus totals. Never builds the 7.6M positive-pair table.

    Assumes matched ids carry ``S2-``/``S3-`` prefixes (validated upstream);
    counts come from ``str.count`` so empty fragments can never inflate them.
    """
    c_s1, c_match = cols["gt_source1"], cols["gt_matched"]
    empty_stats = pd.DataFrame(
        columns=[c_s1, "n_matches", "n_s2_matches", "n_s3_matches",
                 "is_singleton", "is_s2_only", "is_s3_only", "is_mixed"]
    )
    empty_summary: Dict[str, Any] = {
        "n_s1_in_gt": 0,
        "n_positive_pairs": 0,
        "n_s1_s2": 0,
        "n_s1_s3": 0,
        "singleton_rate": 0.0,
        "multi_match_rate": 0.0,
        "pattern_counts": {},
        "total_counts": {},
    }
    if gt_df is None or gt_df.empty:
        return empty_stats, empty_summary
    if c_s1 not in gt_df.columns or c_match not in gt_df.columns:
        return empty_stats, empty_summary

    s1 = gt_df[c_s1].astype(str)
    matched = gt_df[c_match].astype(str)
    # C-speed prefix counts; each matched id contributes exactly one prefix.
    n_s2 = matched.str.count("S2-").fillna(0).astype(np.int64)
    n_s3 = matched.str.count("S3-").fillna(0).astype(np.int64)
    n_matches = (n_s2 + n_s3).astype(np.int64)

    gt_stats = pd.DataFrame({
        c_s1: s1.to_numpy(),
        "n_matches": n_matches.to_numpy(),
        "n_s2_matches": n_s2.to_numpy(),
        "n_s3_matches": n_s3.to_numpy(),
    })
    gt_stats["is_singleton"] = (gt_stats["n_matches"] == 0).astype(np.int64)
    gt_stats["is_s2_only"] = (
        (gt_stats["n_s2_matches"] > 0) & (gt_stats["n_s3_matches"] == 0)
    ).astype(np.int64)
    gt_stats["is_s3_only"] = (
        (gt_stats["n_s3_matches"] > 0) & (gt_stats["n_s2_matches"] == 0)
    ).astype(np.int64)
    gt_stats["is_mixed"] = (
        (gt_stats["n_s2_matches"] > 0) & (gt_stats["n_s3_matches"] > 0)
    ).astype(np.int64)

    n_s1 = int(len(gt_stats))
    n_pos = int(gt_stats["n_matches"].sum())
    n_s1_s2 = int(gt_stats["n_s2_matches"].sum())
    n_s1_s3 = int(gt_stats["n_s3_matches"].sum())
    n_singleton = int(gt_stats["is_singleton"].sum())

    def _bucket(n: int) -> str:
        if n == 0:
            return "0 matches"
        if n == 1:
            return "1 match"
        if n == 2:
            return "2 matches"
        if n == 3:
            return "3 matches"
        return "4+ matches"

    total_counts = gt_stats["n_matches"].map(_bucket).value_counts().to_dict()
    pattern = np.select(
        [gt_stats["is_singleton"] == 1, gt_stats["is_s2_only"] == 1,
         gt_stats["is_s3_only"] == 1, gt_stats["is_mixed"] == 1],
        ["singleton (no match)", "S2-only", "S3-only", "mixed S2+S3"],
        default="other",
    )
    pattern_counts = pd.Series(pattern).value_counts().to_dict()

    summary = {
        "n_s1_in_gt": n_s1,
        "n_positive_pairs": n_pos,
        "n_s1_s2": n_s1_s2,
        "n_s1_s3": n_s1_s3,
        "singleton_rate": float(n_singleton / n_s1) if n_s1 else 0.0,
        "multi_match_rate": float((gt_stats["n_matches"] >= 2).mean()) if n_s1 else 0.0,
        "pattern_counts": {str(k): int(v) for k, v in pattern_counts.items()},
        "total_counts": {str(k): int(v) for k, v in total_counts.items()},
    }
    return gt_stats, summary


def parse_anchor_truth(
    gt_df: Optional[pd.DataFrame],
    anchors: Collection[str],
    cols: Dict[str, str],
) -> Dict[str, List[str]]:
    """Parse ground truth ONLY for a small anchor set (sampled, closed world).

    Filters the 2.2M-row truth table to ``anchors`` via a vectorized ``isin``,
    then parses just those cells. Every anchor gets an entry (``[]`` when the
    S1 has no row or no matches) so callers can treat absence as singleton.
    """
    c_s1, c_match = cols["gt_source1"], cols["gt_matched"]
    anchor_list = sorted({str(a) for a in (anchors or []) if str(a)})
    out: Dict[str, List[str]] = {a: [] for a in anchor_list}
    if gt_df is None or gt_df.empty or not anchor_list:
        return out
    if c_s1 not in gt_df.columns or c_match not in gt_df.columns:
        return out
    anchor_set = set(anchor_list)
    try:
        mask = gt_df[c_s1].astype(str).isin(anchor_set)
    except Exception:
        return out
    sub = gt_df.loc[mask, [c_s1, c_match]]
    if sub.empty:
        return out
    for s1_id, cell in zip(
        sub[c_s1].astype(str).tolist(), sub[c_match].astype(str).tolist()
    ):
        out[str(s1_id)] = parse_matched_list(cell)
    return out


def anchor_truth_to_basic(
    anchor_truth: Dict[str, List[str]],
) -> pd.DataFrame:
    """Convert a SMALL anchor-truth dict to a basic pair table.

    Output columns: [source1_entity_id, candidate_entity_id, source_pair].
    Sorted by (s1, candidate) for deterministic artifacts.
    """
    rows: List[Tuple[str, str, str]] = []
    for s1_id in sorted(anchor_truth.keys()):
        for mid in anchor_truth[s1_id]:
            m = str(mid)
            if m.startswith("S2-"):
                pair = "S1_S2"
            elif m.startswith("S3-"):
                pair = "S1_S3"
            else:
                pair = "S1_OTHER"
            rows.append((str(s1_id), m, pair))
    rows.sort()
    return pd.DataFrame(
        rows, columns=["source1_entity_id", "candidate_entity_id", "source_pair"]
    )


# --------------------------------------------------------------------------
# file metadata (for run logging)
# --------------------------------------------------------------------------

def file_mtime_iso(path: str | Path) -> Optional[str]:
    try:
        ts = os.path.getmtime(path)
    except OSError:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def dataset_file_inventory(cfg: AppConfig) -> Dict[str, Dict[str, Optional[str]]]:
    """Map every expected dataset file -> {exists, mtime_utc, size_bytes}."""
    inv: Dict[str, Dict[str, Optional[str]]] = {}
    all_paths = {**cfg.train_paths(), **cfg.test_paths()}
    for key, p in all_paths.items():
        exists = p.exists()
        size: Optional[str] = None
        if exists:
            try:
                size = str(p.stat().st_size)
            except OSError:
                size = None
        inv[key] = {
            "path": str(p),
            "exists": "yes" if exists else "no",
            "mtime_utc": file_mtime_iso(p),
            "size_bytes": size,
        }
    return inv
