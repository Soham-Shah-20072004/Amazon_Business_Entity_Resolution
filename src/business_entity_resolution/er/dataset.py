"""Load a prepared split (see prepare.py) with integer row ids.

Records keep a RangeIndex; pairs everywhere are (i, j) row numbers, which is
far lighter than string ids at 12M records / 100M+ candidate pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Split:
    name: str
    rec: pd.DataFrame            # all records, RangeIndex
    query: np.ndarray            # S1 row numbers we generate matches for
    truth: pd.DataFrame | None   # positive pairs (i, j) restricted to `query`, or None

    @property
    def has_labels(self) -> bool:
        return self.truth is not None

    @property
    def ids(self) -> np.ndarray:
        return self.rec["entity_id"].to_numpy()


def load_split(work: str | Path, split: str, s1_sample: float | int | None = None,
               seed: int = 42, columns: list[str] | None = None) -> Split:
    """s1_sample: keep a random subset of S1 *queries* (fraction if < 1, else a
    count). The S2/S3 pool is always complete, so retrieval difficulty and the
    look-alike negatives stay realistic."""
    root = Path(work) / split
    rec = pd.read_parquet(root / "records.parquet", columns=columns)
    s1 = np.flatnonzero(rec["source"].to_numpy() == 1)
    if s1_sample:
        n = int(round(len(s1) * s1_sample)) if s1_sample < 1 else min(int(s1_sample), len(s1))
        s1 = np.sort(np.random.default_rng(seed).choice(s1, n, replace=False))
    truth = None
    tpath = root / "truth.parquet"
    if tpath.exists():
        t = pd.read_parquet(tpath)
        row_of = pd.Series(np.arange(len(rec)), index=rec["entity_id"].to_numpy())
        t = pd.DataFrame({"i": row_of.reindex(t["s1"]).to_numpy(),
                          "j": row_of.reindex(t["cand"]).to_numpy()})
        bad = t.isna().any(axis=1)
        if bad.any():
            raise ValueError(f"{int(bad.sum())} ground-truth ids not found in records")
        t = t.astype(np.int64)
        truth = t[np.isin(t["i"].to_numpy(), s1)].reset_index(drop=True)
    return Split(split, rec, s1, truth)
