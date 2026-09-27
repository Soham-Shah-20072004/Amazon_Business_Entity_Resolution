"""Load a prepared split (see prepare.py) with integer row ids.

Records keep a RangeIndex; pairs everywhere are (i, j) row numbers, which is
far lighter than string ids at 12M records / 100M+ candidate pairs. Text
columns stay Arrow-backed (compact, no per-string Python objects) and country
becomes a small integer code (`country_code`, names in `Split.countries`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RETRIEVAL_COLUMNS = ["entity_id", "source", "country_norm", "name_canon", "addr_canon",
                     "name_core", "name_skel", "addr_skel", "addr_post"]


@dataclass
class Split:
    name: str
    rec: pd.DataFrame            # all records, RangeIndex
    query: np.ndarray            # S1 row numbers we generate matches for
    truth: pd.DataFrame | None   # positive pairs (i, j) restricted to `query`, or None
    countries: list[str]         # country_code -> normalised country name

    @property
    def has_labels(self) -> bool:
        return self.truth is not None


def load_split(work: str | Path, split: str, s1_sample: float | int | None = None,
               seed: int = 42, columns: list[str] | None = None) -> Split:
    """s1_sample: keep a random subset of S1 *queries* (fraction if < 1, else a
    count). The S2/S3 pool is always complete, so retrieval difficulty and the
    look-alike negatives stay realistic."""
    root = Path(work) / split
    manifest = json.loads((root / "manifest.json").read_text())
    rec = pd.read_parquet(root / "records.parquet", columns=columns, dtype_backend="pyarrow")
    rec["source"] = rec["source"].to_numpy(dtype=np.int8)
    codes, names = pd.factorize(rec["country_norm"])
    rec["country_code"] = codes.astype(np.int16)
    rec = rec.drop(columns="country_norm")
    s1 = np.flatnonzero(rec["source"].to_numpy() == 1)
    if s1_sample:
        n = int(round(len(s1) * s1_sample)) if s1_sample < 1 else min(int(s1_sample), len(s1))
        s1 = np.sort(np.random.default_rng(seed).choice(s1, n, replace=False))
    truth = None
    tpath = root / "truth.parquet"
    if tpath.exists():
        t = pd.read_parquet(tpath)
        index = pd.Index(rec["entity_id"].to_numpy(dtype=object))
        i, j = index.get_indexer(t["s1"].to_numpy()), index.get_indexer(t["cand"].to_numpy())
        del index
        bad = (i < 0) | (j < 0)
        if bad.any():
            if not manifest.get("max_rows"):
                raise ValueError(f"{int(bad.sum())} ground-truth ids not found in records")
            print(f"  note: truncated smoke-test data, dropping {int(bad.sum()):,} truth pairs "
                  "whose records were not loaded")
        truth = pd.DataFrame({"i": i[~bad], "j": j[~bad]}).astype(np.int64)
        truth = truth[np.isin(truth["i"].to_numpy(), s1)].reset_index(drop=True)
    return Split(split, rec, s1, truth, [str(x) for x in names])


def take_rows(records: str | Path, rows: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """Columns of records.parquet for the given row numbers only (one column in
    memory at a time), indexed by row number. Used by the cached stages, which
    need text for a few million rows out of ~12M."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    rows = np.unique(np.asarray(rows, dtype=np.int64))
    pf = pq.ParquetFile(records)
    out = {}
    for c in columns:
        col = pf.read(columns=[c]).column(0).take(rows)
        if pa.types.is_string(col.type) or pa.types.is_large_string(col.type):
            col = pc.fill_null(col, "")
        out[c] = col.to_numpy(zero_copy_only=False)
    return pd.DataFrame(out, index=rows)
