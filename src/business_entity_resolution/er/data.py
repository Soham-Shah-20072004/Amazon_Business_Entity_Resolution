"""Load one split (train or test) into a single record table with text views."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..io import parse_matched_list, read_tsv
from . import text as T


@dataclass
class Split:
    name: str
    records: pd.DataFrame            # all S1/S2/S3 rows, index = entity_id
    s1_ids: list[str]
    truth: dict[str, set[str]] = field(default_factory=dict)  # empty for test

    @property
    def has_labels(self) -> bool:
        return bool(self.truth)


def _prepare(df: pd.DataFrame, source: int) -> pd.DataFrame:
    df = df.copy()
    df["source"] = source
    df["name_norm"] = df["business_name"].map(T.norm)
    df["addr_norm"] = df["business_address"].map(T.norm)
    df["name_ascii"] = df["name_norm"].map(T.to_ascii)
    df["addr_ascii"] = df["addr_norm"].map(T.to_ascii)
    df["name_canon"] = df["name_ascii"].map(T.canon)
    df["addr_canon"] = df["addr_ascii"].map(T.canon)
    df["name_core"] = df["name_canon"].map(T.core_name)
    df["name_acr"] = df["name_canon"].map(T.acronym)
    df["addr_nums"] = df["addr_ascii"].map(T.numbers)
    df["addr_post"] = [T.postcodes(n, a) for n, a in zip(df["addr_nums"], df["addr_ascii"])]
    df["country_norm"] = df["country"].map(T.norm)
    raw = df["business_name"] + " " + df["business_address"]
    df["non_ascii"] = raw.map(lambda s: any(ord(c) > 127 for c in s)).astype(np.int8)
    return df


def load_split(data_root: str | Path, split: str) -> Split:
    root = Path(data_root) / split
    tables = []
    for src in (1, 2, 3):
        df = read_tsv(root / f"{split}_source{src}.tsv")
        tables.append(_prepare(df, src))
    records = pd.concat(tables, ignore_index=True).set_index("entity_id", drop=False)
    if records.index.duplicated().any():
        dups = records.index[records.index.duplicated()].unique()[:5].tolist()
        raise ValueError(f"duplicate entity ids across sources: {dups}")
    s1_ids = records.index[records["source"] == 1].tolist()
    truth: dict[str, set[str]] = {}
    gt_path = root / f"{split}_ground_truth.tsv"
    if gt_path.exists():
        gt = read_tsv(gt_path)
        truth = {s1: set(parse_matched_list(m))
                 for s1, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"])}
        for s1 in s1_ids:
            truth.setdefault(s1, set())
    return Split(split, records, s1_ids, truth)


def positive_pairs(split: Split) -> pd.DataFrame:
    rows = [(s1, c) for s1, cs in split.truth.items() for c in cs]
    return pd.DataFrame(rows, columns=["s1", "cand"])
