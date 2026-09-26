"""Turn pair probabilities into per-S1 match lists, tuned on macro F0.5.

Rules:
  * keep a pair only if p >= t (per-source thresholds optional)
  * one_owner: S1 is deduplicated, so a candidate record can belong to at most
    one S1 — if several S1s claim it, only the highest-probability S1 keeps it
  * no forced top-1: an S1 with nothing above threshold gets an empty list
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .metrics import macro_f05


def decide(df: pd.DataFrame, t2: float, t3: float | None = None,
           one_owner: bool = True, prob_col: str = "p") -> dict[str, set[str]]:
    t3 = t2 if t3 is None else t3
    thr = np.where(df["src"].to_numpy() == 2, t2, t3)
    keep = df[df[prob_col].to_numpy() >= thr]
    if one_owner and len(keep):
        best = keep.groupby("cand")[prob_col].transform("max")
        keep = keep[keep[prob_col] >= best]
    out: dict[str, set[str]] = {}
    for s1, c in zip(keep["s1"], keep["cand"]):
        out.setdefault(s1, set()).add(c)
    return out


def tune(df: pd.DataFrame, truth, s1_ids, grid=None, per_source: bool = False,
         prob_col: str = "p") -> pd.DataFrame:
    grid = np.round(np.arange(0.10, 0.96, 0.025), 3) if grid is None else grid
    rows = []
    for one_owner in (False, True):
        pairs = itertools.product(grid, grid) if per_source else ((t, t) for t in grid)
        for t2, t3 in pairs:
            pred = decide(df, t2, t3, one_owner, prob_col)
            rows.append({"t2": t2, "t3": t3, "one_owner": one_owner,
                         "macro_f05": macro_f05(truth, pred, s1_ids)})
    return pd.DataFrame(rows).sort_values("macro_f05", ascending=False).reset_index(drop=True)
