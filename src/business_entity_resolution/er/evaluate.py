"""Competition metric, decision rule and slice report on integer pair arrays.

macro F0.5 per S1 (singletons: 1.0 for an empty prediction, else 0.0), fully
vectorised so it can be evaluated hundreds of times while tuning thresholds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None


def _in_sorted(sorted_keys: np.ndarray, x: np.ndarray) -> np.ndarray:
    if len(sorted_keys) == 0:
        return np.zeros(len(x), bool)
    pos = np.minimum(np.searchsorted(sorted_keys, x), len(sorted_keys) - 1)
    return sorted_keys[pos] == x


def per_entity_f05(queries: np.ndarray, ti: np.ndarray, tj: np.ndarray,
                   pi: np.ndarray, pj: np.ndarray, n: int) -> np.ndarray:
    """F0.5 for every S1 in `queries` (sorted). Truth/pred are (i, j) pair arrays."""
    nq = len(queries)
    tpos, ppos = np.searchsorted(queries, ti), np.searchsorted(queries, pi)
    n_true = np.bincount(tpos, minlength=nq)
    n_pred = np.bincount(ppos, minlength=nq)
    hit = _in_sorted(np.sort(ti.astype(np.int64) * n + tj), pi.astype(np.int64) * n + pj)
    tp = np.bincount(ppos[hit], minlength=nq).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(n_pred > 0, tp / np.maximum(n_pred, 1), 0.0)
        rec = np.where(n_true > 0, tp / np.maximum(n_true, 1), 0.0)
        f = np.where(tp > 0, 1.25 * prec * rec / (0.25 * prec + rec), 0.0)
    return np.where(n_true == 0, (n_pred == 0).astype(float), f)


def summary(queries, ti, tj, pi, pj, n) -> dict:
    f = per_entity_f05(queries, ti, tj, pi, pj, n)
    hit = _in_sorted(np.sort(ti.astype(np.int64) * n + tj), pi.astype(np.int64) * n + pj)
    single = np.bincount(np.searchsorted(queries, ti), minlength=len(queries)) == 0
    predicted = np.bincount(np.searchsorted(queries, pi), minlength=len(queries)) > 0
    return {"macro_f05": float(f.mean()),
            "pair_precision": float(hit.mean()) if len(pi) else 1.0,
            "pair_recall": float(hit.sum() / max(1, len(ti))),
            "singleton_share": float(single.mean()),
            "singleton_acc": float((~predicted[single]).mean()) if single.any() else float("nan"),
            "pred_per_s1": float(len(pi) / max(1, len(queries)))}


def decide(df: pd.DataFrame, t2: float, t3: float | None = None, one_owner: bool = True,
           prob: str = "p") -> np.ndarray:
    """Boolean mask of pairs kept: p >= threshold (per source), and with
    one_owner each candidate record keeps only its highest-probability S1."""
    t3 = t2 if t3 is None else t3
    p = df[prob].to_numpy()
    keep = p >= np.where(df["src"].to_numpy() == 2, t2, t3)
    if one_owner and keep.any():
        idx = np.flatnonzero(keep)
        best = pd.Series(p[idx]).groupby(df["j"].to_numpy()[idx]).transform("max").to_numpy()
        keep[idx[p[idx] < best]] = False
    return keep


def owner_mask(df: pd.DataFrame, p: np.ndarray) -> np.ndarray:
    """True where this S1 is the highest-probability claimant of the record j."""
    best = pd.Series(p).groupby(df["j"].to_numpy()).transform("max").to_numpy()
    return p >= best


def decide_expected(df: pd.DataFrame, lam: float = 0.2, power: float = 1.0, one_owner: bool = True,
                    prob: str = "p") -> np.ndarray:
    """Per-S1 decision instead of one global threshold: sort the S1's candidates by
    p and keep the top k that maximise its expected F0.5,
        E[F0.5 | top k] ~ 1.25 * sum(p_1..p_k) / (0.25 * (sum(all p) + lam) + k),
    or keep nothing when P(no true match) = prod(1 - p) * exp(-lam) is higher.
    lam = expected true matches the candidate list does not contain (blocking
    misses); power recalibrates p (p ** power). Both are tuned on OOF scores."""
    p = np.clip(df[prob].to_numpy().astype(np.float64) ** power, 0.0, 1 - 1e-9)
    idx = np.flatnonzero(owner_mask(df, p)) if one_owner else np.arange(len(df))
    keep = np.zeros(len(df), bool)
    if len(idx) == 0:
        return keep
    i, q = df["i"].to_numpy()[idx], p[idx]
    o = np.lexsort((-q, i))
    i_s, q_s = i[o], q[o]
    new = np.r_[True, i_s[1:] != i_s[:-1]]
    starts, g = np.flatnonzero(new), np.cumsum(new) - 1
    k = np.arange(len(q_s)) - starts[g] + 1
    cs = np.cumsum(q_s)
    top_sum = cs - (cs - q_s)[starts][g]
    total = np.add.reduceat(q_s, starts)[g]
    ef = 1.25 * top_sum / (0.25 * (total + lam) + k)
    ef_empty = np.exp(np.add.reduceat(np.log1p(-q_s), starts) - lam)
    best = np.maximum.reduceat(ef, starts)
    k_best = np.minimum.reduceat(np.where(ef >= best[g], k, np.iinfo(np.int64).max), starts)
    take = (best[g] > ef_empty[g]) & (k <= k_best[g])
    keep[idx[o[take]]] = True
    return keep


def tune_expected(df: pd.DataFrame, queries, ti, tj, n, prob="p",
                  lams=(0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0), powers=(0.8, 1.0, 1.25, 1.5)) -> pd.DataFrame:
    rows = []
    i, j = df["i"].to_numpy(), df["j"].to_numpy()
    for one_owner in (False, True):
        for lam in lams:
            for pw in powers:
                k = decide_expected(df, lam, pw, one_owner, prob)
                rows.append({"lam": lam, "power": pw, "one_owner": one_owner,
                             "macro_f05": per_entity_f05(queries, ti, tj, i[k], j[k], n).mean()})
    return pd.DataFrame(rows).sort_values("macro_f05", ascending=False).reset_index(drop=True)


def apply_rule(df: pd.DataFrame, rule: dict, prob: str) -> np.ndarray:
    """rule from tune() (t2, t3, one_owner) or tune_expected() (lam, power, one_owner)."""
    if "lam" in rule:
        return decide_expected(df, rule["lam"], rule["power"], bool(rule["one_owner"]), prob)
    return decide(df, rule["t2"], rule["t3"], bool(rule["one_owner"]), prob)


def tune(df: pd.DataFrame, queries, ti, tj, n, per_source=False, prob="p",
         grid=None) -> pd.DataFrame:
    grid = np.round(np.arange(0.05, 0.96, 0.025), 3) if grid is None else grid
    rows = []
    i, j = df["i"].to_numpy(), df["j"].to_numpy()
    for one_owner in (False, True):
        pairs = [(a, b) for a in grid for b in grid] if per_source else [(t, t) for t in grid]
        for t2, t3 in pairs:
            k = decide(df, t2, t3, one_owner, prob)
            rows.append({"t2": t2, "t3": t3, "one_owner": one_owner,
                         "macro_f05": per_entity_f05(queries, ti, tj, i[k], j[k], n).mean()})
    return pd.DataFrame(rows).sort_values("macro_f05", ascending=False).reset_index(drop=True)


def rank_in_list(df: pd.DataFrame, col: str) -> np.ndarray:
    """1 = highest `col` within each S1 (column i); ties by order of appearance,
    as pandas method="first". Polars when installed (multi-threaded), same result."""
    if pl is not None:
        return (pl.DataFrame({"i": df["i"].to_numpy(), "v": df[col].to_numpy()})
                .select(pl.col("v").rank("ordinal", descending=True).over("i")).to_series().to_numpy())
    return df.groupby("i", sort=False)[col].rank(ascending=False, method="first").to_numpy()


def focus_set(df: pd.DataFrame, min_p: float, top_k: int, prob: str = "p") -> np.ndarray:
    """Cascade step: keep pairs whose first-stage matcher probability is at least
    min_p and that rank in the top_k of their S1. Pairs below ~1% are never
    predicted anyway (thresholds sit near 0.5), so this only shrinks the set the
    next model (and candidate_pairs.tsv) has to cover."""
    return (df[prob].to_numpy() >= min_p) & (rank_in_list(df, prob) <= top_k)


def slice_report(scores: np.ndarray, slices: dict[str, np.ndarray]) -> pd.DataFrame:
    """points_lost = share of the total score (out of 100) each slice costs."""
    n = len(scores)
    rows = []
    for name, mask in slices.items():
        m = np.asarray(mask, bool)
        if m.sum() == 0:
            continue
        rows.append({"slice": name, "n": int(m.sum()), "share": m.mean(), "f05": scores[m].mean(),
                     "points_lost": (1 - scores[m]).sum() / n * 100})
    return pd.DataFrame(rows).sort_values("points_lost", ascending=False).reset_index(drop=True)
