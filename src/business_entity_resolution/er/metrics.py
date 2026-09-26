"""Competition metric (macro F0.5 per Source-1 entity) + slice reports."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def f05_entity(truth: set[str], pred: set[str]) -> float:
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(truth & pred)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def per_entity_scores(truth: dict[str, set[str]], pred: dict[str, set[str]],
                      s1_ids: Iterable[str]) -> pd.Series:
    ids = list(s1_ids)
    return pd.Series([f05_entity(truth.get(s, set()), pred.get(s, set())) for s in ids],
                     index=ids, dtype=float)


def macro_f05(truth, pred, s1_ids) -> float:
    return float(per_entity_scores(truth, pred, s1_ids).mean())


def pr_summary(truth, pred, s1_ids) -> dict:
    """Pair-level precision/recall + entity-level singleton behaviour."""
    ids = list(s1_ids)
    tp = sum(len(truth.get(s, set()) & pred.get(s, set())) for s in ids)
    n_pred = sum(len(pred.get(s, set())) for s in ids)
    n_true = sum(len(truth.get(s, set())) for s in ids)
    singles = [s for s in ids if not truth.get(s)]
    return {
        "macro_f05": macro_f05(truth, pred, ids),
        "pair_precision": tp / n_pred if n_pred else 1.0,
        "pair_recall": tp / n_true if n_true else 1.0,
        "singleton_acc": float(np.mean([not pred.get(s) for s in singles])) if singles else float("nan"),
        "n_entities": len(ids),
    }


def slice_report(scores: pd.Series, slices: pd.DataFrame) -> pd.DataFrame:
    """For each boolean slice column: size, mean F0.5, and *points lost*.

    points_lost = sum over the slice of (1 - F0.5) / N_total * 100, i.e. how many
    leaderboard points (out of 100) this slice costs us. Sort by it to decide
    what to fix next — a big, mediocre slice beats a tiny, terrible one.
    """
    n_total = len(scores)
    rows = []
    for col in slices.columns:
        mask = slices[col].reindex(scores.index).fillna(False).astype(bool)
        n = int(mask.sum())
        if n == 0:
            continue
        s = scores[mask]
        rows.append({"slice": col, "n": n, "share": n / n_total, "f05": s.mean(),
                     "points_lost": (1 - s).sum() / n_total * 100})
    return pd.DataFrame(rows).sort_values("points_lost", ascending=False).reset_index(drop=True)
