"""LightGBM helpers: grouped out-of-fold (OOF) predictions and final fits.

Folds are assigned per S1 row, so every candidate of one S1 lands in the same
fold: an S1 is always scored by a model that never saw it (no leakage), and
OOF probabilities for *all* sampled S1s are available to tune thresholds.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

PRE_PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_child_samples=50,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=7)
MATCH_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_child_samples=40,
                    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                    verbose=-1, seed=42)


def fold_of(i: np.ndarray, n_folds: int, seed: int = 42) -> np.ndarray:
    """Deterministic fold per S1 row id (hash-like, so it is stable across runs)."""
    return ((i.astype(np.int64) * 2654435761 + seed) % 2_147_483_647 % n_folds).astype(np.int8)


def oof(df: pd.DataFrame, cols: list[str], params: dict, n_folds: int = 5, rounds: int = 3000,
        workers: int = 4, log=print):
    params = {**params, "num_threads": workers}
    fold = fold_of(df["i"].to_numpy(), n_folds)
    X, y = df[cols], df["label"].to_numpy()
    pred = np.zeros(len(df), np.float32)
    iters, imps = [], []
    for k in range(n_folds):
        tr, va = fold != k, fold == k
        m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), rounds,
                      valid_sets=[lgb.Dataset(X[va], y[va])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        pred[va] = m.predict(X[va], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        imps.append(pd.Series(m.feature_importance("gain"), index=cols))
        log(f"    fold {k}: best iteration {m.best_iteration}")
    imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
    return pred, iters, imp


def fit(df: pd.DataFrame, cols: list[str], params: dict, rounds: int, workers: int = 4):
    return lgb.train({**params, "num_threads": workers},
                     lgb.Dataset(df[cols], df["label"].to_numpy()), max(10, rounds))
