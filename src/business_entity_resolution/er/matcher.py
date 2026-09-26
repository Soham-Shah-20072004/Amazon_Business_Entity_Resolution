"""LightGBM pair classifier with grouped, stratified K-fold (no S1 leakage).

Training pairs = the blocker's own candidates, labelled by ground truth. The
negatives are therefore exactly the look-alikes the model must reject at test
time (hard negatives by construction, same distribution as inference).
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .data import Split

DEFAULT_PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=20,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    verbose=-1, seed=42, num_threads=0,
)


def add_labels(split: Split, feats: pd.DataFrame) -> pd.DataFrame:
    truth_keys = {f"{s}|{c}" for s, cs in split.truth.items() for c in cs}
    feats["label"] = (feats["s1"] + "|" + feats["cand"]).isin(truth_keys).astype(np.int8)
    return feats


def entity_strata(split: Split) -> pd.Series:
    """Country x match-count bucket, used to stratify folds."""
    r = split.records
    n = pd.Series({s: min(len(split.truth.get(s, ())), 2) for s in split.s1_ids})
    return r.loc[split.s1_ids, "country_norm"].astype(str) + "_" + n.astype(str)


def fold_of_s1(split: Split, n_folds: int = 5, seed: int = 42) -> pd.Series:
    strata = entity_strata(split)
    ids = np.array(split.s1_ids)
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = pd.Series(-1, index=ids)
    for k, (_, va) in enumerate(skf.split(ids, strata.to_numpy(), groups=ids)):
        fold.iloc[va] = k
    return fold


def cross_validate(feats: pd.DataFrame, cols: list[str], fold: pd.Series,
                   params: dict | None = None, rounds: int = 2000):
    params = {**DEFAULT_PARAMS, **(params or {})}
    f = fold.reindex(feats["s1"]).to_numpy()
    oof = np.zeros(len(feats), np.float32)
    best_iters, importances = [], []
    X, y = feats[cols], feats["label"].to_numpy()
    for k in sorted(set(f)):
        tr, va = f != k, f == k
        dtr = lgb.Dataset(X[tr], y[tr])
        dva = lgb.Dataset(X[va], y[va], reference=dtr)
        m = lgb.train(params, dtr, rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        best_iters.append(m.best_iteration)
        importances.append(pd.Series(m.feature_importance("gain"), index=cols))
    imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    return oof, best_iters, imp


def fit_full(feats: pd.DataFrame, cols: list[str], n_rounds: int, params: dict | None = None):
    params = {**DEFAULT_PARAMS, **(params or {})}
    return lgb.train(params, lgb.Dataset(feats[cols], feats["label"].to_numpy()), n_rounds)
