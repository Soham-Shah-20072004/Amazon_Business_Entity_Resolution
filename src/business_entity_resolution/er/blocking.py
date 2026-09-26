"""Multi-pass candidate generation (union of independent blockers).

Blockers (each run separately for S1->S2 and S1->S3 so one source can't crowd
out the other):
  name   char n-gram TF-IDF on ascii name, top-k per S1
  addr   char n-gram TF-IDF on ascii address, top-k per S1
  full   char n-gram TF-IDF on name + address, top-k per S1
  word   word TF-IDF on core name (rare-token retrieval), top-k per S1
  rev    reverse direction: for every S2/S3 record, its top-k S1s by `full`
         (rescues S1s whose forward list is crowded by look-alikes)
  exact  identical core name (legal suffixes dropped), only for blocks <= cap

TF-IDF is fit on the split's own records without labels (transductive,
allowed on test since it uses no ground truth and no external data).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import Split

BLOCKERS = ("name", "addr", "full", "word", "rev", "exact")


@dataclass
class BlockingConfig:
    k_name: int = 10
    k_addr: int = 10
    k_full: int = 10
    k_word: int = 5
    k_rev: int = 3
    exact_max_block: int = 30
    mem_budget: float = 3e8   # bytes per dense score chunk


def _vec(kind: str) -> TfidfVectorizer:
    if kind == "word":
        return TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b",
                               sublinear_tf=True, dtype=np.float32)
    ngram = (2, 4) if kind == "name" else (3, 5)
    return TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, min_df=2,
                           sublinear_tf=True, dtype=np.float32)


def build_matrices(split: Split) -> dict[str, sp.csr_matrix]:
    r = split.records
    texts = {
        "name": r["name_ascii"],
        "addr": r["addr_ascii"],
        "full": r["name_ascii"] + " | " + r["addr_ascii"],
        "word": r["name_core"],
    }
    out = {}
    for kind, col in texts.items():
        v = _vec(kind)
        out[kind] = v.fit_transform(col.fillna("").tolist()).tocsr()
    return out


def topk(A: sp.csr_matrix, B: sp.csr_matrix, k: int, mem_budget: float):
    """Row-wise top-k cosine of A against B (rows L2-normalised by TF-IDF)."""
    if k <= 0 or A.shape[0] == 0 or B.shape[0] == 0:
        return np.empty(0, int), np.empty(0, int), np.empty(0, np.float32)
    k = min(k, B.shape[0])
    BT = B.T.tocsc()
    chunk = max(1, int(mem_budget / (4 * B.shape[0])))
    ri, ci, sc = [], [], []
    for s in range(0, A.shape[0], chunk):
        S = (A[s:s + chunk] @ BT).toarray()
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(S, idx, axis=1)
        rows = np.repeat(np.arange(s, s + S.shape[0]), k)
        keep = vals.ravel() > 0
        ri.append(rows[keep]); ci.append(idx.ravel()[keep]); sc.append(vals.ravel()[keep])
    return np.concatenate(ri), np.concatenate(ci), np.concatenate(sc)


def pair_cosine(M: sp.csr_matrix, ii: np.ndarray, jj: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    out = np.empty(len(ii), np.float32)
    for s in range(0, len(ii), chunk):
        a, b = M[ii[s:s + chunk]], M[jj[s:s + chunk]]
        out[s:s + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


def generate_candidates(split: Split, cfg: BlockingConfig,
                        mats: dict[str, sp.csr_matrix] | None = None) -> pd.DataFrame:
    """Return one row per unique (s1, cand) with blocker flags + TF-IDF cosines."""
    r = split.records
    mats = mats if mats is not None else build_matrices(split)
    pos = {s: np.flatnonzero(r["source"].to_numpy() == s) for s in (1, 2, 3)}
    s1p = pos[1]
    found: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {b: [] for b in BLOCKERS}

    for tgt in (2, 3):
        tp = pos[tgt]
        for kind, k in (("name", cfg.k_name), ("addr", cfg.k_addr),
                        ("full", cfg.k_full), ("word", cfg.k_word)):
            M = mats[kind]
            ri, ci, _ = topk(M[s1p], M[tp], k, cfg.mem_budget)
            found[kind].append((s1p[ri], tp[ci]))
        M = mats["full"]
        ri, ci, _ = topk(M[tp], M[s1p], cfg.k_rev, cfg.mem_budget)
        found["rev"].append((s1p[ci], tp[ri]))

    # exact core-name blocks (skip generic names shared by many records)
    core = r["name_core"].to_numpy()
    src = r["source"].to_numpy()
    groups = pd.Series(np.arange(len(r))).groupby(core).indices
    ex_i, ex_j = [], []
    for key, members in groups.items():
        if not key or len(members) > cfg.exact_max_block:
            continue
        a = members[src[members] == 1]
        b = members[src[members] != 1]
        if len(a) and len(b):
            ex_i.append(np.repeat(a, len(b))); ex_j.append(np.tile(b, len(a)))
    if ex_i:
        found["exact"].append((np.concatenate(ex_i), np.concatenate(ex_j)))

    n = len(r)
    keys_by_blocker = {
        b: np.unique(np.concatenate([ii.astype(np.int64) * n + jj for ii, jj in found[b]]))
        if found[b] else np.empty(0, np.int64)
        for b in BLOCKERS
    }
    keys = np.unique(np.concatenate(list(keys_by_blocker.values())))
    cand = pd.DataFrame({"i": keys // n, "j": keys % n})
    for b in BLOCKERS:
        cand[f"blk_{b}"] = np.isin(keys, keys_by_blocker[b], assume_unique=True).astype(np.int8)
    cand["n_blockers"] = cand[[f"blk_{b}" for b in BLOCKERS]].sum(axis=1).astype(np.int8)
    ii, jj = cand["i"].to_numpy(), cand["j"].to_numpy()
    for kind in ("name", "addr", "full", "word"):
        cand[f"cos_{kind}"] = pair_cosine(mats[kind], ii, jj)
    ids = r.index.to_numpy()
    cand.insert(0, "s1", ids[ii])
    cand.insert(1, "cand", ids[jj])
    return cand


def blocking_report(split: Split, cand: pd.DataFrame) -> dict:
    """Recall ceiling + candidate burden (only meaningful with labels)."""
    per_s1 = cand.groupby("s1").size().reindex(split.s1_ids, fill_value=0)
    rep = {"pairs": len(cand), "avg_per_s1": per_s1.mean(),
           "p50_per_s1": per_s1.median(), "p95_per_s1": per_s1.quantile(.95),
           "max_per_s1": per_s1.max()}
    if split.has_labels:
        truth_pairs = {(s, c) for s, cs in split.truth.items() for c in cs}
        got = set(zip(cand["s1"], cand["cand"]))
        rep["pair_recall"] = len(truth_pairs & got) / max(1, len(truth_pairs))
        for b in BLOCKERS:
            sub = cand[cand[f"blk_{b}"] == 1]
            only = cand[(cand[f"blk_{b}"] == 1) & (cand["n_blockers"] == 1)]
            rep[f"recall_{b}"] = len(truth_pairs & set(zip(sub["s1"], sub["cand"]))) / max(1, len(truth_pairs))
            rep[f"unique_{b}"] = len(truth_pairs & set(zip(only["s1"], only["cand"])))
    return rep
