"""Scalable candidate generation (blocking) for ~2M S1 x ~10M S2/S3 records.

Two ideas make it tractable:
  1. Matching is within-country (no true train pair crosses countries), so each
     country is an independent, smaller problem. Countries are an open set:
     France (test only) is simply one more group.
  2. No blocker compares all pairs; each looks at a small neighbourhood per S1:
       ann_<view>  approximate nearest neighbours (FAISS IVF) on d-dim vectors
                   = char n-gram TF-IDF compressed by truncated SVD ("LSA").
                   views: name, addr, skel (sound skeleton of name + address,
                   bridges Latin <-> transliterated Devanagari/Gujarati)
       rare        shared rare tokens: an inverted index written as a sparse
                   product, IDF-weighted, only tokens seen in <= rare_max_df
                   pool records of that country/source
       exact       identical core name (legal suffixes dropped), block <= cap
     Every blocker runs separately for S2 and S3 so neither crowds out the other.

The output has one row per unique (i, j) with blocker flags plus similarity
scores for *every* view (not only the view that retrieved the pair), which the
downstream pre-ranker and matcher use as features.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from multiprocessing import Pool

import faiss
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

from .dataset import Split
from .text import LEGAL_TOKENS

HASH_BITS = 24


@dataclass
class RetrievalConfig:
    views: tuple[str, ...] = ("name", "addr", "skel")
    k_ann: int = 15            # neighbours per S1, per view, per target source
    k_rare: int = 15           # rare-token candidates per S1, per target source
    rare_max_df: int = 100     # "rare" = appears in <= this many pool records
    exact_max_block: int = 50  # skip exact-name blocks bigger than this
    svd_dim: int = 128
    fit_sample: int = 400_000  # rows used to fit TF-IDF vocab + SVD
    nprobe: int = 32           # IVF cells searched per query (recall vs speed)
    workers: int = 8
    seed: int = 42
    blockers: tuple[str, ...] = field(init=False)

    def __post_init__(self):
        self.blockers = tuple(f"ann_{v}" for v in self.views) + ("rare", "exact")


# ---------------------------------------------------------------- LSA views

def view_text(rec: pd.DataFrame, view: str) -> pd.Series:
    if view == "name":
        return rec["name_canon"]
    if view == "addr":
        return rec["addr_canon"]
    if view == "skel":
        return rec["name_skel"] + " | " + rec["addr_skel"]
    raise ValueError(view)


def _vectorizer(view: str) -> TfidfVectorizer:
    ngram = (3, 4) if view == "addr" else (2, 4)
    return TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, min_df=3,
                           max_features=1 << 18, sublinear_tf=True, dtype=np.float32)


class LSA:
    def __init__(self, view: str, dim: int, seed: int):
        self.view, self.dim, self.seed = view, dim, seed

    def fit(self, texts: list[str]) -> "LSA":
        self.vec = _vectorizer(self.view).fit(texts)
        X = self.vec.transform(texts)
        svd = TruncatedSVD(self.dim, algorithm="randomized", n_iter=4, random_state=self.seed)
        svd.fit(X)
        self.comp = np.ascontiguousarray(svd.components_.T, dtype=np.float32)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        Z = np.asarray(self.vec.transform(texts) @ self.comp, dtype=np.float32)
        faiss.normalize_L2(Z)       # zero rows (empty text) stay zero
        return Z


_W: dict = {}


def _init_worker(obj):
    _W["obj"] = obj


def _lsa_chunk(texts):
    return _W["obj"].transform(texts)


def _hash_chunk(docs):
    return _W["obj"].transform(docs)


def _chunks(seq: list, size: int) -> list[list]:
    return [seq[s:s + size] for s in range(0, len(seq), size)]


def parallel_map(func, obj, chunks, workers):
    if workers <= 1 or len(chunks) == 1:
        _init_worker(obj)
        return [func(c) for c in chunks]
    with Pool(workers, initializer=_init_worker, initargs=(obj,)) as pool:
        return pool.map(func, chunks, chunksize=1)


def embed_view(rec: pd.DataFrame, rows: np.ndarray, view: str, cfg: RetrievalConfig,
               log=print) -> np.ndarray:
    """Return an (n_rec, dim) float32 array; rows outside `rows` stay zero."""
    t = time.time()
    texts = view_text(rec, view)
    rng = np.random.default_rng(cfg.seed)
    fit_rows = rng.choice(rows, min(cfg.fit_sample, len(rows)), replace=False)
    lsa = LSA(view, cfg.svd_dim, cfg.seed).fit(texts.iloc[fit_rows].tolist())
    Z = np.zeros((len(rec), lsa.comp.shape[1]), np.float32)
    parts = parallel_map(_lsa_chunk, lsa, _chunks(texts.iloc[rows].tolist(), 50_000), cfg.workers)
    Z[rows] = np.vstack(parts)
    log(f"    view {view}: vocab {len(lsa.vec.vocabulary_):,} -> {lsa.comp.shape[1]}-d, "
        f"{len(rows):,} rows embedded in {time.time() - t:.0f}s")
    return Z


def ann_search(Zq: np.ndarray, Zp: np.ndarray, k: int, nprobe: int, seed: int):
    n, d = Zp.shape
    k = min(k, n)
    if n <= 20_000:
        index = faiss.IndexFlatIP(d)
    else:
        nlist = int(max(16, min(4 * math.sqrt(n), n / 40)))
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
        rng = np.random.default_rng(seed)
        index.train(Zp[rng.choice(n, min(n, nlist * 64), replace=False)])
        index.nprobe = nprobe
    index.add(Zp)
    return index.search(Zq, k)


# ---------------------------------------------------------------- rare tokens

def rare_analyzer(doc: str) -> list[str]:
    name, addr, post = doc.split("\t")
    toks = [f"n:{t}" for t in set(name.split()) if len(t) >= 3 and not t.isdigit()
            and t not in LEGAL_TOKENS]
    toks += [f"a:{t}" for t in set(addr.split()) if len(t) >= 3 and not t.isdigit()]
    toks += [f"p:{t}" for t in post.split()]
    return toks


def hashed_tokens(rec: pd.DataFrame, rows: np.ndarray, workers: int) -> sp.csr_matrix:
    """Binary token matrix over all records (rows outside `rows` left empty).

    Tokens are hashed into 2^24 columns (no vocabulary to build or ship to
    workers); the rare collisions only perturb blocking scores slightly."""
    if np.any(np.diff(rows) <= 0):
        raise ValueError("rows must be strictly increasing")
    docs = (rec["name_core"] + "\t" + rec["addr_canon"] + "\t" + rec["addr_post"]).iloc[rows].tolist()
    hv = HashingVectorizer(analyzer=rare_analyzer, n_features=1 << HASH_BITS, alternate_sign=False,
                           norm=None, binary=True, dtype=np.float32)
    H = sp.vstack(parallel_map(_hash_chunk, hv, _chunks(docs, 100_000), workers)).tocsr()
    counts = np.zeros(len(rec), np.int64)
    counts[rows] = np.diff(H.indptr)
    indptr = np.concatenate([[0], np.cumsum(counts)])
    return sp.csr_matrix((H.data, H.indices, indptr), shape=(len(rec), H.shape[1]))


def rare_weights(P: sp.csr_matrix, max_df: int) -> np.ndarray:
    df = np.bincount(P.indices, minlength=P.shape[1])
    rare = (df > 0) & (df <= max_df)
    w = np.zeros(P.shape[1], np.float32)
    w[rare] = np.log((P.shape[0] + 1) / (df[rare] + 1)) + 1.0
    return w


def rare_search(Hq: sp.csr_matrix, Hp: sp.csr_matrix, w: np.ndarray, k: int,
                chunk: int = 100_000):
    """Top-k pool rows by summed IDF of shared rare tokens, per query row."""
    Pb = Hp.copy()
    Pb.data = (w[Pb.indices] > 0).astype(np.float32)
    Pb.eliminate_zeros()
    PT = Pb.T.tocsr()
    qi, pj, sc = [], [], []
    for s in range(0, Hq.shape[0], chunk):
        Q = Hq[s:s + chunk].copy()
        Q.data = Q.data * w[Q.indices]
        Q.eliminate_zeros()
        S = (Q @ PT).tocoo()
        if S.nnz == 0:
            continue
        order = np.lexsort((-S.data, S.row))
        r, c, v = S.row[order], S.col[order], S.data[order]
        first = np.r_[0, np.flatnonzero(np.diff(r)) + 1]
        rank = np.arange(len(r)) - np.repeat(first, np.diff(np.r_[first, len(r)]))
        keep = rank < k
        qi.append(r[keep] + s); pj.append(c[keep]); sc.append(v[keep])
    if not qi:
        return np.empty(0, int), np.empty(0, int), np.empty(0, np.float32)
    return np.concatenate(qi), np.concatenate(pj), np.concatenate(sc)


def rowwise_sparse_dot(A: sp.csr_matrix, B: sp.csr_matrix, ii, jj, w=None,
                       chunk: int = 2_000_000) -> np.ndarray:
    out = np.empty(len(ii), np.float32)
    for s in range(0, len(ii), chunk):
        a = A[ii[s:s + chunk]]
        if w is not None:
            a = a.copy(); a.data = a.data * w[a.indices]
        out[s:s + chunk] = np.asarray(a.multiply(B[jj[s:s + chunk]]).sum(axis=1)).ravel()
    return out


def rowwise_dense_dot(Z: np.ndarray, ii, jj, chunk: int = 2_000_000) -> np.ndarray:
    out = np.empty(len(ii), np.float32)
    for s in range(0, len(ii), chunk):
        out[s:s + chunk] = np.einsum("ij,ij->i", Z[ii[s:s + chunk]], Z[jj[s:s + chunk]])
    return out


# ---------------------------------------------------------------- driver

def generate_candidates(split: Split, cfg: RetrievalConfig, log=print) -> pd.DataFrame:
    rec = split.rec
    n = len(rec)
    faiss.omp_set_num_threads(cfg.workers)
    src = rec["source"].to_numpy()
    country = rec["country_norm"].to_numpy()
    q_countries = pd.unique(country[split.query])
    pool_mask = (src != 1) & np.isin(country, q_countries)
    active = np.union1d(split.query, np.flatnonzero(pool_mask))
    log(f"  queries {len(split.query):,} S1 | pool {pool_mask.sum():,} S2/S3 | "
        f"countries {list(q_countries)}")

    found: dict[str, list] = {b: [] for b in cfg.blockers}
    t0 = time.time()
    Z = {v: embed_view(rec, active, v, cfg, log) for v in cfg.views}
    t = time.time()
    H = hashed_tokens(rec, active, cfg.workers)
    log(f"    rare-token index: {H.nnz:,} tokens in {time.time() - t:.0f}s")
    rare_w: dict = {}

    for c in q_countries:
        q = split.query[country[split.query] == c]
        for tgt in (2, 3):
            p = np.flatnonzero((src == tgt) & (country == c))
            if len(p) == 0 or len(q) == 0:
                continue
            t = time.time()
            for v in cfg.views:
                D, I = ann_search(Z[v][q], Z[v][p], cfg.k_ann, cfg.nprobe, cfg.seed)
                ok = (I >= 0) & (D > 0)
                found[f"ann_{v}"].append((np.repeat(q, I.shape[1])[ok.ravel()], p[I[ok]]))
            w = rare_weights(H[p], cfg.rare_max_df)
            rare_w[(c, tgt)] = (p, w)
            qi, pj, _ = rare_search(H[q], H[p], w, cfg.k_rare)
            found["rare"].append((q[qi], p[pj]))
            log(f"    {c} -> S{tgt}: {len(q):,} x {len(p):,} searched in {time.time() - t:.0f}s")

    # exact core-name blocks (within country, skip generic names)
    key = pd.Series(country.astype(object) + "|" + rec["name_core"].to_numpy().astype(object))
    pool_rows = np.flatnonzero(pool_mask & (rec["name_core"].to_numpy() != ""))
    pool_keys = key.iloc[pool_rows]
    sizes = pool_keys.map(pool_keys.value_counts())
    pool_df = pd.DataFrame({"key": pool_keys.to_numpy(), "j": pool_rows})[
        sizes.to_numpy() <= cfg.exact_max_block]
    q_df = pd.DataFrame({"key": key.iloc[split.query].to_numpy(), "i": split.query})
    m = q_df.merge(pool_df, on="key")
    found["exact"].append((m["i"].to_numpy(), m["j"].to_numpy()))

    # union + flags
    keys_b = {b: np.unique(np.concatenate([a.astype(np.int64) * n + b_ for a, b_ in found[b]]))
              if found[b] else np.empty(0, np.int64) for b in cfg.blockers}
    keys = np.unique(np.concatenate(list(keys_b.values())))
    cand = pd.DataFrame({"i": (keys // n).astype(np.int64), "j": (keys % n).astype(np.int64)})
    for b in cfg.blockers:
        flag = np.zeros(len(keys), np.int8)
        flag[np.searchsorted(keys, keys_b[b])] = 1
        cand[f"blk_{b}"] = flag
    cand["n_blockers"] = cand[[f"blk_{b}" for b in cfg.blockers]].sum(axis=1).astype(np.int8)
    ii, jj = cand["i"].to_numpy(), cand["j"].to_numpy()
    cand["src"] = src[jj]
    for v in cfg.views:
        cand[f"cos_{v}"] = rowwise_dense_dot(Z[v], ii, jj)
    # rare-token score for every pair, with the weights of its (country, source)
    cand["rare_score"] = np.float32(0)
    for (c, tgt), (p, w) in rare_w.items():
        sel = np.flatnonzero((cand["src"].to_numpy() == tgt) & (country[jj] == c))
        if len(sel):
            cand.loc[sel, "rare_score"] = rowwise_sparse_dot(H, H, ii[sel], jj[sel], w)
    log(f"  union: {len(cand):,} pairs ({len(cand) / max(1, len(split.query)):.1f} per S1) "
        f"in {time.time() - t0:.0f}s total")
    return cand


def _in_sorted(sorted_keys: np.ndarray, x: np.ndarray) -> np.ndarray:
    if len(sorted_keys) == 0:
        return np.zeros(len(x), bool)
    pos = np.minimum(np.searchsorted(sorted_keys, x), len(sorted_keys) - 1)
    return sorted_keys[pos] == x


def blocking_report(split: Split, cand: pd.DataFrame, cfg: RetrievalConfig) -> dict:
    """Recall ceiling + candidate burden. `cand` rows are sorted by (i, j)."""
    nq = len(split.query)
    per_s1 = np.bincount(np.searchsorted(split.query, cand["i"].to_numpy()), minlength=nq)
    rep = {"queries": nq, "pairs": len(cand), "avg_per_s1": float(per_s1.mean()),
           "p50_per_s1": float(np.median(per_s1)), "p95_per_s1": float(np.percentile(per_s1, 95)),
           "max_per_s1": int(per_s1.max()) if nq else 0}
    if not split.has_labels:
        return rep
    n = len(split.rec)
    tk = split.truth["i"].to_numpy() * n + split.truth["j"].to_numpy()
    ck = cand["i"].to_numpy() * n + cand["j"].to_numpy()
    hit = _in_sorted(ck, tk)
    rep["true_pairs"] = int(len(tk))
    rep["pair_recall"] = float(hit.mean()) if len(tk) else float("nan")
    tsrc = split.rec["source"].to_numpy()[split.truth["j"].to_numpy()]
    for s in (2, 3):
        rep[f"recall_S{s}"] = float(hit[tsrc == s].mean()) if (tsrc == s).any() else float("nan")
    tc = split.rec["country_norm"].to_numpy()[split.truth["i"].to_numpy()]
    for c in pd.unique(tc):
        rep[f"recall_{c}"] = float(hit[tc == c].mean())
    nb = cand["n_blockers"].to_numpy()
    for b in cfg.blockers:
        f = cand[f"blk_{b}"].to_numpy() == 1
        rep[f"recall_{b}"] = float(_in_sorted(ck[f], tk).mean())
        rep[f"only_{b}"] = int(_in_sorted(ck[f & (nb == 1)], tk).sum())
    # entity level: S1s whose every true match was retrieved (singletons count as complete)
    rep["s1_all_matches_found"] = float(1 - len(np.unique(split.truth["i"].to_numpy()[~hit])) / nq)
    return rep
