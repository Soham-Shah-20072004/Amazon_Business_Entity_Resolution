"""Scalable candidate generation (blocking) for ~2M S1 x ~10M S2/S3 records.

Two ideas make it tractable:
  1. Matching is within-country (no true train pair crosses countries), so each
     country is an independent, smaller problem. Countries are an open set:
     France (test only) is simply one more group.
  2. No blocker compares all pairs; each looks at a small neighbourhood per S1:
       ann_<view>  approximate nearest neighbours (FAISS IVF) on d-dim vectors
                   = char n-gram TF-IDF compressed by truncated SVD ("LSA").
                   views: full (name + address), addr, skel (sound skeleton
                   of name + address, bridges Latin <-> transliterated
                   Devanagari/Gujarati); name-only is available but weak
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
import multiprocessing as mp
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dataclasses import dataclass, field

try:
    import faiss
except ImportError:      # cached stages (combine, diagnosis) never search
    faiss = None
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
    views: tuple[str, ...] = ("full", "addr", "skel")
    k_ann: int = 25            # neighbours per S1, per view, per target source
    k_rare: int = 20           # rare-token candidates per S1, per target source
    rare_max_df: int = 100     # "rare" = in <= max(rare_max_df, rare_per_million * pool/1e6)
    rare_per_million: int = 300  # pool records; scales with pool size (3M pool -> 900)
    exact_max_block: int = 50  # skip exact-name blocks bigger than this
    svd_dim: int = 128
    fit_sample: int = 200_000  # rows used to fit TF-IDF vocab + SVD (vocab saturates early)
    nprobe: int = 32           # IVF cells searched per query (recall vs speed)
    workers: int = 8
    seed: int = 42
    scratch: str = "/tmp/ber_scratch"   # on-disk (memory-mapped) vectors live here
    blockers: tuple[str, ...] = field(init=False)

    def __post_init__(self):
        self.blockers = tuple(f"ann_{v}" for v in self.views) + ("rare", "exact")


# ---------------------------------------------------------------- LSA views

def view_text(rec: pd.DataFrame, view: str) -> pd.Series:
    if view == "name":
        return rec["name_canon"]
    if view == "addr":
        return rec["addr_canon"]
    if view == "full":
        return rec["name_canon"] + " | " + rec["addr_canon"]
    if view == "skel":
        return rec["name_skel"] + " | " + rec["addr_skel"]
    raise ValueError(view)


def _vectorizer(view: str) -> TfidfVectorizer:
    ngram = (3, 4) if view in ("addr", "full") else (2, 4)
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


# Workers are forked, so they inherit the (Arrow-backed) text columns and the
# fitted models through this dict instead of receiving pickled copies; each
# task is just a (lo, hi) range into `rows`. Keeps memory flat on small boxes.
_G: dict = {}


def _ranges(n: int, size: int) -> list[tuple[int, int]]:
    return [(s, min(s + size, n)) for s in range(0, n, size)]


def imap_ranges(func, n: int, size: int, workers: int, **shared):
    """Yield ((lo, hi), func((lo, hi))) in order; `shared` is visible to func via _G."""
    _G.clear()
    _G.update(shared)
    rs = _ranges(n, size)
    if workers <= 1 or len(rs) == 1:
        for r in rs:
            yield r, func(r)
        return
    with mp.get_context("fork").Pool(workers) as pool:
        yield from zip(rs, pool.imap(func, rs, chunksize=1))


def _lsa_range(r):
    lo, hi = r
    texts = _G["texts"].iloc[_G["rows"][lo:hi]].tolist()
    return _G["lsa"].transform(texts).astype(np.float16)


def _hash_range(r):
    lo, hi = r
    rows = _G["rows"][lo:hi]
    docs = [f"{a}\t{b}\t{c}" for a, b, c in zip(_G["name"].iloc[rows].tolist(),
                                                  _G["addr"].iloc[rows].tolist(),
                                                  _G["post"].iloc[rows].tolist())]
    return _G["hv"].transform(docs)


def embed_view(rec: pd.DataFrame, rows: np.ndarray, view: str, cfg: RetrievalConfig,
               tag: str, log=print) -> np.ndarray:
    """(n_rec, dim) float16 memory-mapped array on disk; rows outside `rows` stay zero.

    float16 halves the footprint (cosines only need ~3 digits) and the OS
    page cache keeps the hot parts in RAM when there is room."""
    t = time.time()
    texts = view_text(rec, view)
    rng = np.random.default_rng(cfg.seed)
    fit_rows = np.sort(rng.choice(rows, min(cfg.fit_sample, len(rows)), replace=False))
    lsa = LSA(view, cfg.svd_dim, cfg.seed).fit(texts.iloc[fit_rows].tolist())
    Path(cfg.scratch).mkdir(parents=True, exist_ok=True)
    Z = np.lib.format.open_memmap(Path(cfg.scratch) / f"Z_{tag}_{view}.npy", mode="w+",
                                  dtype=np.float16, shape=(len(rec), lsa.comp.shape[1]))
    for (lo, hi), part in imap_ranges(_lsa_range, len(rows), 50_000, cfg.workers,
                                      texts=texts, rows=rows, lsa=lsa):
        Z[rows[lo:hi]] = part
    Z.flush()
    log(f"    view {view}: vocab {len(lsa.vec.vocabulary_):,} -> {lsa.comp.shape[1]}-d, "
        f"{len(rows):,} rows embedded in {time.time() - t:.0f}s")
    return Z


def f32(Z: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(Z[rows], dtype=np.float32)


# ---------------------------------------------------------------- rare tokens

def name_analyzer(doc: str) -> list[str]:
    return [f"n:{t}" for t in set(doc.split()) if len(t) >= 3 and not t.isdigit()
            and t not in LEGAL_TOKENS]


def addr_analyzer(doc: str) -> list[str]:
    addr, post = doc.split("\t")
    toks = [f"a:{t}" for t in set(addr.split()) if len(t) >= 3 and not t.isdigit()]
    return toks + [f"p:{t}" for t in post.split()]


def _hasher(analyzer) -> HashingVectorizer:
    return HashingVectorizer(analyzer=analyzer, n_features=1 << HASH_BITS, alternate_sign=False,
                             norm=None, binary=True, dtype=np.float32)


def _hash_range(r):
    lo, hi = r
    rows = _G["rows"][lo:hi]
    names = _G["name"].iloc[rows].tolist()
    docs = [f"{a}\t{b}" for a, b in zip(_G["addr"].iloc[rows].tolist(), _G["post"].iloc[rows].tolist())]
    return _hasher(name_analyzer).transform(names), _hasher(addr_analyzer).transform(docs)


def _scatter_rows(H: sp.csr_matrix, rows: np.ndarray, n: int) -> sp.csr_matrix:
    counts = np.zeros(n, np.int64)
    counts[rows] = np.diff(H.indptr)
    indptr = np.concatenate([[0], np.cumsum(counts)])
    return sp.csr_matrix((H.data, H.indices, indptr), shape=(n, H.shape[1]))


def hashed_tokens(rec: pd.DataFrame, rows: np.ndarray, workers: int):
    """(Hn, Ha): binary name-token and address-token matrices over all records
    (rows outside `rows` left empty). Tokens are hashed into 2^24 columns: no
    vocabulary to build or ship to workers; rare collisions only nudge scores."""
    if np.any(np.diff(rows) <= 0):
        raise ValueError("rows must be strictly increasing")
    parts = [m for _, m in imap_ranges(_hash_range, len(rows), 100_000, workers, rows=rows,
                                       name=rec["name_core"], addr=rec["addr_canon"],
                                       post=rec["addr_post"])]
    Hn = _scatter_rows(sp.vstack([a for a, _ in parts]).tocsr(), rows, len(rec))
    Ha = _scatter_rows(sp.vstack([b for _, b in parts]).tocsr(), rows, len(rec))
    return Hn, Ha


def idf_weights(H: sp.csr_matrix) -> np.ndarray:
    df = np.bincount(H.indices, minlength=H.shape[1])
    n = max(1, int((np.diff(H.indptr) > 0).sum()))
    return (np.log((n + 1) / (df + 1)) + 1.0).astype(np.float32)


def rare_weights(P: sp.csr_matrix, max_df: int) -> np.ndarray:
    df = np.bincount(P.indices, minlength=P.shape[1])
    rare = (df > 0) & (df <= max_df)
    w = np.zeros(P.shape[1], np.float32)
    w[rare] = np.log((P.shape[0] + 1) / (df[rare] + 1)) + 1.0
    return w


def rare_search(Hq: sp.csr_matrix, Hp: sp.csr_matrix, w: np.ndarray, k: int,
                chunk: int = 20_000):
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


def rowwise_dense_dot(Z: np.ndarray, ii, jj, chunk: int = 250_000, workers: int | None = None) -> np.ndarray:
    """Z[ii[k]] . Z[jj[k]] for every k. Threads over chunks (numpy releases the
    GIL), and each S1 row is read and cast once per chunk instead of once per
    candidate: ~11x faster than a single-threaded loop, same numbers."""
    out = np.empty(len(ii), np.float32)

    def part(s: int) -> None:
        e = min(s + chunk, len(ii))
        ui, inv = np.unique(ii[s:e], return_inverse=True)
        a = np.asarray(Z[ui], dtype=np.float32)[inv]
        b = np.asarray(Z[jj[s:e]], dtype=np.float32)
        out[s:e] = np.einsum("ij,ij->i", a, b)

    with ThreadPoolExecutor(workers or os.cpu_count() or 1) as ex:
        list(ex.map(part, range(0, len(ii), chunk)))
    return out


# ---------------------------------------------------------------- driver

@dataclass
class Resources:
    """Everything built once per split and reused for every batch of queries."""
    Z: dict                   # view -> (n_rec, dim) float16 memmap
    Hn: sp.csr_matrix         # name tokens
    Ha: sp.csr_matrix         # address tokens
    H: sp.csr_matrix          # Hn + Ha (rare-token blocker)
    idf_name: np.ndarray
    idf_addr: np.ndarray
    rare_w: dict              # (country, source) -> IDF weights of rare pool tokens
    src: np.ndarray
    country: np.ndarray
    all_s1: np.ndarray        # every S1 row of the query countries (reverse check)


def build_resources(split: Split, cfg: RetrievalConfig, log=print) -> Resources:
    rec = split.rec
    faiss.omp_set_num_threads(cfg.workers)
    src = rec["source"].to_numpy()
    country = rec["country_code"].to_numpy()
    q_countries = np.unique(country[split.query])
    in_c = np.isin(country, q_countries)
    all_s1 = np.flatnonzero((src == 1) & in_c)
    active = np.flatnonzero(in_c)          # every S1 (reverse check) + every S2/S3 of those countries
    log(f"  queries {len(split.query):,} S1 | all S1 {len(all_s1):,} | pool {int(((src != 1) & in_c).sum()):,} "
        f"S2/S3 | countries {[split.countries[c] for c in q_countries]}")
    Z = {v: embed_view(rec, active, v, cfg, split.name, log) for v in cfg.views}
    t = time.time()
    Hn, Ha = hashed_tokens(rec, active, cfg.workers)
    H = (Hn + Ha).tocsr()
    H.data[:] = 1.0
    rare_w = {}
    for c in q_countries:
        for tgt in (2, 3):
            p = np.flatnonzero((src == tgt) & (country == c))
            if len(p):
                max_df = max(cfg.rare_max_df, int(cfg.rare_per_million * len(p) / 1e6))
                rare_w[(int(c), tgt)] = rare_weights(H[p], max_df)
    log(f"    token index: {Hn.nnz:,} name + {Ha.nnz:,} address tokens in {time.time() - t:.0f}s")
    return Resources(Z, Hn, Ha, H, idf_weights(Hn), idf_weights(Ha), rare_w, src, country, all_s1)


def search(split: Split, res: Resources, cfg: RetrievalConfig, queries: np.ndarray,
           log=print, chunk: int = 200_000) -> dict:
    """blocker -> (i, j) int32 arrays for all `queries` (one index per country/source/view)."""
    src, country = res.src, res.country
    found: dict[str, list] = {b: [] for b in cfg.blockers}
    for c in np.unique(country[queries]):
        q = queries[country[queries] == c]
        for tgt in (2, 3):
            p = np.flatnonzero((src == tgt) & (country == c))
            if len(p) == 0:
                continue
            t = time.time()
            for v in cfg.views:
                index = ann_index(f32(res.Z[v], p), cfg.nprobe, cfg.seed)
                k = min(cfg.k_ann, len(p))
                for s in range(0, len(q), chunk):
                    qs = q[s:s + chunk]
                    D, I = index.search(f32(res.Z[v], qs), k)
                    ok = (I >= 0) & (D > 0)
                    found[f"ann_{v}"].append((np.repeat(qs, k)[ok.ravel()].astype(np.int32),
                                              p[I[ok]].astype(np.int32)))
                del index
            w = res.rare_w[(int(c), tgt)]
            qi, pj, _ = rare_search(res.H[q], res.H[p], w, cfg.k_rare)
            found["rare"].append((q[qi].astype(np.int32), p[pj].astype(np.int32)))
            log(f"    {split.countries[c]} -> S{tgt}: {len(q):,} x {len(p):,} searched "
                f"in {time.time() - t:.0f}s")
    found["exact"].append(exact_blocks(split, res, cfg, queries))
    return {b: (np.concatenate([a for a, _ in v]) if v else np.empty(0, np.int32),
                np.concatenate([b_ for _, b_ in v]) if v else np.empty(0, np.int32))
            for b, v in found.items()}


def n_gpus() -> int:
    return faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0


def ann_index(Zp: np.ndarray, nprobe: int, seed: int):
    """IVF index (exact flat index for small pools). Uses every GPU when the
    faiss-gpu build is installed and a GPU is present - the IVF scan is the
    slowest step on CPU (~20-28 min per country/source for 1.7M test S1s)."""
    n, d = Zp.shape
    gpu = n_gpus() > 0
    ivf = n > 20_000
    if ivf:
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, _nlist(n), faiss.METRIC_INNER_PRODUCT)
        index.cp.niter = 10
    else:
        index = faiss.IndexFlatIP(d)
    if gpu:
        index = faiss.index_cpu_to_all_gpus(index)
    if ivf:
        index.train(_train_sample(Zp, seed))
        if gpu:
            faiss.GpuParameterSpace().set_index_parameter(index, "nprobe", nprobe)
        else:
            index.nprobe = nprobe
    index.add(Zp)
    return index


def _nlist(n: int) -> int:
    return int(max(16, min(2 * math.sqrt(n), n / 40)))   # fewer cells = faster build


def _train_sample(Zp: np.ndarray, seed: int) -> np.ndarray:
    n = len(Zp)
    rng = np.random.default_rng(seed)
    return Zp[np.sort(rng.choice(n, min(n, _nlist(n) * 40), replace=False))]


def exact_blocks(split: Split, res: Resources, cfg: RetrievalConfig, queries: np.ndarray):
    """Identical core name within country; generic names (big blocks) skipped."""
    rec = split.rec
    name_code, _ = pd.factorize(rec["name_core"])
    has_name = rec["name_core"].str.len().to_numpy(dtype=np.int64, na_value=0) > 0
    key = res.country.astype(np.int64) * (int(name_code.max()) + 2) + name_code
    pool_rows = np.flatnonzero((res.src != 1) & has_name & np.isin(res.country, np.unique(res.country[queries])))
    _, inv, cnt = np.unique(key[pool_rows], return_inverse=True, return_counts=True)
    pool_df = pd.DataFrame({"key": key[pool_rows], "j": pool_rows})[cnt[inv] <= cfg.exact_max_block]
    m = pd.DataFrame({"key": key[queries], "i": queries}).merge(pool_df, on="key")
    return m["i"].to_numpy(np.int32), m["j"].to_numpy(np.int32)


def assemble(split: Split, res: Resources, cfg: RetrievalConfig, found: dict,
             rows: np.ndarray | None = None) -> pd.DataFrame:
    """Union of blockers for the S1 `rows` (all found if None), sorted by (i, j),
    with blocker flags and similarity scores for every view."""
    n = len(split.rec)
    keys_b = {}
    for b in cfg.blockers:
        i, j = found[b]
        if rows is not None:
            sel = np.isin(i, rows)
            i, j = i[sel], j[sel]
        keys_b[b] = np.unique(i.astype(np.int64) * n + j)
    keys = np.unique(np.concatenate(list(keys_b.values())))
    cand = pd.DataFrame({"i": (keys // n).astype(np.int64), "j": (keys % n).astype(np.int64)})
    for b in cfg.blockers:
        flag = np.zeros(len(keys), np.int8)
        flag[np.searchsorted(keys, keys_b[b])] = 1
        cand[f"blk_{b}"] = flag
    cand["n_blockers"] = cand[[f"blk_{b}" for b in cfg.blockers]].sum(axis=1).astype(np.int8)
    ii, jj = cand["i"].to_numpy(), cand["j"].to_numpy()
    cand["src"] = res.src[jj]
    for v in cfg.views:
        cand[f"cos_{v}"] = rowwise_dense_dot(res.Z[v], ii, jj)
    cand["rare_score"] = np.float32(0)
    for (c, tgt), w in res.rare_w.items():
        sel = np.flatnonzero((cand["src"].to_numpy() == tgt) & (res.country[jj] == c))
        if len(sel):
            cand.loc[sel, "rare_score"] = rowwise_sparse_dot(res.H, res.H, ii[sel], jj[sel], w)
    return cand


def reverse_best(split: Split, res: Resources, cfg: RetrievalConfig, cand_j: np.ndarray,
                 view: str | None = None, k: int = 3) -> pd.DataFrame:
    """For each candidate record j: its top-k S1s among ALL S1 of its country.

    S1 is deduplicated and no record belongs to two S1s (checked on the full
    train ground truth), so "is this S1 the best S1 for j?" is strong evidence.
    Computed against every S1, not just the sampled queries, so train and test
    see the same competition."""
    view = view or cfg.views[0]
    js = np.unique(cand_j)
    best_i = np.full((len(js), k), -1, np.int64)
    best_s = np.zeros((len(js), k), np.float32)
    for c in np.unique(res.country[js]):
        s1c = res.all_s1[res.country[res.all_s1] == c]
        sel = np.flatnonzero(res.country[js] == c)
        if len(s1c) == 0 or len(sel) == 0:
            continue
        index = ann_index(f32(res.Z[view], s1c), cfg.nprobe, cfg.seed)
        kk = min(k, len(s1c))
        for s in range(0, len(sel), 500_000):
            part = sel[s:s + 500_000]
            D, I = index.search(f32(res.Z[view], js[part]), kk)
            best_i[part, :kk] = np.where(I >= 0, s1c[np.maximum(I, 0)], -1)
            best_s[part, :kk] = np.where(I >= 0, D, 0)
        del index
    return pd.DataFrame({"j": js, "rev_i1": best_i[:, 0], "rev_s1": best_s[:, 0],
                         "rev_s2": best_s[:, 1] if k > 1 else 0.0,
                         "rev_i2": best_i[:, 1] if k > 1 else -1})


def generate_candidates(split: Split, cfg: RetrievalConfig, log=print) -> pd.DataFrame:
    t0 = time.time()
    res = build_resources(split, cfg, log)
    found = search(split, res, cfg, split.query, log)
    cand = assemble(split, res, cfg, found)
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
    tc = split.rec["country_code"].to_numpy()[split.truth["i"].to_numpy()]
    for c in np.unique(tc):
        rep[f"recall_{split.countries[c]}"] = float(hit[tc == c].mean())
    nb = cand["n_blockers"].to_numpy()
    for b in cfg.blockers:
        f = cand[f"blk_{b}"].to_numpy() == 1
        rep[f"recall_{b}"] = float(_in_sorted(ck[f], tk).mean())
        rep[f"only_{b}"] = int(_in_sorted(ck[f & (nb == 1)], tk).sum())
    # entity level: S1s whose every true match was retrieved (singletons count as complete)
    rep["s1_all_matches_found"] = float(1 - len(np.unique(split.truth["i"].to_numpy()[~hit])) / nq)
    return rep
