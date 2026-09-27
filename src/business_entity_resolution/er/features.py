"""Pair features at scale: integer rows, chunked, multi-core.

Two feature sets:
  CHEAP  (pre-ranker, computed for every blocked pair, ~100 per S1):
         blocker flags, per-view cosines, rare-token score, rank / gap of the
         pair inside its S1's candidate list, and the reverse check (is this
         S1 the best S1 for the candidate among ALL S1s?)
  FULL   (matcher, only for pairs that survive the pre-ranker, ~10 per S1):
         CHEAP + string similarities (rapidfuzz), IDF-weighted token overlap
         for names and addresses, number / postcode agreement vs conflict vs
         missing, legal-suffix mismatch, acronym, name-inside-address.

Country equality is not a feature: blocking is within-country by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

try:
    import polars as pl
except ImportError:
    pl = None

from .retrieval import Resources, _G, imap_ranges
from .text import LEGAL_TOKENS

MATCH_COLUMNS = ["name_acr", "addr_nums", "non_ascii"]   # loaded on top of RETRIEVAL_COLUMNS


# ---------------------------------------------------------------- cheap

def group_features(cand: pd.DataFrame, cols: list[str], prefix: str = "grp") -> pd.DataFrame:
    """Rank (1 = best) and gap-to-best of each score inside its S1's list, per source.
    Polars runs these grouped windows multi-threaded (~3x pandas); pandas is the
    fallback. Both give identical numbers, so saved models stay valid."""
    if pl is not None:
        df = pl.DataFrame({c: cand[c].to_numpy() for c in ["i", "src", *cols]}, nan_to_null=True)
        g = ["i", "src"]
        exprs = [pl.len().over(g).cast(pl.Float32).alias(f"{prefix}_n")]
        for c in cols:
            exprs += [pl.col(c).rank("min", descending=True).over(g).cast(pl.Float32).alias(f"{prefix}_rank_{c}"),
                      (pl.col(c).max().over(g) - pl.col(c)).cast(pl.Float32).alias(f"{prefix}_gap_{c}")]
        res = df.select(exprs)
        return pd.DataFrame({k: res[k].to_numpy() for k in res.columns}, index=cand.index)
    out = {}
    g = cand.groupby(["i", "src"], sort=False)
    out[f"{prefix}_n"] = g["j"].transform("size").to_numpy(np.float32)
    for c in cols:
        out[f"{prefix}_rank_{c}"] = g[c].rank(ascending=False, method="min").to_numpy(np.float32)
        out[f"{prefix}_gap_{c}"] = (g[c].transform("max") - cand[c]).to_numpy(np.float32)
    return pd.DataFrame(out, index=cand.index)


def reverse_features(cand: pd.DataFrame, rev: pd.DataFrame, view: str) -> pd.DataFrame:
    r = rev.set_index("j").reindex(cand["j"].to_numpy())
    i = cand["i"].to_numpy()
    cos = cand[f"cos_{view}"].to_numpy()
    return pd.DataFrame({
        "rev_is_best": (r["rev_i1"].to_numpy() == i).astype(np.int8),
        "rev_is_second": (r["rev_i2"].to_numpy() == i).astype(np.int8),
        "rev_gap": (r["rev_s1"].to_numpy() - cos).astype(np.float32),
        "rev_margin": (r["rev_s1"].to_numpy() - r["rev_s2"].to_numpy()).astype(np.float32),
    }, index=cand.index)


def cheap_features(cand: pd.DataFrame, rev: pd.DataFrame, views: tuple[str, ...]) -> pd.DataFrame:
    score_cols = [f"cos_{v}" for v in views] + ["rare_score"]
    return pd.concat([cand, group_features(cand, score_cols), reverse_features(cand, rev, views[0])],
                     axis=1)


def cheap_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("blk_", "cos_", "grp_", "rev_"))
            or c in ("n_blockers", "src", "rare_score")]


# ---------------------------------------------------------------- full

def _cp(a, b, scorer, scale=100.0) -> np.ndarray:
    return (process.cpdist(a, b, scorer=scorer, workers=-1) / scale).astype(np.float32)


def string_features(rec: pd.DataFrame, ii: np.ndarray, jj: np.ndarray) -> pd.DataFrame:
    col = lambda c, rows: rec[c].iloc[rows].tolist()  # noqa: E731
    n1, n2 = col("name_canon", ii), col("name_canon", jj)
    c1, c2 = col("name_core", ii), col("name_core", jj)
    k1, k2 = col("name_skel", ii), col("name_skel", jj)
    a1, a2 = col("addr_canon", ii), col("addr_canon", jj)
    s1, s2 = col("addr_skel", ii), col("addr_skel", jj)
    r1, r2 = col("name_acr", ii), col("name_acr", jj)
    f = {
        "name_ratio": _cp(n1, n2, fuzz.ratio),
        "name_partial": _cp(n1, n2, fuzz.partial_ratio),
        "name_tsort": _cp(n1, n2, fuzz.token_sort_ratio),
        "name_tset": _cp(n1, n2, fuzz.token_set_ratio),
        "name_jw": _cp(n1, n2, JaroWinkler.normalized_similarity, 1.0),
        "name_core_ratio": _cp(c1, c2, fuzz.ratio),
        "name_core_tset": _cp(c1, c2, fuzz.token_set_ratio),
        "name_skel_ratio": _cp(k1, k2, fuzz.ratio),
        "name_skel_tset": _cp(k1, k2, fuzz.token_set_ratio),
        "addr_ratio": _cp(a1, a2, fuzz.ratio),
        "addr_partial": _cp(a1, a2, fuzz.partial_ratio),
        "addr_tsort": _cp(a1, a2, fuzz.token_sort_ratio),
        "addr_tset": _cp(a1, a2, fuzz.token_set_ratio),
        "addr_skel_tset": _cp(s1, s2, fuzz.token_set_ratio),
        "name_in_addr": np.maximum(_cp(c1, a2, fuzz.partial_ratio), _cp(c2, a1, fuzz.partial_ratio)),
    }
    arr = lambda x: np.array(x, dtype=object)  # noqa: E731
    f["name_exact"] = (arr(n1) == arr(n2)).astype(np.int8)
    f["name_core_exact"] = (arr(c1) == arr(c2)).astype(np.int8)
    f["addr_exact"] = (arr(a1) == arr(a2)).astype(np.int8)
    l1 = np.fromiter(map(len, n1), np.float32, len(n1))
    l2 = np.fromiter(map(len, n2), np.float32, len(n2))
    al1 = np.fromiter(map(len, a1), np.float32, len(a1))
    al2 = np.fromiter(map(len, a2), np.float32, len(a2))
    f.update(name_len_1=l1, name_len_2=l2, name_len_ratio=np.minimum(l1, l2) / np.maximum(np.maximum(l1, l2), 1),
             addr_len_1=al1, addr_len_2=al2, addr_empty_any=((al1 == 0) | (al2 == 0)).astype(np.int8))
    comp1 = [x.replace(" ", "") for x in c1]
    comp2 = [x.replace(" ", "") for x in c2]
    f["name_acronym"] = np.fromiter(((a != "" and a == d) or (b != "" and b == e)
                                     for a, b, d, e in zip(r1, r2, comp2, comp1)), np.int8, len(r1))
    return pd.DataFrame(f)


NUM_COLS = ("num_n1", "num_n2", "num_jacc", "num_inter", "num_conflict", "num_subset",
            "num_prefix", "num_first_eq", "num_first_conflict", "post_eq", "post_conflict",
            "post_missing", "legal_mismatch")


def _numeric_range(r):
    lo, hi = r
    ii, jj = _G["ii"][lo:hi], _G["jj"][lo:hi]
    nums, post, name = _G["nums"], _G["post"], _G["name"]
    N1, N2 = nums.iloc[ii].tolist(), nums.iloc[jj].tolist()
    P1, P2 = post.iloc[ii].tolist(), post.iloc[jj].tolist()
    M1, M2 = name.iloc[ii].tolist(), name.iloc[jj].tolist()
    out = np.zeros((hi - lo, len(NUM_COLS)), np.float32)
    for k in range(hi - lo):
        a, b = N1[k].split(), N2[k].split()
        sa, sb = set(a), set(b)
        out[k, 0], out[k, 1] = len(sa), len(sb)
        if sa and sb:
            inter = sa & sb
            out[k, 2] = len(inter) / len(sa | sb)
            out[k, 3] = len(inter)
            out[k, 4] = float(not inter)
            out[k, 5] = float(sa <= sb or sb <= sa)
            # a dropped digit ("3352" vs "335") is a prefix, not a real conflict
            out[k, 6] = float(any(x != y and (x.startswith(y) or y.startswith(x))
                                  for x in sa - inter for y in sb - inter))
            out[k, 7] = float(a[0] == b[0])
            out[k, 8] = float(a[0] != b[0] and a[0] not in sb and b[0] not in sa)
        pa, pb = set(P1[k].split()), set(P2[k].split())
        if pa and pb:
            out[k, 9] = float(bool(pa & pb))
            out[k, 10] = float(not (pa & pb))
        else:
            out[k, 11] = 1.0
        la = {t for t in M1[k].split() if t in LEGAL_TOKENS}
        lb = {t for t in M2[k].split() if t in LEGAL_TOKENS}
        out[k, 12] = float(bool(la) and bool(lb) and la != lb)
    return out


def numeric_features(rec: pd.DataFrame, ii: np.ndarray, jj: np.ndarray, workers: int) -> pd.DataFrame:
    parts = [m for _, m in imap_ranges(_numeric_range, len(ii), 100_000, workers, ii=ii, jj=jj,
                                       nums=rec["addr_nums"], post=rec["addr_post"],
                                       name=rec["name_canon"])]
    return pd.DataFrame(np.vstack(parts) if parts else np.zeros((0, len(NUM_COLS))), columns=NUM_COLS)


def _row_idf_sum(M: sp.csr_matrix, idf: np.ndarray) -> np.ndarray:
    rows = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
    return np.bincount(rows, weights=idf[M.indices] * M.data, minlength=M.shape[0])


def _token_block(H: sp.csr_matrix, idf: np.ndarray, ii, jj, prefix: str) -> dict:
    A, B = H[ii], H[jj]
    shared = A.multiply(B).tocsr()
    shared.data = idf[shared.indices]
    sa, sb = _row_idf_sum(A, idf), _row_idf_sum(B, idf)   # (never multiply by a dense 2^24 vector)
    ss = np.asarray(shared.sum(axis=1)).ravel()
    union = (A + B).tocsr()
    only = union.copy()
    only.data = np.where(union.data == 1, idf[union.indices], 0).astype(np.float32)
    denom = sa + sb - ss
    return {
        f"{prefix}_idf_jacc": np.where(denom > 0, ss / np.maximum(denom, 1e-9), 0).astype(np.float32),
        f"{prefix}_idf_shared": ss.astype(np.float32),
        f"{prefix}_max_idf_shared": shared.max(axis=1).toarray().ravel().astype(np.float32),
        f"{prefix}_max_idf_unshared": only.max(axis=1).toarray().ravel().astype(np.float32),
        f"{prefix}_n_shared": np.diff(shared.indptr).astype(np.float32),
    }


def token_features(res: Resources, ii: np.ndarray, jj: np.ndarray) -> pd.DataFrame:
    f = _token_block(res.Hn, res.idf_name, ii, jj, "name")
    f.update(_token_block(res.Ha, res.idf_addr, ii, jj, "addr"))
    return pd.DataFrame(f)


def full_features(rec: pd.DataFrame, res: Resources, cheap: pd.DataFrame, workers: int) -> pd.DataFrame:
    ii, jj = cheap["i"].to_numpy(), cheap["j"].to_numpy()
    base = cheap.reset_index(drop=True)
    f = pd.concat([base, string_features(rec, ii, jj), token_features(res, ii, jj),
                   numeric_features(rec, ii, jj, workers)], axis=1)
    return pd.concat([f, group_features(f, ["name_tset", "addr_tset", "name_idf_jacc"], "grp2")], axis=1)


META = ("i", "j", "label", "pre_p", "p", "fold")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META]


# ---------------------------------------------------------------- siblings

SIB_TEXT = ["name_canon", "addr_canon", "name_skel", "addr_skel"]


def sibling_features(df: pd.DataFrame, text: pd.DataFrame, prob: str = "p", top: int = 3) -> pd.DataFrame:
    """Collective evidence for multi-match S1s: how much does candidate j look like
    the S1's other strongest candidates? Matched records of one business often
    resemble each other more than they resemble the S1 itself.

    For each pair, the `top` highest-p other candidates of the same S1 (by the
    first-stage p) are compared with j. Features: p and similarities of the
    strongest sibling, the p-weighted best similarity over the top siblings, and
    how many other candidates are confident (p >= 0.5). `text` holds SIB_TEXT
    columns indexed by row number (dataset.take_rows)."""
    n = len(df)
    i, j = df["i"].to_numpy(), df["j"].to_numpy()
    p = df[prob].to_numpy().astype(np.float32)
    o = np.lexsort((-p, i))
    i_s = i[o]
    new = np.r_[True, i_s[1:] != i_s[:-1]]
    starts, g = np.flatnonzero(new), np.cumsum(new) - 1
    size = np.diff(np.r_[starts, n])[g]
    pos = np.arange(n) - starts[g]                           # rank of the pair in its S1 list
    # partner r (r = 0..top-1) = r-th best candidate of the same S1, skipping the pair itself
    rows = np.searchsorted(text.index.to_numpy(), j)
    name = text["name_canon"].to_numpy()[rows]
    addr = text["addr_canon"].to_numpy()[rows]
    skel = (text["name_skel"].to_numpy()[rows] + " " + text["addr_skel"].to_numpy()[rows])
    conf = (p[o] >= 0.5).astype(np.int32)
    n_conf = np.add.reduceat(conf, starts)[g] - conf
    out = {c: np.zeros(n, np.float32) for c in
           ("sib1_p", "sib1_name", "sib1_addr", "sib1_skel", "sib_w_name", "sib_w_addr", "sib_w_skel")}
    for r in range(top):
        k = r + (pos <= r)                                    # skip self
        ok = k < size
        a, b = o[ok], o[starts[g[ok]] + k[ok]]                # pair rows (original order) and partner rows
        sims = {"name": _cp(name[a].tolist(), name[b].tolist(), fuzz.token_set_ratio),
                "addr": _cp(addr[a].tolist(), addr[b].tolist(), fuzz.token_set_ratio),
                "skel": _cp(skel[a].tolist(), skel[b].tolist(), fuzz.token_set_ratio)}
        pb = p[b]
        if r == 0:
            out["sib1_p"][a] = pb
            for f, v in sims.items():
                out[f"sib1_{f}"][a] = v
        for f, v in sims.items():
            out[f"sib_w_{f}"][a] = np.maximum(out[f"sib_w_{f}"][a], pb * v)
    out["sib_n_conf"] = np.empty(n, np.float32)
    out["sib_n_conf"][o] = n_conf
    return pd.DataFrame(out, index=df.index)
