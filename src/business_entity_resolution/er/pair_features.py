"""Pair features for (S1 record, candidate) pairs.

Groups:
  name_*   string similarity of names (canonical ascii view), exact-equality
           flags at several normalisation levels, IDF-weighted token overlap,
           rare-token agreement / disagreement, acronym match
  addr_*   same for addresses
  num_* / post_*  numeric evidence: house numbers, PIN/ZIP; *missing* and
           *conflicting* are kept as separate signals (absence != contradiction)
  grp_*    context inside the candidate list: rank and gap-to-best among the
           S1's candidates, and the reverse (is this S1 the best S1 for the
           candidate?). S1 is deduplicated, so a candidate should belong to at
           most one S1 — the reverse rank encodes that.
Country is only used as an equality flag (open set: France is unseen in train).
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from .data import Split
from .text import LEGAL_TOKENS


def _idf(token_lists) -> dict[str, float]:
    df = Counter()
    for toks in token_lists:
        df.update(set(toks))
    n = len(token_lists)
    return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}


def _cp(a, b, scorer) -> np.ndarray:
    return process.cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32) / (
        1.0 if scorer is JaroWinkler.normalized_similarity else 100.0)


def _token_block(t1, t2, idf) -> np.ndarray:
    """IDF-weighted jaccard, max IDF shared, max IDF unshared, shared count."""
    out = np.zeros((len(t1), 4), np.float32)
    for k, (a, b) in enumerate(zip(t1, t2)):
        if not a or not b:
            continue
        inter, union = a & b, a | b
        wi = sum(idf.get(t, 1.0) for t in inter)
        wu = sum(idf.get(t, 1.0) for t in union)
        out[k, 0] = wi / wu if wu else 0.0
        out[k, 1] = max((idf.get(t, 1.0) for t in inter), default=0.0)
        out[k, 2] = max((idf.get(t, 1.0) for t in (union - inter)), default=0.0)
        out[k, 3] = len(inter)
    return out


def _numeric_block(n1, n2, p1, p2) -> np.ndarray:
    cols = ("num_n1", "num_n2", "num_jacc", "num_inter", "num_conflict", "num_subset",
            "num_first_eq", "num_first_conflict", "post_eq", "post_conflict", "post_missing")
    out = np.zeros((len(n1), len(cols)), np.float32)
    for k in range(len(n1)):
        a, b = n1[k], n2[k]
        sa, sb = set(a), set(b)
        out[k, 0], out[k, 1] = len(sa), len(sb)
        if sa and sb:
            inter = sa & sb
            out[k, 2] = len(inter) / len(sa | sb)
            out[k, 3] = len(inter)
            out[k, 4] = float(not inter)
            out[k, 5] = float(sa <= sb or sb <= sa)
            out[k, 6] = float(a[0] == b[0])
            out[k, 7] = float(a[0] != b[0] and a[0] not in sb and b[0] not in sa)
        pa, pb = set(p1[k]), set(p2[k])
        if pa and pb:
            out[k, 8] = float(bool(pa & pb))
            out[k, 9] = float(not (pa & pb))
        else:
            out[k, 10] = 1.0
    return pd.DataFrame(out, columns=cols)


def _group_block(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    g1 = df.groupby("s1")
    out["grp_n_cands"] = g1["cand"].transform("size").astype(np.float32)
    for col in ("cos_full", "cos_name", "cos_addr", "name_tset", "addr_tset"):
        out[f"grp_rank_{col}"] = g1[col].rank(ascending=False, method="min").astype(np.float32)
        out[f"grp_gap_{col}"] = (g1[col].transform("max") - df[col]).astype(np.float32)
    g2 = df.groupby("cand")
    out["grp_rev_n_s1"] = g2["s1"].transform("size").astype(np.float32)
    for col in ("cos_full", "name_tset"):
        out[f"grp_rev_rank_{col}"] = g2[col].rank(ascending=False, method="min").astype(np.float32)
        out[f"grp_rev_gap_{col}"] = (g2[col].transform("max") - df[col]).astype(np.float32)
    return out


def build_features(split: Split, cand: pd.DataFrame) -> pd.DataFrame:
    r = split.records
    A = r.loc[cand["s1"].to_numpy()].reset_index(drop=True)
    B = r.loc[cand["cand"].to_numpy()].reset_index(drop=True)
    f = cand.reset_index(drop=True).copy()

    f["src"] = B["source"].to_numpy(np.int8)
    f["country_eq"] = (A["country_norm"].to_numpy() == B["country_norm"].to_numpy()).astype(np.int8)
    f["non_ascii_1"], f["non_ascii_2"] = A["non_ascii"].to_numpy(), B["non_ascii"].to_numpy()

    # ---- names ----
    n1, n2 = A["name_canon"].tolist(), B["name_canon"].tolist()
    c1, c2 = A["name_core"].tolist(), B["name_core"].tolist()
    f["name_ratio"] = _cp(n1, n2, fuzz.ratio)
    f["name_partial"] = _cp(n1, n2, fuzz.partial_ratio)
    f["name_tsort"] = _cp(n1, n2, fuzz.token_sort_ratio)
    f["name_tset"] = _cp(n1, n2, fuzz.token_set_ratio)
    f["name_jw"] = _cp(n1, n2, JaroWinkler.normalized_similarity)
    f["name_core_ratio"] = _cp(c1, c2, fuzz.ratio)
    f["name_core_tset"] = _cp(c1, c2, fuzz.token_set_ratio)
    f["name_exact_norm"] = (A["name_norm"].to_numpy() == B["name_norm"].to_numpy()).astype(np.int8)
    f["name_exact_canon"] = (np.array(n1, object) == np.array(n2, object)).astype(np.int8)
    f["name_exact_core"] = (np.array(c1, object) == np.array(c2, object)).astype(np.int8)
    l1, l2 = A["name_canon"].str.len().to_numpy(), B["name_canon"].str.len().to_numpy()
    f["name_len_1"], f["name_len_2"] = l1, l2
    f["name_len_ratio"] = np.minimum(l1, l2) / np.maximum(np.maximum(l1, l2), 1)
    f["name_ntok_1"] = A["name_core"].str.count(" ").to_numpy() + 1
    f["name_ntok_2"] = B["name_core"].str.count(" ").to_numpy() + 1
    first1 = A["name_core"].str.split(" ").str[0].to_numpy()
    first2 = B["name_core"].str.split(" ").str[0].to_numpy()
    f["name_first_tok_eq"] = (first1 == first2).astype(np.int8)
    acr1, acr2 = A["name_acr"].to_numpy(), B["name_acr"].to_numpy()
    comp1 = A["name_core"].str.replace(" ", "", regex=False).to_numpy()
    comp2 = B["name_core"].str.replace(" ", "", regex=False).to_numpy()
    f["name_acronym"] = (((acr1 != "") & (acr1 == comp2)) | ((acr2 != "") & (acr2 == comp1))).astype(np.int8)
    leg1 = [frozenset(t for t in s.split() if t in LEGAL_TOKENS) for s in n1]
    leg2 = [frozenset(t for t in s.split() if t in LEGAL_TOKENS) for s in n2]
    f["name_legal_mismatch"] = np.array([bool(a) and bool(b) and a != b for a, b in zip(leg1, leg2)], np.int8)

    name_tok = [frozenset(s.split()) for s in r["name_core"]]
    name_idf = _idf(name_tok)
    tb = _token_block([frozenset(s.split()) for s in c1], [frozenset(s.split()) for s in c2], name_idf)
    f["name_idf_jacc"], f["name_max_idf_shared"], f["name_max_idf_unshared"], f["name_n_shared"] = tb.T

    # ---- addresses ----
    a1, a2 = A["addr_canon"].tolist(), B["addr_canon"].tolist()
    f["addr_ratio"] = _cp(a1, a2, fuzz.ratio)
    f["addr_partial"] = _cp(a1, a2, fuzz.partial_ratio)
    f["addr_tsort"] = _cp(a1, a2, fuzz.token_sort_ratio)
    f["addr_tset"] = _cp(a1, a2, fuzz.token_set_ratio)
    f["addr_exact"] = (np.array(a1, object) == np.array(a2, object)).astype(np.int8)
    al1, al2 = A["addr_canon"].str.len().to_numpy(), B["addr_canon"].str.len().to_numpy()
    f["addr_len_1"], f["addr_len_2"] = al1, al2
    f["addr_empty_any"] = ((al1 == 0) | (al2 == 0)).astype(np.int8)
    addr_tok = [frozenset(s.split()) for s in r["addr_canon"]]
    addr_idf = _idf(addr_tok)
    tb = _token_block([frozenset(s.split()) for s in a1], [frozenset(s.split()) for s in a2], addr_idf)
    f["addr_idf_jacc"], f["addr_max_idf_shared"], f["addr_max_idf_unshared"], f["addr_n_shared"] = tb.T

    num = _numeric_block(A["addr_nums"].tolist(), B["addr_nums"].tolist(),
                         A["addr_post"].tolist(), B["addr_post"].tolist())
    f = pd.concat([f, num], axis=1)

    # ---- cross-field: name of one inside address of other (DBA / landmark style) ----
    f["name_in_addr"] = np.maximum(_cp(c1, a2, fuzz.partial_ratio), _cp(c2, a1, fuzz.partial_ratio))

    f = pd.concat([f, _group_block(f)], axis=1)
    return f


META_COLS = ("s1", "cand", "i", "j", "label")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]
