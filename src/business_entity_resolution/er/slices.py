"""Entity-level slice tags for validation reporting (an S1 can be in many)."""

from __future__ import annotations

import pandas as pd

from .data import Split


def entity_slices(split: Split, feats: pd.DataFrame) -> pd.DataFrame:
    r = split.records
    s1 = r.loc[split.s1_ids]
    n_true = pd.Series({s: len(split.truth.get(s, ())) for s in split.s1_ids})
    out = pd.DataFrame(index=split.s1_ids)

    for c in sorted(s1["country_norm"].unique()):
        out[f"country={c}"] = (s1["country_norm"] == c).to_numpy()
    out["singleton"] = (n_true == 0).to_numpy()
    out["one_match"] = (n_true == 1).to_numpy()
    out["multi_match"] = (n_true >= 2).to_numpy()
    out["name_short(<=8ch)"] = (s1["name_canon"].str.len() <= 8).to_numpy()
    out["name_long(>=30ch)"] = (s1["name_canon"].str.len() >= 30).to_numpy()
    out["s1_addr_missing"] = (s1["addr_canon"].str.len() == 0).to_numpy()
    out["s1_no_numbers"] = (s1["addr_nums"].str.len() == 0).to_numpy()
    out["s1_non_ascii"] = (s1["non_ascii"] == 1).to_numpy()

    pos = feats[feats["label"] == 1]
    neg = feats[feats["label"] == 0]
    g = pos.groupby("s1")
    out["easy_all_exact_name"] = (g["name_exact_canon"].min().reindex(out.index) == 1).to_numpy()
    out["pos_low_name(<0.6)"] = (g["name_ratio"].min().reindex(out.index) < 0.6).to_numpy()
    out["pos_numeric_conflict"] = (g["num_conflict"].max().reindex(out.index) == 1).to_numpy()
    translit = (pos["non_ascii_1"] != pos["non_ascii_2"]).groupby(pos["s1"]).any()
    out["pos_cross_script"] = translit.reindex(out.index, fill_value=False).to_numpy()
    out["same_name_distractor"] = (neg.groupby("s1")["name_exact_core"].max()
                                   .reindex(out.index) == 1).to_numpy()

    truth_pairs = {(s, c) for s, cs in split.truth.items() for c in cs}
    got = set(zip(feats["s1"], feats["cand"]))
    missed = {s for s, c in truth_pairs - got}
    out["blocking_missed_a_match"] = out.index.isin(list(missed))
    out["ALL"] = True
    return out.fillna(False)
