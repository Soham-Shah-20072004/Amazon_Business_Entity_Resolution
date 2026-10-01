#!/usr/bin/env python3
"""Where do the points go? Error analysis on the cached Stage B train output
(out-of-fold scores for the sampled train S1s). CPU, ~10 minutes, no retrieval.

  python scripts/diagnose.py --pairs-dir work/models/m1 \
      --train-records work/train/records.parquet --out diag

Prints (paste the block after "DIAGNOSIS"):
  1. points lost by error type: false matches, true matches scored below the
     rule, true matches that never became candidates (blocking / pre-ranker)
  2. what the missed matches look like: source, script, country, S1 size,
     similarity to the S1, and similarity to the S1's *predicted* matches
     (tests the sibling idea: are misses near-copies of matches we found?)
  3. decision rules on the same scores: global threshold, per source, per-S1
     expected F0.5
  4. points lost per country and per number of true matches
Writes diag/examples.tsv: real examples of each error type to read by eye.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" if (ROOT / "src").is_dir() else ROOT))  # repo or submission layout

from business_entity_resolution.er import evaluate as E  # noqa: E402
from business_entity_resolution.er.dataset import take_rows  # noqa: E402

TEXT = ["name_norm", "addr_norm", "name_canon", "addr_canon", "name_skel", "addr_skel",
        "country", "non_ascii", "source"]


def sim(a, b) -> np.ndarray:
    return process.cpdist(list(a), list(b), scorer=fuzz.token_set_ratio, workers=-1).astype(np.float32)


def rounded(row: pd.Series) -> dict:
    """Row of a tuning table as a JSON-friendly dict (mixed dtypes, so no Series.round)."""
    return {k: (bool(v) if isinstance(v, (bool, np.bool_)) else round(float(v), 4) if isinstance(v, (int, float, np.number))
                else v) for k, v in row.to_dict().items()}


def sibling_sim(ci, cj, pi, pj, pp, text, top=3) -> dict:
    """For pairs (ci, cj): best similarity of cj to the top-`top` predicted
    matches (pi, pj, prob pp) of the same S1, excluding cj itself."""
    o = np.lexsort((-pp, pi))
    pi, pj = pi[o], pj[o]
    lo, hi = np.searchsorted(pi, ci, "left"), np.searchsorted(pi, ci, "right")
    best = {f: np.full(len(ci), -1.0, np.float32) for f in ("name", "addr", "full")}
    for r in range(top + 1):
        k = lo + r
        ok = k < hi
        ok[ok] &= pj[k[ok]] != cj[ok]
        a, b = cj[ok], pj[k[ok]]
        s = {"name": sim(text.loc[a, "name_canon"], text.loc[b, "name_canon"]),
             "addr": sim(text.loc[a, "addr_canon"], text.loc[b, "addr_canon"]),
             "full": sim(text.loc[a, "skel"], text.loc[b, "skel"])}
        for f in best:
            best[f][ok] = np.maximum(best[f][ok], s[f])
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--train-records", required=True)
    ap.add_argument("--out", default="diag")
    ap.add_argument("--examples", type=int, default=40)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pdir = Path(args.pairs_dir)
    meta = json.loads((pdir / "meta.json").read_text())
    df = pd.read_parquet(pdir / "train_pairs.parquet", columns=["i", "j", "src", "label", "p"])
    q = np.load(pdir / "queries.npy")
    truth = pd.read_parquet(pdir / "truth.parquet")
    ti, tj = truth["i"].to_numpy(), truth["j"].to_numpy()
    n = int(max(df["i"].max(), df["j"].max(), ti.max(), tj.max())) + 1
    rule = meta["rule"]
    keep = E.decide(df, rule["t2"], rule["t3"], bool(rule["one_owner"]))
    pi, pj, pp = df["i"].to_numpy()[keep], df["j"].to_numpy()[keep], df["p"].to_numpy()[keep]
    lab = df["label"].to_numpy() == 1
    F = lambda a, b: E.per_entity_f05(q, ti, tj, a, b, n)  # noqa: E731
    nt = np.bincount(np.searchsorted(q, ti), minlength=len(q))

    # ---- 1. waterfall
    base = F(pi, pj)
    tp = keep & lab
    no_fp = F(df["i"].to_numpy()[tp], df["j"].to_numpy()[tp])
    all_cand = F(df["i"].to_numpy()[lab], df["j"].to_numpy()[lab])
    fix_fn = F(np.r_[pi, df["i"].to_numpy()[lab & ~keep]], np.r_[pj, df["j"].to_numpy()[lab & ~keep]])
    ckey = np.sort(df["i"].to_numpy().astype(np.int64) * n + df["j"].to_numpy())
    in_cand = E._in_sorted(ckey, ti.astype(np.int64) * n + tj)
    fix_miss = F(np.r_[pi, ti[~in_cand]], np.r_[pj, tj[~in_cand]])
    pts = lambda a, b: round(float((b - a).mean() * 100), 2)  # noqa: E731
    water = {
        "oof_macro_f05": round(float(base.mean()), 4),
        "points_lost_total": round(float((1 - base).mean() * 100), 2),
        "points_from_false_matches": pts(base, no_fp),
        "points_from_true_candidates_below_rule": pts(no_fp, all_cand),
        "points_from_never_candidates": pts(all_cand, np.ones_like(base)),
        "gain_if_only_below_rule_fixed": pts(base, fix_fn),
        "gain_if_only_never_candidates_fixed": pts(base, fix_miss),
        "union_recall_stage_b": round(meta.get("union_recall", float("nan")), 4),
        "kept_recall_stage_b": round(meta.get("kept_recall", float("nan")), 4),
        "true_pairs": int(len(ti)),
        "false_matches": int((keep & ~lab).sum()),
        "true_below_rule": int((lab & ~keep).sum()),
        "never_candidates": int((~in_cand).sum()),
    }

    # ---- 2. what the errors look like
    miss_i, miss_j = ti[~in_cand], tj[~in_cand]
    fn = lab & ~keep
    fp = keep & ~lab
    tn = ~keep & ~lab
    rng = np.random.default_rng(0)
    tn_idx = np.flatnonzero(tn)
    tn_idx = tn_idx[rng.permutation(len(tn_idx))[:200_000]]
    tp_idx = np.flatnonzero(tp)
    tp_idx = tp_idx[rng.permutation(len(tp_idx))[:200_000]]
    groups = {
        "never_candidate": (miss_i, miss_j, np.full(len(miss_i), np.nan)),
        "true_below_rule": (df["i"].to_numpy()[fn], df["j"].to_numpy()[fn], df["p"].to_numpy()[fn]),
        "false_match": (df["i"].to_numpy()[fp], df["j"].to_numpy()[fp], df["p"].to_numpy()[fp]),
        "found_match(sample)": (df["i"].to_numpy()[tp_idx], df["j"].to_numpy()[tp_idx], df["p"].to_numpy()[tp_idx]),
        "rejected_non_match(sample)": (df["i"].to_numpy()[tn_idx], df["j"].to_numpy()[tn_idx], df["p"].to_numpy()[tn_idx]),
    }
    rows = np.unique(np.concatenate([q, tj, df["j"].to_numpy()]))
    text = take_rows(args.train_records, rows, TEXT)
    text["skel"] = text["name_skel"] + " | " + text["addr_skel"]
    stats, examples = {}, []
    for g, (ci, cj, cp) in groups.items():
        if len(ci) == 0:
            continue
        s_name = sim(text.loc[ci, "name_canon"], text.loc[cj, "name_canon"])
        s_addr = sim(text.loc[ci, "addr_canon"], text.loc[cj, "addr_canon"])
        sib = sibling_sim(ci, cj, pi, pj, pp, text)
        has_sib = sib["full"] >= 0
        nt_c = nt[np.searchsorted(q, ci)]
        st = {
            "pairs": int(len(ci)),
            "cross_script_share": round(float(text.loc[cj, "non_ascii"].to_numpy().mean()), 3),
            "from_S3_share": round(float((text.loc[cj, "source"].to_numpy() == 3).mean()), 3),
            "s1_true_matches_median": float(np.median(nt_c)),
            "s1_has_4plus_share": round(float((nt_c >= 4).mean()), 3),
            "name_sim_to_s1_median": float(np.median(s_name)),
            "addr_sim_to_s1_median": float(np.median(s_addr)),
            "s1_has_predicted_match_share": round(float(has_sib.mean()), 3),
            "sibling_sim_ge90_share": round(float((sib["full"] >= 90).mean()), 3),
            "sibling_sim_ge95_share": round(float((sib["full"] >= 95).mean()), 3),
            "sibling_closer_than_s1_share": round(float((np.maximum(sib["name"], sib["addr"]) >
                                                         np.maximum(s_name, s_addr)).mean()), 3),
        }
        cc = text.loc[ci, "country"].to_numpy()
        st["country_share"] = {str(k): round(float(v), 3) for k, v in pd.Series(cc).value_counts(normalize=True).items()}
        stats[g] = st
        if "sample" in g:
            continue
        take = rng.permutation(len(ci))[:args.examples]
        # strongest predicted match of the same S1 (the "sibling"), for context
        o = np.lexsort((-pp, pi))
        spi, spj = pi[o], pj[o]
        first = np.searchsorted(spi, ci[take])
        ok = (first < len(spi)) & (spi[np.minimum(first, len(spi) - 1)] == ci[take])
        sib_row = np.where(ok, spj[np.minimum(first, len(spi) - 1)], -1)
        for t, srow in zip(take, sib_row):
            i, j = ci[t], cj[t]
            examples.append({
                "kind": g, "country": text.at[i, "country"], "p": None if np.isnan(cp[t]) else round(float(cp[t]), 3),
                "s1_true_matches": int(nt_c[t]), "name_sim": int(s_name[t]), "addr_sim": int(s_addr[t]),
                "sibling_sim": int(sib["full"][t]),
                "s1_name": text.at[i, "name_norm"], "s1_addr": text.at[i, "addr_norm"],
                "rec_name": text.at[j, "name_norm"], "rec_addr": text.at[j, "addr_norm"],
                "top_predicted_name": text.at[srow, "name_norm"] if srow >= 0 else "",
                "top_predicted_addr": text.at[srow, "addr_norm"] if srow >= 0 else ""})
    ex = pd.DataFrame(examples)
    ex.to_csv(out / "examples.tsv", sep="\t", index=False)

    # ---- 3. decision rules on the same Stage B scores
    thr = E.tune(df, q, ti, tj, n)
    per_src = E.tune(df, q, ti, tj, n, per_source=True)
    exp = E.tune_expected(df, q, ti, tj, n)
    rules = {"global_threshold": rounded(thr.iloc[0]),
             "per_source_threshold": rounded(per_src.iloc[0]),
             "per_s1_expected_f05": rounded(exp.iloc[0])}

    # ---- 4. points lost by country and S1 size
    cc = take_rows(args.train_records, q, ["country"])["country"].to_numpy()
    lost = pd.DataFrame({"country": cc, "size": pd.cut(nt, [-1, 0, 1, 3, 6, 10**6],
                                                       labels=["0 (singleton)", "1", "2-3", "4-6", "7+"]),
                         "lost": (1 - base) / len(q) * 100, "f05": base})
    by_c = lost.groupby("country").agg(s1=("f05", "size"), f05=("f05", "mean"), points_lost=("lost", "sum"))
    by_s = lost.groupby("size", observed=True).agg(s1=("f05", "size"), f05=("f05", "mean"), points_lost=("lost", "sum"))

    print("\n===== DIAGNOSIS (paste this) =====")
    print(json.dumps({"waterfall": water, "decision_rules": rules}, indent=1, default=float))
    print("\nerror groups:")
    print(json.dumps(stats, indent=1, default=float))
    print("\npoints lost by country:\n" + by_c.round(3).to_string())
    print("\npoints lost by number of true matches:\n" + by_s.round(3).to_string())
    print(f"\nexamples written to {out / 'examples.tsv'}; a few of each:")
    short = lambda s: (s[:48] + "…") if isinstance(s, str) and len(s) > 49 else s  # noqa: E731
    for g in ("never_candidate", "true_below_rule", "false_match"):
        sub = ex[ex["kind"] == g].head(8)
        print(f"\n--- {g}")
        for _, r in sub.iterrows():
            print(f"[{r['country']}, p={r['p']}, n={r['s1_true_matches']}, sim n/a/sib={r['name_sim']}/{r['addr_sim']}/{r['sibling_sim']}]")
            print(f"   S1 : {short(r['s1_name'])} | {short(r['s1_addr'])}")
            print(f"   rec: {short(r['rec_name'])} | {short(r['rec_addr'])}")
            if r["top_predicted_name"]:
                print(f"   sib: {short(r['top_predicted_name'])} | {short(r['top_predicted_addr'])}")


if __name__ == "__main__":
    main()
