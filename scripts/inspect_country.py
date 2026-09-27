#!/usr/bin/env python3
"""Label-free look at test predictions per country (France has no training
labels, so this is the only way to see how it behaves). CPU, ~5 minutes.

  python scripts/inspect_country.py --pairs-dir work/models/m1 \
      --test-records work/test/records.parquet --country France --out inspect

Per country: share of S1 with >= 1 predicted match, matches per S1, how
confident the best candidate is (quantiles of the top p per S1), how many S1
sit just under the threshold, and how similar their best candidate looks.
Then prints S1s of --country that got an EMPTY prediction, with their top-3
candidates, so we can see whether France fails at retrieval (candidates are
wrong) or at calibration (candidates look right but p is low).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from business_entity_resolution.er import evaluate as E  # noqa: E402
from business_entity_resolution.er.dataset import take_rows  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--test-records", required=True)
    ap.add_argument("--country", default="France")
    ap.add_argument("--examples", type=int, default=25)
    ap.add_argument("--out", default="inspect")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pdir = Path(args.pairs_dir)
    meta = json.loads((pdir / "meta.json").read_text())
    rule = meta["rule"]
    cols = ["i", "j", "src", "p", "name_tset", "addr_tset", "cos_full"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted((pdir / "test_pairs").glob("part-*.parquet"))],
                   ignore_index=True)
    rec = pq.read_table(args.test_records, columns=["source", "country"]).to_pandas()
    s1 = np.flatnonzero(rec["source"].to_numpy() == 1)
    country = rec["country"].to_numpy()
    keep = E.decide(df, rule["t2"], rule["t3"], bool(rule["one_owner"]))
    t = min(rule["t2"], rule["t3"])

    # best candidate per S1 (by p)
    top = df.sort_values(["i", "p"], ascending=[True, False]).drop_duplicates("i")
    s = pd.DataFrame({"i": s1, "country": country[s1]})
    s = s.merge(top[["i", "p", "name_tset", "addr_tset"]].rename(columns={"p": "top_p"}), on="i", how="left")
    s["n_pred"] = s["i"].map(pd.Series(df["i"].to_numpy()[keep]).value_counts()).fillna(0)
    s["n_cand"] = s["i"].map(df["i"].value_counts()).fillna(0)
    g = s.groupby("country")
    per = pd.DataFrame({
        "s1": g.size(),
        "with_match": g["n_pred"].apply(lambda x: (x > 0).mean()),
        "matches_per_s1": g["n_pred"].mean(),
        "cands_per_s1": g["n_cand"].mean(),
        "top_p_q10": g["top_p"].quantile(0.10),
        "top_p_q25": g["top_p"].quantile(0.25),
        "top_p_median": g["top_p"].median(),
        f"top_p_in_[0.3,{t})": g["top_p"].apply(lambda x: ((x >= 0.3) & (x < t)).mean()),
        "top_p_below_0.3": g["top_p"].apply(lambda x: (x.fillna(0) < 0.3).mean()),
        "top_name_tset_median": g["name_tset"].median(),
        "top_addr_tset_median": g["addr_tset"].median(),
    })
    # among S1 with no prediction: how similar is the best candidate?
    empty = s[s["n_pred"] == 0]
    ge = empty.groupby("country")
    per["empty_top_name_tset_median"] = ge["name_tset"].median()
    per["empty_top_addr_tset_median"] = ge["addr_tset"].median()

    print("\n===== COUNTRY INSPECTION (paste this) =====")
    print(f"rule: {rule}")
    print(per.round(3).T.to_string())

    # examples: S1 of --country with an empty prediction, top-3 candidates each
    rng = np.random.default_rng(0)
    pick = empty[empty["country"] == args.country]["i"].to_numpy()
    pick = pick[rng.permutation(len(pick))[:args.examples]]
    cand = df[np.isin(df["i"].to_numpy(), pick)].sort_values(["i", "p"], ascending=[True, False])
    cand = cand.groupby("i").head(3)
    text = take_rows(args.test_records, np.r_[pick, cand["j"].to_numpy()], ["name_norm", "addr_norm"])
    short = lambda x: (x[:55] + "…") if len(x) > 56 else x  # noqa: E731
    rows = []
    print(f"\n--- {args.country}: S1 with NO predicted match ({len(pick)} random of {int((empty['country'] == args.country).sum()):,})")
    for i in pick:
        print(f"S1 : {short(text.at[i, 'name_norm'])} | {short(text.at[i, 'addr_norm'])}")
        for _, r in cand[cand["i"] == i].iterrows():
            j = int(r["j"])
            print(f"   p={r['p']:.3f} S{int(r['src'])} n/a={r['name_tset']:.2f}/{r['addr_tset']:.2f}  "
                  f"{short(text.at[j, 'name_norm'])} | {short(text.at[j, 'addr_norm'])}")
            rows.append({"s1_row": i, "s1_name": text.at[i, "name_norm"], "s1_addr": text.at[i, "addr_norm"],
                         "p": r["p"], "src": int(r["src"]), "rec_name": text.at[j, "name_norm"],
                         "rec_addr": text.at[j, "addr_norm"]})
        if not (cand["i"] == i).any():
            print("   (no candidates at all)")
    pd.DataFrame(rows).to_csv(out / f"empty_{args.country}.tsv", sep="\t", index=False)
    per.to_csv(out / "per_country.tsv", sep="\t")


if __name__ == "__main__":
    main()
