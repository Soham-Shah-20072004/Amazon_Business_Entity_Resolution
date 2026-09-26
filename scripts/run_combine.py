#!/usr/bin/env python3
"""Stacking: add extra pair scores (e.g. the cross-encoder) to the cached Stage B
tables, retrain the LightGBM matcher, re-tune the decision rule, write the
submission. No retrieval is repeated, so this takes minutes on CPU.

  python scripts/run_combine.py --pairs-dir work/models/m1 \
      --extra-train ce_out/ce_train.parquet --extra-test ce_out/ce_test.parquet \
      --test-records work/test/records.parquet --test-dir dataset/test --out output

Prints the out-of-fold macro F0.5 of the Stage B matcher and of the stacked one
on the same S1s, so we can see whether the extra score actually helps.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from business_entity_resolution.er import evaluate as E  # noqa: E402
from business_entity_resolution.er import model as M  # noqa: E402
from business_entity_resolution.er.prepare import default_workers  # noqa: E402

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f}m] {msg}", flush=True)


def add_extra(df: pd.DataFrame, extra: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    """Merge extra score columns on (i, j) and add their rank / gap inside each S1 list."""
    if extra is None:
        return df, []
    score_cols = [c for c in extra.columns if c not in ("i", "j")]
    df = df.merge(extra, on=["i", "j"], how="left")
    new = []
    g = df.groupby(["i", "src"], sort=False)
    for c in score_cols:
        df[c] = df[c].fillna(0).astype(np.float32)
        df[f"{c}_rank"] = g[c].rank(ascending=False, method="min").astype(np.float32)
        df[f"{c}_gap"] = (g[c].transform("max") - df[c]).astype(np.float32)
        new += [c, f"{c}_rank", f"{c}_gap"]
    return df, new


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--extra-train", default=None)
    ap.add_argument("--extra-test", default=None)
    ap.add_argument("--test-records", required=True)
    ap.add_argument("--test-dir", default=None, help="raw test TSVs, for the submission check")
    ap.add_argument("--out", default=str(ROOT / "output"))
    ap.add_argument("--per-source", action="store_true", help="tune separate S2 / S3 thresholds")
    ap.add_argument("--workers", type=int, default=default_workers())
    args = ap.parse_args()

    pdir = Path(args.pairs_dir)
    meta = json.loads((pdir / "meta.json").read_text())
    tr = pd.read_parquet(pdir / "train_pairs.parquet")
    q = np.load(pdir / "queries.npy")
    truth = pd.read_parquet(pdir / "truth.parquet")
    ti, tj = truth["i"].to_numpy(), truth["j"].to_numpy()
    n = int(max(tr["i"].max(), tr["j"].max(), ti.max(), tj.max())) + 1

    base = E.tune(tr, q, ti, tj, n, per_source=args.per_source, prob="p").iloc[0]
    log(f"Stage B matcher (OOF): macro F0.5 {base['macro_f05']:.4f}")

    extra_tr = pd.read_parquet(args.extra_train) if args.extra_train else None
    tr, new_cols = add_extra(tr, extra_tr)
    cols = meta["full_columns"] + new_cols
    log(f"stacked matcher: {len(cols)} features ({len(new_cols)} new: {new_cols}) on {len(tr):,} pairs")
    oof, iters, imp = M.oof(tr, cols, M.MATCH_PARAMS, 5, workers=args.workers, log=log)
    tr["p2"] = oof
    table = E.tune(tr, q, ti, tj, n, per_source=args.per_source, prob="p2")
    best = table.iloc[0].to_dict()
    keep = E.decide(tr, best["t2"], best["t3"], bool(best["one_owner"]), prob="p2")
    summ = E.summary(q, ti, tj, tr["i"].to_numpy()[keep], tr["j"].to_numpy()[keep], n)
    log(f"stacked matcher (OOF): macro F0.5 {summ['macro_f05']:.4f}  (Stage B: {base['macro_f05']:.4f})")
    model = M.fit(tr, cols, M.MATCH_PARAMS, int(np.mean(iters) * 1.1), args.workers)

    # ---- test
    extra_te = pd.read_parquet(args.extra_test) if args.extra_test else None
    parts = []
    for f in sorted((pdir / "test_pairs").glob("part-*.parquet")):
        te = pd.read_parquet(f)
        if extra_te is not None:
            lo, hi = te["i"].min(), te["i"].max()
            te, _ = add_extra(te, extra_te[(extra_te["i"] >= lo) & (extra_te["i"] <= hi)])
        te["p2"] = model.predict(te[cols])
        parts.append(te[["i", "j", "src", "p2"]])
    allc = pd.concat(parts, ignore_index=True)
    pred = allc[E.decide(allc, best["t2"], best["t3"], bool(best["one_owner"]), prob="p2")]

    meta_rec = pq.read_table(args.test_records, columns=["entity_id", "source"]).to_pandas()
    ids = meta_rec["entity_id"].to_numpy(dtype=object)
    qt = np.flatnonzero(meta_rec["source"].to_numpy() == 1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def write(df: pd.DataFrame, path: Path, header: str) -> None:
        lists = (pd.DataFrame({"i": df["i"].to_numpy(), "id": ids[df["j"].to_numpy()]})
                 .sort_values(["i", "id"]).groupby("i")["id"].agg(",".join))
        col = lists.reindex(qt).fillna("").to_numpy()
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(f"source1_entity_id\t{header}\n")
            fh.writelines(f"{a}\t{b}\n" for a, b in zip(ids[qt], col))

    write(allc, out / "candidate_pairs.tsv", "candidate_entity_ids")
    write(pred, out / "matching_results.tsv", "matched_entity_ids")
    print("\n" + imp.head(15).round(0).to_string())
    print("\n===== COMBINE SUMMARY (paste this) =====")
    print(json.dumps({"stage_b_oof_f05": round(float(base["macro_f05"]), 4),
                      "stacked_oof_f05": round(summ["macro_f05"], 4), "rule": best,
                      "test_s1": len(qt), "candidates_per_s1": round(len(allc) / len(qt), 2),
                      "matches_per_s1": round(len(pred) / len(qt), 3),
                      **{k: round(v, 4) for k, v in summ.items()}}, indent=1, default=float))
    if args.test_dir:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "check_submission.py"),
                        "--out", str(out), "--test-dir", args.test_dir], check=False)


if __name__ == "__main__":
    main()
