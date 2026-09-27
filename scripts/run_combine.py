#!/usr/bin/env python3
"""Stacking: add extra pair scores (e.g. the cross-encoder) to the cached Stage B
tables, retrain the LightGBM matcher, re-tune the decision rule, write the
submission. No retrieval is repeated, so this takes minutes on CPU.

  python scripts/run_combine.py --pairs-dir work/models/m1 \
      --extra-train ce_out/ce_train.parquet --extra-test ce_out/ce_test.parquet \
      --train-records work/train/records.parquet --test-records work/test/records.parquet \
      --test-dir dataset/test --out output

Also adds sibling features (--train-records) and tries two decision rules (one
global threshold vs a per-S1 expected-F0.5 choice). Prints the out-of-fold macro
F0.5 of the Stage B matcher and of each stacked variant on the same S1s (all,
without siblings, without the extra score), keeps the best, writes the files.
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
from business_entity_resolution.er import features as F  # noqa: E402
from business_entity_resolution.er import model as M  # noqa: E402
from business_entity_resolution.er.dataset import take_rows  # noqa: E402
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


def pick_rule(df, q, ti, tj, n, prob, per_source=False) -> tuple[dict, pd.DataFrame]:
    """Best of: one global threshold (per source optional) vs the per-S1 expected-F0.5 rule."""
    thr = E.tune(df, q, ti, tj, n, per_source=per_source, prob=prob)
    exp = E.tune_expected(df, q, ti, tj, n, prob=prob)
    best_thr = {"type": "threshold", **thr.iloc[0].to_dict()}
    best_exp = {"type": "expected_f05", **exp.iloc[0].to_dict()}
    both = pd.DataFrame([best_thr, best_exp])
    return (best_exp if best_exp["macro_f05"] > best_thr["macro_f05"] else best_thr), both


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--extra-train", default=None)
    ap.add_argument("--extra-test", default=None)
    ap.add_argument("--train-records", default=None, help="train records.parquet: enables sibling features")
    ap.add_argument("--test-records", required=True)
    ap.add_argument("--no-siblings", action="store_true")
    ap.add_argument("--test-dir", default=None, help="raw test TSVs, for the submission check")
    ap.add_argument("--out", default=str(ROOT / "output"))
    ap.add_argument("--per-source", action="store_true", help="tune separate S2 / S3 thresholds")
    ap.add_argument("--min-p", type=float, default=0.01, help="focus set: first-stage p threshold")
    ap.add_argument("--top-k", type=int, default=8, help="focus set: max pairs per S1")
    ap.add_argument("--workers", type=int, default=default_workers())
    args = ap.parse_args()
    use_sib = bool(args.train_records) and not args.no_siblings

    pdir = Path(args.pairs_dir)
    meta = json.loads((pdir / "meta.json").read_text())
    tr = pd.read_parquet(pdir / "train_pairs.parquet")
    n_pos = int(tr["label"].sum())
    tr = tr[E.focus_set(tr, args.min_p, args.top_k)].reset_index(drop=True)
    q = np.load(pdir / "queries.npy")
    truth = pd.read_parquet(pdir / "truth.parquet")
    ti, tj = truth["i"].to_numpy(), truth["j"].to_numpy()
    n = int(max(tr["i"].max(), tr["j"].max(), ti.max(), tj.max())) + 1

    log(f"focus set (p >= {args.min_p}, top {args.top_k}/S1): {len(tr) / len(q):.2f} pairs/S1, "
        f"keeps {tr['label'].sum() / max(1, n_pos):.4f} of the Stage B train positives")
    base_rule, base_both = pick_rule(tr, q, ti, tj, n, "p", args.per_source)
    log("Stage B matcher (OOF), best threshold vs per-S1 expected-F rule:\n" + base_both.round(4).to_string())

    extra_tr = pd.read_parquet(args.extra_train) if args.extra_train else None
    tr, new_cols = add_extra(tr, extra_tr)
    sib_cols: list[str] = []
    if use_sib:
        sib = F.sibling_features(tr, take_rows(args.train_records, tr["j"].to_numpy(), F.SIB_TEXT))
        tr = pd.concat([tr, sib], axis=1)
        sib_cols = list(sib.columns)
        log(f"sibling features: {sib_cols}")

    # stacked matcher variants, all scored out-of-fold on the same S1s
    base_cols = meta["full_columns"]
    variants = {"all": base_cols + new_cols + sib_cols}
    if new_cols and sib_cols:
        variants["no_siblings"] = base_cols + new_cols
        variants["no_extra"] = base_cols + sib_cols
    results = {}
    for name, cols in variants.items():
        log(f"stacked matcher [{name}]: {len(cols)} features on {len(tr):,} pairs")
        oof, iters, imp = M.oof(tr, cols, M.MATCH_PARAMS, 5, workers=args.workers, log=log)
        tr[f"p_{name}"] = oof
        rule, both = pick_rule(tr, q, ti, tj, n, f"p_{name}", args.per_source)
        results[name] = {"cols": cols, "iters": iters, "imp": imp, "rule": rule, "both": both}
        log(f"  [{name}] OOF macro F0.5: threshold {both['macro_f05'].iloc[0]:.4f}, "
            f"expected-F {both['macro_f05'].iloc[1]:.4f}")
    name = max(results, key=lambda k: results[k]["rule"]["macro_f05"])
    chosen = results[name]
    cols, best = chosen["cols"], chosen["rule"]
    tr["p2"] = tr[f"p_{name}"]
    keep = E.apply_rule(tr, best, "p2")
    summ = E.summary(q, ti, tj, tr["i"].to_numpy()[keep], tr["j"].to_numpy()[keep], n)
    log(f"chosen [{name}] with {best['type']} rule: OOF macro F0.5 {summ['macro_f05']:.4f} "
        f"(Stage B alone: {base_rule['macro_f05']:.4f})")
    model = M.fit(tr, cols, M.MATCH_PARAMS, int(np.mean(chosen["iters"]) * 1.1), args.workers)

    # ---- test
    extra_te = pd.read_parquet(args.extra_test) if args.extra_test else None
    files = sorted((pdir / "test_pairs").glob("part-*.parquet"))
    text_te = None
    if use_sib and sib_cols and any(c in cols for c in sib_cols):
        js = []
        for f in files:
            te = pd.read_parquet(f, columns=["i", "j", "p"])
            js.append(te.loc[E.focus_set(te, args.min_p, args.top_k), "j"].to_numpy())
        text_te = take_rows(args.test_records, np.concatenate(js), F.SIB_TEXT)
        log(f"test text loaded for {len(text_te):,} candidate records")
    parts = []
    for f in files:
        te = pd.read_parquet(f)
        te = te[E.focus_set(te, args.min_p, args.top_k)].reset_index(drop=True)
        if extra_te is not None:
            lo, hi = te["i"].min(), te["i"].max()
            te, _ = add_extra(te, extra_te[(extra_te["i"] >= lo) & (extra_te["i"] <= hi)])
        if text_te is not None:
            te = pd.concat([te, F.sibling_features(te, text_te)], axis=1)
        te["p2"] = model.predict(te[cols])
        parts.append(te[["i", "j", "src", "p2"]])
    allc = pd.concat(parts, ignore_index=True)
    pred = allc[E.apply_rule(allc, best, "p2")]

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
    print("\n" + chosen["imp"].head(20).round(0).to_string())
    print("\n===== COMBINE SUMMARY (paste this) =====")
    print(json.dumps({"stage_b_oof_f05": round(float(base_rule["macro_f05"]), 4),
                      "stage_b_rule": base_rule["type"],
                      "variants_oof_f05": {k: round(float(v["rule"]["macro_f05"]), 4) for k, v in results.items()},
                      "chosen": name, "rule": best,
                      "test_s1": len(qt), "candidates_per_s1": round(len(allc) / len(qt), 2),
                      "matches_per_s1": round(len(pred) / len(qt), 3),
                      **{k: round(v, 4) for k, v in summ.items()}}, indent=1, default=float))
    if args.test_dir:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "check_submission.py"),
                        "--out", str(out), "--test-dir", args.test_dir], check=False)


if __name__ == "__main__":
    main()
