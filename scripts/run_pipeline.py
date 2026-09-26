#!/usr/bin/env python3
"""End-to-end ER pipeline: blocking -> features -> LightGBM -> F0.5 decision.

  python scripts/run_pipeline.py cv      --data-root dataset --tag v1
  python scripts/run_pipeline.py predict --data-root dataset --tag v1

`cv` trains 5 grouped folds on train, tunes the decision rule on the
out-of-fold (OOF) probabilities of *every* train S1 and writes
artifacts/<tag>/ (cv_summary.json, slices, importances, oof pairs).
`predict` retrains on all train, applies the tuned rule from cv_summary.json
to test and writes output/candidate_pairs.tsv + output/matching_results.tsv.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from business_entity_resolution.er import blocking, decision, matcher, metrics, slices  # noqa: E402
from business_entity_resolution.er.data import load_split  # noqa: E402
from business_entity_resolution.er.pair_features import build_features, feature_columns  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def candidates_and_features(split, bcfg):
    t = time.time()
    cand = blocking.generate_candidates(split, bcfg)
    rep = blocking.blocking_report(split, cand)
    log(f"{split.name}: {len(cand):,} candidate pairs in {time.time() - t:.0f}s  "
        + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rep.items()))
    t = time.time()
    feats = build_features(split, cand)
    log(f"{split.name}: {feats.shape[1]} columns of features in {time.time() - t:.0f}s")
    return feats, rep


def write_outputs(split, feats, pred, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cands = feats.groupby("s1")["cand"].apply(lambda s: ",".join(sorted(set(s))))
    with open(out_dir / "candidate_pairs.tsv", "w", encoding="utf-8", newline="") as fh:
        fh.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1 in split.s1_ids:
            fh.write(f"{s1}\t{cands.get(s1, '')}\n")
    with open(out_dir / "matching_results.tsv", "w", encoding="utf-8", newline="") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for s1 in split.s1_ids:
            fh.write(f"{s1}\t{','.join(sorted(pred.get(s1, ())))}\n")


def cmd_cv(args, bcfg) -> None:
    art = ROOT / "artifacts" / args.tag
    art.mkdir(parents=True, exist_ok=True)
    tr = load_split(args.data_root, "train")
    log(f"train: {len(tr.s1_ids):,} S1, {len(tr.records):,} records")

    # sanity check for the one_owner rule: does any S2/S3 id belong to 2+ S1s?
    owners = pd.Series([c for cs in tr.truth.values() for c in cs]).value_counts()
    log(f"S2/S3 ids matched to >1 S1 in ground truth: {(owners > 1).sum()} of {len(owners)}")

    feats, rep = candidates_and_features(tr, bcfg)
    feats = matcher.add_labels(tr, feats)
    cols = feature_columns(feats)
    fold = matcher.fold_of_s1(tr, args.folds)
    t = time.time()
    oof, iters, imp = matcher.cross_validate(feats, cols, fold)
    feats["p"] = oof
    log(f"CV done in {time.time() - t:.0f}s, best iterations {iters}")

    table = decision.tune(feats, tr.truth, tr.s1_ids, per_source=args.per_source)
    best = table.iloc[0].to_dict()
    pred = decision.decide(feats, best["t2"], best["t3"], bool(best["one_owner"]))
    summ = metrics.pr_summary(tr.truth, pred, tr.s1_ids)
    scores = metrics.per_entity_scores(tr.truth, pred, tr.s1_ids)
    sl = metrics.slice_report(scores, slices.entity_slices(tr, feats))

    log(f"best rule: {best}")
    log("summary: " + json.dumps({k: round(v, 4) for k, v in summ.items()}))
    print(table.head(8).to_string())
    print(sl.to_string())
    print(imp.head(25).to_string())

    summary = {"tag": args.tag, "blocking": asdict(bcfg), "blocking_report": rep,
               "best_rule": best, "summary": summ, "best_iters": iters,
               "n_rounds_full": int(np.mean(iters) * 1.1), "features": cols}
    (art / "cv_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    sl.to_csv(art / "slices.csv", index=False)
    imp.to_csv(art / "importance.csv")
    table.to_csv(art / "threshold_table.csv", index=False)
    feats.assign(fold=fold.reindex(feats["s1"]).to_numpy()).to_parquet(art / "oof_pairs.parquet") \
        if args.save_oof else None
    log(f"artifacts -> {art}")


def cmd_predict(args, bcfg) -> None:
    art = ROOT / "artifacts" / args.tag
    summary = json.loads((art / "cv_summary.json").read_text())
    rule = summary["best_rule"]
    tr = load_split(args.data_root, "train")
    feats, _ = candidates_and_features(tr, bcfg)
    feats = matcher.add_labels(tr, feats)
    cols = summary["features"]
    model = matcher.fit_full(feats, cols, summary["n_rounds_full"])

    te = load_split(args.data_root, "test")
    log(f"test: {len(te.s1_ids):,} S1, {len(te.records):,} records, "
        f"countries={te.records['country'].value_counts().to_dict()}")
    tf, _ = candidates_and_features(te, bcfg)
    tf["p"] = model.predict(tf[cols])
    pred = decision.decide(tf, rule["t2"], rule["t3"], bool(rule["one_owner"]))
    out = ROOT / "output"
    write_outputs(te, tf, pred, out)
    n_pred = sum(len(v) for v in pred.values())
    log(f"wrote {out}: {n_pred:,} matches for {len(pred):,}/{len(te.s1_ids):,} S1 "
        f"({1 - len(pred) / len(te.s1_ids):.1%} predicted singleton); "
        f"avg candidates/S1 = {len(tf) / len(te.s1_ids):.2f}")
    by_c = tf.assign(country=te.records.loc[tf["s1"], "country"].to_numpy()) \
        .groupby("country")["p"].describe()
    print(by_c.to_string())
    model.save_model(str(art / "model_full.txt"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cv", "predict"])
    ap.add_argument("--data-root", default=str(ROOT / "dataset"))
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--per-source", action="store_true", help="tune separate S2/S3 thresholds")
    ap.add_argument("--save-oof", action="store_true")
    for k, v in asdict(blocking.BlockingConfig()).items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    args = ap.parse_args()
    bcfg = blocking.BlockingConfig(**{k: getattr(args, k) for k in asdict(blocking.BlockingConfig())})
    (cmd_cv if args.mode == "cv" else cmd_predict)(args, bcfg)


if __name__ == "__main__":
    main()
