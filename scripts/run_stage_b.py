#!/usr/bin/env python3
"""Stage B: blocking -> pre-ranker -> matcher -> decision -> submission files.

  # train on a sample of train S1s (full S2/S3 pool), tune everything on OOF scores
  python scripts/run_stage_b.py train   --work work --s1-sample 100000 --tag m1
  # apply the saved models to every test S1 and write output/*.tsv
  python scripts/run_stage_b.py predict --work work --tag m1 --out output

Pipeline per S1 (see docs/pipeline_overview.png):
  union of blockers (~100 candidates) -> pre-ranker (cheap features) keeps the
  few plausible ones (= candidate_pairs.tsv) -> matcher (full features) gives
  P(match) -> threshold + one-owner rule -> matching_results.tsv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from business_entity_resolution.er import evaluate as E  # noqa: E402
from business_entity_resolution.er import model as M  # noqa: E402
from business_entity_resolution.er.dataset import RETRIEVAL_COLUMNS, load_split  # noqa: E402
from business_entity_resolution.er.features import (  # noqa: E402
    MATCH_COLUMNS, cheap_columns, cheap_features, feature_columns, full_features)
from business_entity_resolution.er.prepare import default_workers  # noqa: E402
from business_entity_resolution.er.retrieval import (  # noqa: E402
    RetrievalConfig, assemble, build_resources, reverse_best, search)

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f}m] {msg}", flush=True)


# ---------------------------------------------------------------- shared

def sort_found(found: dict) -> dict:
    out = {}
    for b, (i, j) in found.items():
        o = np.argsort(i, kind="stable")
        out[b] = (i[o], j[o])
    return out


def slice_found(found_sorted: dict, lo: int, hi: int) -> dict:
    """Pairs whose S1 row is in [lo, hi] (found arrays sorted by i)."""
    out = {}
    for b, (i, j) in found_sorted.items():
        a, z = np.searchsorted(i, lo, "left"), np.searchsorted(i, hi, "right")
        out[b] = (i[a:z], j[a:z])
    return out


def retrieve(args, cfg, split_name, s1_sample):
    split = load_split(args.work, split_name, s1_sample=s1_sample,
                       columns=RETRIEVAL_COLUMNS + MATCH_COLUMNS)
    log(f"{split_name}: {len(split.rec):,} records, {len(split.query):,} S1 queries")
    res = build_resources(split, cfg, log)
    found = sort_found(search(split, res, cfg, split.query, log))
    all_j = np.unique(np.concatenate([j for _, j in found.values()]))
    rev = reverse_best(split, res, cfg, all_j)
    log(f"reverse check done for {len(all_j):,} candidate records")
    return split, res, found, rev


def prerank_keep(df: pd.DataFrame, t_pre: float, k_pre: int) -> np.ndarray:
    rank = df.groupby("i", sort=False)["pre_p"].rank(ascending=False, method="first").to_numpy()
    return (df["pre_p"].to_numpy() >= t_pre) & (rank <= k_pre)


def label(split, df: pd.DataFrame) -> np.ndarray:
    n = len(split.rec)
    tk = np.sort(split.truth["i"].to_numpy() * n + split.truth["j"].to_numpy())
    return E._in_sorted(tk, df["i"].to_numpy() * n + df["j"].to_numpy()).astype(np.int8)


# ---------------------------------------------------------------- train

def cmd_train(args, cfg) -> None:
    out = Path(args.work) / "models" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    split, res, found, rev = retrieve(args, cfg, "train", args.s1_sample)
    n, q = len(split.rec), split.query
    ti, tj = split.truth["i"].to_numpy(), split.truth["j"].to_numpy()

    cand = assemble(split, res, cfg, found)
    cheap = cheap_features(cand, rev, cfg.views)
    del cand
    cheap["label"] = label(split, cheap)
    ccols = cheap_columns(cheap)
    n_pos_union = int(cheap["label"].sum())
    log(f"union: {len(cheap):,} pairs ({len(cheap) / len(q):.1f}/S1), recall {n_pos_union / len(ti):.4f}")

    # pre-ranker: OOF on all pairs, trained on all positives + a share of negatives
    rng = np.random.default_rng(0)
    train_rows = (cheap["label"].to_numpy() == 1) | (rng.random(len(cheap)) < args.pre_neg_frac)
    log(f"pre-ranker: {len(ccols)} cheap features, training rows {int(train_rows.sum()):,}")
    pre_oof = np.zeros(len(cheap), np.float32)
    fold = M.fold_of(cheap["i"].to_numpy(), 3)
    iters = []
    import lightgbm as lgb
    for k in range(3):
        tr, va = train_rows & (fold != k), fold == k
        m = lgb.train({**M.PRE_PARAMS, "num_threads": cfg.workers},
                      lgb.Dataset(cheap.loc[tr, ccols], cheap.loc[tr, "label"]), 2000,
                      valid_sets=[lgb.Dataset(cheap.loc[va & train_rows, ccols],
                                              cheap.loc[va & train_rows, "label"])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        pre_oof[va] = m.predict(cheap.loc[va, ccols], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
    cheap["pre_p"] = pre_oof
    pre_model = lgb.train({**M.PRE_PARAMS, "num_threads": cfg.workers},
                          lgb.Dataset(cheap.loc[train_rows, ccols], cheap.loc[train_rows, "label"]),
                          int(np.mean(iters) * 1.1) + 1)
    pre_model.save_model(str(out / "prerank.txt"))

    # smallest candidate set that loses <= max_loss of the positives the union found
    rows = []
    for k_pre in (3, 5, 8, 10, 15, 20, 30):
        for t_pre in (0.0, 0.001, 0.003, 0.01, 0.03):
            keep = prerank_keep(cheap, t_pre, k_pre)
            rows.append({"k_pre": k_pre, "t_pre": t_pre, "per_s1": keep.sum() / len(q),
                         "loss": 1 - cheap["label"].to_numpy()[keep].sum() / max(1, n_pos_union)})
    grid = pd.DataFrame(rows)
    ok = grid[grid["loss"] <= args.max_prerank_loss]
    choice = (ok if len(ok) else grid.sort_values("loss").head(1)).sort_values("per_s1").iloc[0]
    k_pre, t_pre = int(choice["k_pre"]), float(choice["t_pre"])
    keep = prerank_keep(cheap, t_pre, k_pre)
    red = cheap[keep].reset_index(drop=True)
    log(f"pre-ranker keeps k<={k_pre}, p>={t_pre}: {len(red) / len(q):.2f}/S1, "
        f"recall {red['label'].sum() / len(ti):.4f}")
    print(grid.pivot(index="k_pre", columns="t_pre", values="per_s1").round(2).to_string())

    # matcher on the survivors
    full = full_features(split.rec, res, red, cfg.workers)
    fcols = feature_columns(full)
    log(f"matcher: {len(fcols)} features on {len(full):,} pairs")
    p_oof, iters_m, imp = M.oof(full, fcols, M.MATCH_PARAMS, 5, workers=cfg.workers, log=log)
    full["p"] = p_oof
    match_model = M.fit(full, fcols, M.MATCH_PARAMS, int(np.mean(iters_m) * 1.1), cfg.workers)
    match_model.save_model(str(out / "matcher.txt"))

    table = E.tune(full, q, ti, tj, n, per_source=False)
    best = table.iloc[0].to_dict()
    kept = E.decide(full, best["t2"], best["t3"], bool(best["one_owner"]))
    pi, pj = full["i"].to_numpy()[kept], full["j"].to_numpy()[kept]
    summ = E.summary(q, ti, tj, pi, pj, n)
    scores = E.per_entity_f05(q, ti, tj, pi, pj, n)

    # slices
    rec = split.rec
    nt = np.bincount(np.searchsorted(q, ti), minlength=len(q))
    in_union = E._in_sorted(np.sort(cheap["i"].to_numpy() * n + cheap["j"].to_numpy()), ti * n + tj)
    in_kept = E._in_sorted(np.sort(red["i"].to_numpy() * n + red["j"].to_numpy()), ti * n + tj)
    miss_block = np.bincount(np.searchsorted(q, ti[~in_union]), minlength=len(q)) > 0
    miss_pre = np.bincount(np.searchsorted(q, ti[in_union & ~in_kept]), minlength=len(q)) > 0
    xscript = np.bincount(np.searchsorted(q, ti[rec["non_ascii"].to_numpy()[tj] == 1]), minlength=len(q)) > 0
    neg = full[full["label"] == 0]
    same_name = np.isin(q, neg.loc[neg["name_core_exact"] == 1, "i"].to_numpy())
    cc = rec["country_code"].to_numpy()[q]
    sl = {"ALL": np.ones(len(q), bool), "singleton": nt == 0, "1 match": nt == 1,
          "2-3 matches": (nt >= 2) & (nt <= 3), "4+ matches": nt >= 4,
          "has cross-script match": xscript, "blocking missed a match": miss_block,
          "pre-ranker dropped a match": miss_pre, "same-name distractor": same_name,
          "S1 address has no numbers": rec["addr_nums"].str.len().to_numpy(na_value=0)[q] == 0}
    for c in np.unique(cc):
        sl[f"country={split.countries[c]}"] = cc == c
    slices = E.slice_report(scores, sl)

    meta = {"tag": args.tag, "s1_sample": args.s1_sample, "retrieval": asdict(cfg),
            "cheap_columns": ccols, "full_columns": fcols, "k_pre": k_pre, "t_pre": t_pre,
            "rule": best, "summary": summ,
            "union_recall": n_pos_union / len(ti), "kept_recall": float(red["label"].sum() / len(ti)),
            "kept_per_s1": len(red) / len(q), "union_per_s1": len(cheap) / len(q)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
    if args.save_oof:
        full[["i", "j", "src", "label", "pre_p", "p"]].to_parquet(out / "oof.parquet", index=False)

    print("\n" + table.head(6).to_string())
    print("\n" + slices.to_string())
    print("\n" + imp.head(20).round(0).to_string())
    print("\n===== REPORT (paste this) =====")
    print(json.dumps({"tag": args.tag, "s1_sample": args.s1_sample, "minutes": round((time.time() - T0) / 60, 1),
                      "union_per_s1": round(meta["union_per_s1"], 1), "union_recall": round(meta["union_recall"], 4),
                      "kept_per_s1": round(meta["kept_per_s1"], 2), "kept_recall": round(meta["kept_recall"], 4),
                      "k_pre": k_pre, "t_pre": t_pre, "rule": best,
                      **{k: round(v, 4) for k, v in summ.items()}}, indent=1, default=float))
    print("slices (points lost out of 100):")
    print(slices[["slice", "n", "f05", "points_lost"]].round(3).to_string(index=False))


# ---------------------------------------------------------------- predict

def cmd_predict(args, cfg) -> None:
    mdir = Path(args.work) / "models" / args.tag
    meta = json.loads((mdir / "meta.json").read_text())
    cfg = RetrievalConfig(**{k: v for k, v in meta["retrieval"].items()
                             if k in {f.name for f in fields(RetrievalConfig) if f.init}}
                          | {"workers": cfg.workers, "scratch": cfg.scratch})
    import lightgbm as lgb
    pre_model = lgb.Booster(model_file=str(mdir / "prerank.txt"))
    match_model = lgb.Booster(model_file=str(mdir / "matcher.txt"))
    split, res, found, rev = retrieve(args, cfg, "test", None)
    q = split.query
    kept_parts = []
    n_union = 0
    for s in range(0, len(q), args.chunk):
        lo, hi = q[s], q[min(s + args.chunk, len(q)) - 1]
        cand = assemble(split, res, cfg, slice_found(found, lo, hi))
        n_union += len(cand)
        cheap = cheap_features(cand, rev, cfg.views)
        del cand
        cheap["pre_p"] = pre_model.predict(cheap[meta["cheap_columns"]])
        red = cheap[prerank_keep(cheap, meta["t_pre"], meta["k_pre"])].reset_index(drop=True)
        del cheap
        full = full_features(split.rec, res, red, cfg.workers)
        red["p"] = match_model.predict(full[meta["full_columns"]])
        kept_parts.append(red[["i", "j", "src", "pre_p", "p"]])
        log(f"  S1 {s + 1:,}-{min(s + args.chunk, len(q)):,}: {len(red):,} candidates kept")
    allc = pd.concat(kept_parts, ignore_index=True)
    rule = meta["rule"]
    keep = E.decide(allc, rule["t2"], rule["t3"], bool(rule["one_owner"]))
    pred = allc[keep]

    ids = split.rec["entity_id"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s1_ids = ids.iloc[q].to_numpy(dtype=object)

    def write(df: pd.DataFrame, path: Path, header: str) -> None:
        lists = (pd.DataFrame({"i": df["i"].to_numpy(), "id": ids.iloc[df["j"].to_numpy()].to_numpy(dtype=object)})
                 .sort_values(["i", "id"]).groupby("i")["id"].agg(",".join))
        col = lists.reindex(q).fillna("").to_numpy()
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(f"source1_entity_id\t{header}\n")
            fh.writelines(f"{a}\t{b}\n" for a, b in zip(s1_ids, col))

    write(allc, out / "candidate_pairs.tsv", "candidate_entity_ids")
    write(pred, out / "matching_results.tsv", "matched_entity_ids")

    cc = split.rec["country_code"].to_numpy()
    per = pd.DataFrame({"country": [split.countries[c] for c in cc[q]],
                        "cands": np.bincount(np.searchsorted(q, allc["i"]), minlength=len(q)),
                        "matches": np.bincount(np.searchsorted(q, pred["i"]), minlength=len(q))})
    by_c = per.groupby("country").agg(s1=("cands", "size"), cands_per_s1=("cands", "mean"),
                                      matches_per_s1=("matches", "mean"),
                                      share_with_match=("matches", lambda x: (x > 0).mean()))
    print("\n" + by_c.round(3).to_string())
    print("\n===== SUBMISSION SUMMARY (paste this) =====")
    print(json.dumps({"tag": args.tag, "minutes": round((time.time() - T0) / 60, 1), "test_s1": len(q),
                      "union_per_s1": round(n_union / len(q), 1), "candidates_per_s1": round(len(allc) / len(q), 2),
                      "matches_per_s1": round(len(pred) / len(q), 3),
                      "share_s1_with_match": round(float((per["matches"] > 0).mean()), 4),
                      "train_oof": meta["summary"]}, indent=1, default=float))
    log(f"wrote {out}/candidate_pairs.tsv and matching_results.tsv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "predict"])
    ap.add_argument("--work", default=str(ROOT / "work"))
    ap.add_argument("--tag", default="m1")
    ap.add_argument("--s1-sample", type=float, default=100_000)
    ap.add_argument("--out", default=str(ROOT / "output"))
    ap.add_argument("--chunk", type=int, default=150_000, help="test S1s per batch")
    ap.add_argument("--pre-neg-frac", type=float, default=0.3)
    ap.add_argument("--max-prerank-loss", type=float, default=0.003)
    ap.add_argument("--save-oof", action="store_true")
    ap.add_argument("--views", nargs="+", default=list(RetrievalConfig.views))
    for f in fields(RetrievalConfig):
        if f.init and f.name != "views":
            ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    args = ap.parse_args()
    if args.workers == RetrievalConfig.workers:
        args.workers = default_workers()
    cfg = RetrievalConfig(**{f.name: getattr(args, f.name) for f in fields(RetrievalConfig)
                             if f.init and f.name != "views"}, views=tuple(args.views))
    (cmd_train if args.mode == "train" else cmd_predict)(args, cfg)


if __name__ == "__main__":
    main()
