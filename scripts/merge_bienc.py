#!/usr/bin/env python3
"""Add the bi-encoder's extra candidates to the stacked result, only where it helps.

  python scripts/merge_bienc.py --work work --stack-dir output --pairs-dir work/models/m2 \
      --bienc-dir bienc_out --test-dir dataset/test --out output_merged

Inputs: the combine step's scores (stack_train/test.parquet, stack_rule.json),
the bi-encoder pairs (bienc_train/test.parquet) and the prepared records.
Steps:
  1. new pairs = bi-encoder pairs the stacked pipeline never scored
  2. features: string / number similarities (same functions as the matcher),
     bi-encoder cosine and ranks (forward, reverse, propagation), and context
     (how confident the S1's best current match is, whether the record is
     already given to another S1)
  3. a LightGBM trained out-of-fold on the held-out train S1s (the bi-encoder
     never trained on them) scores the new pairs
  4. threshold t chosen on the SAME held-out S1s to maximise macro F0.5 of
     (current predictions + new pairs with p >= t); nothing is added unless it
     beats the current score
Prints a MERGE SUMMARY with the out-of-fold score before and after.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from business_entity_resolution.er import evaluate as E  # noqa: E402
from business_entity_resolution.er import model as M  # noqa: E402
from business_entity_resolution.er.dataset import load_split  # noqa: E402
from business_entity_resolution.er.features import numeric_features, string_features  # noqa: E402
from business_entity_resolution.er.prepare import default_workers  # noqa: E402

T0 = time.time()
REC_COLS = ["entity_id", "source", "country_norm", "name_canon", "addr_canon", "name_core", "name_skel",
            "addr_skel", "name_acr", "addr_nums", "addr_post"]
BI_COLS = ["cos", "fwd_rank", "rev_rank", "hop", "hop_cos"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f}m] {msg}", flush=True)


def new_pairs(bi: pd.DataFrame, stack: pd.DataFrame, n: int, per_s1: int) -> pd.DataFrame:
    """Bi-encoder pairs the stacked pipeline did not score; at most per_s1 per S1 (best cosine first)."""
    sk = np.sort(stack["i"].to_numpy().astype(np.int64) * n + stack["j"].to_numpy())
    bk = bi["i"].to_numpy().astype(np.int64) * n + bi["j"].to_numpy()
    bi = bi[~E._in_sorted(sk, bk)]
    bi = bi.sort_values(["i", "cos"], ascending=[True, False])
    return bi.groupby("i", sort=False).head(per_s1).reset_index(drop=True)


def context(nw: pd.DataFrame, stack: pd.DataFrame, base_keep: np.ndarray) -> pd.DataFrame:
    top = stack.groupby("i")["p2"].max()
    npred = pd.Series(base_keep.astype(np.int32), index=stack.index).groupby(stack["i"]).sum()
    taken = stack.loc[base_keep].groupby("j")["p2"].max()
    ctx = pd.DataFrame(index=nw.index)
    ctx["s1_top_p2"] = nw["i"].map(top).fillna(0).to_numpy(np.float32)
    ctx["s1_n_pred"] = nw["i"].map(npred).fillna(0).to_numpy(np.float32)
    ctx["j_taken_p2"] = nw["j"].map(taken).fillna(0).to_numpy(np.float32)
    ctx["j_taken"] = (ctx["j_taken_p2"] > 0).astype(np.int8)
    return ctx


def features(split, nw: pd.DataFrame, ctx: pd.DataFrame, workers: int) -> pd.DataFrame:
    ii, jj = nw["i"].to_numpy(), nw["j"].to_numpy()
    f = pd.concat([nw[BI_COLS].reset_index(drop=True), ctx.reset_index(drop=True),
                   string_features(split.rec, ii, jj), numeric_features(split.rec, ii, jj, workers)], axis=1)
    f["src"] = split.rec["source"].to_numpy()[jj]
    return f


def decide_adds(nw: pd.DataFrame, p: np.ndarray, t: float) -> np.ndarray:
    """New pairs with p >= t whose record no current prediction owns; one owner among them."""
    keep = (p >= t) & (nw["j_taken"].to_numpy() == 0)
    if keep.any():
        idx = np.flatnonzero(keep)
        best = pd.Series(p[idx]).groupby(nw["j"].to_numpy()[idx]).transform("max").to_numpy()
        keep[idx[p[idx] < best]] = False
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="prepared data (train/ and test/ records.parquet)")
    ap.add_argument("--stack-dir", required=True, help="run_combine output: stack_*.parquet, stack_rule.json")
    ap.add_argument("--pairs-dir", required=True, help="Stage B model dir: queries.npy, truth.parquet")
    ap.add_argument("--bienc-dir", required=True)
    ap.add_argument("--test-dir", default=None)
    ap.add_argument("--out", default="output_merged")
    ap.add_argument("--per-s1", type=int, default=12, help="new pairs kept per S1 (best bi-encoder cosine)")
    ap.add_argument("--workers", type=int, default=default_workers())
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sd, pdir, bdir = Path(args.stack_dir), Path(args.pairs_dir), Path(args.bienc_dir)
    rule = json.loads((sd / "stack_rule.json").read_text())["rule"]

    # ---- train: held-out S1s
    q = np.load(pdir / "queries.npy")
    truth = pd.read_parquet(pdir / "truth.parquet")
    ti, tj = truth["i"].to_numpy(), truth["j"].to_numpy()
    st = pd.read_parquet(sd / "stack_train.parquet")
    tr = load_split(args.work, "train", columns=REC_COLS)
    n = len(tr.rec)
    base_keep = E.apply_rule(st, rule, "p2")
    base_f = E.per_entity_f05(q, ti, tj, st["i"].to_numpy()[base_keep], st["j"].to_numpy()[base_keep], n)
    bi = pd.read_parquet(bdir / "bienc_train.parquet")
    bi = bi[np.isin(bi["i"].to_numpy(), q)]
    nw = new_pairs(bi, st, n, args.per_s1)
    tk = np.sort(ti.astype(np.int64) * n + tj)
    nw["label"] = E._in_sorted(tk, nw["i"].to_numpy().astype(np.int64) * n + nw["j"].to_numpy()).astype(np.int8)
    ctx = context(nw, st, base_keep)
    nw = pd.concat([nw, ctx], axis=1)
    log(f"train: {len(nw):,} new pairs ({len(nw) / len(q):.1f}/S1), {int(nw['label'].sum()):,} of them true "
        f"(= {nw['label'].sum() / len(ti):.2%} of all true pairs); current OOF macro F0.5 {base_f.mean():.4f}")
    F = features(tr, nw, ctx, args.workers)
    cols = list(F.columns)
    F["i"], F["label"] = nw["i"].to_numpy(), nw["label"].to_numpy()
    oof, iters, imp = M.oof(F, cols, M.MATCH_PARAMS, 5, workers=args.workers, log=log)
    rows = []
    for t in np.round(np.arange(0.3, 0.96, 0.05), 2):
        add = decide_adds(nw, oof, t)
        pi = np.r_[st["i"].to_numpy()[base_keep], nw["i"].to_numpy()[add]]
        pj = np.r_[st["j"].to_numpy()[base_keep], nw["j"].to_numpy()[add]]
        rows.append({"t": t, "added_per_s1": add.sum() / len(q), "added_true": int(nw["label"].to_numpy()[add].sum()),
                     "macro_f05": E.per_entity_f05(q, ti, tj, pi, pj, n).mean()})
    table = pd.DataFrame(rows).sort_values("macro_f05", ascending=False)
    best = table.iloc[0]
    use = best["macro_f05"] > base_f.mean() + 0.0005
    t = float(best["t"]) if use else 2.0
    log("threshold table (top 5):\n" + table.head(5).round(4).to_string(index=False))
    model = M.fit(F, cols, M.MATCH_PARAMS, int(np.mean(iters) * 1.1), args.workers)
    del tr, F

    # ---- test
    stt = pd.read_parquet(sd / "stack_test.parquet")
    te = load_split(args.work, "test", columns=REC_COLS)
    n_te = len(te.rec)
    keep_te = E.apply_rule(stt, rule, "p2")
    bt = pd.read_parquet(bdir / "bienc_test.parquet")
    nt = new_pairs(bt, stt, n_te, args.per_s1)
    cx = context(nt, stt, keep_te)
    nt = pd.concat([nt, cx], axis=1)
    log(f"test: {len(nt):,} new pairs ({len(nt) / len(te.query):.1f}/S1)")
    Ft = features(te, nt, cx, args.workers)
    pt = model.predict(Ft[cols]) if len(nt) else np.zeros(0)
    add_te = decide_adds(nt, pt, t)
    cand_new = pt >= min(t, 0.02)             # the lean model is the final matcher for these pairs
    ids = te.rec["entity_id"].to_numpy(dtype=object)
    qt = te.query

    def write(i, j, path, header):
        lists = (pd.DataFrame({"i": i, "id": ids[j]}).drop_duplicates()
                 .sort_values(["i", "id"]).groupby("i")["id"].agg(",".join))
        col = lists.reindex(qt).fillna("").to_numpy()
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(f"source1_entity_id\t{header}\n")
            fh.writelines(f"{a}\t{b}\n" for a, b in zip(ids[qt], col))

    ci = np.r_[stt["i"].to_numpy(), nt["i"].to_numpy()[cand_new]]
    cj = np.r_[stt["j"].to_numpy(), nt["j"].to_numpy()[cand_new]]
    mi = np.r_[stt["i"].to_numpy()[keep_te], nt["i"].to_numpy()[add_te]]
    mj = np.r_[stt["j"].to_numpy()[keep_te], nt["j"].to_numpy()[add_te]]
    write(ci, cj, out / "candidate_pairs.tsv", "candidate_entity_ids")
    write(mi, mj, out / "matching_results.tsv", "matched_entity_ids")
    print("\n" + imp.head(12).round(0).to_string())
    print("\n===== MERGE SUMMARY (paste this) =====")
    print(json.dumps({"oof_f05_before": round(float(base_f.mean()), 4), "oof_f05_after": round(float(best["macro_f05"]), 4),
                      "merged": bool(use), "t": t if use else None,
                      "train_new_true_pairs": int(nw["label"].sum()), "train_added_true": int(best["added_true"]),
                      "test_added_per_s1": round(float(add_te.sum()) / len(qt), 3),
                      "test_candidates_per_s1": round(len(np.unique(ci * n_te + cj)) / len(qt), 2),
                      "test_matches_per_s1": round(len(mi) / len(qt), 3)}, indent=1))
    if args.test_dir:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "check_submission.py"),
                        "--out", str(out), "--test-dir", args.test_dir], check=False)


if __name__ == "__main__":
    main()
