#!/usr/bin/env python3
"""Retrieval recall on real labels, one question per row (research plan, Exp. 1-2).

  python scripts/recall_check.py --work work --s1-sample 20000 --views full addr skel name

Builds the search vectors once for a random sample of train S1s (full S2/S3
pool), then measures, against the ground truth:
  forward_ivf     current setting: each S1 searches its top-k, approximate IVF
  forward_exact   same, exact search (GPU only)
  + reverse       every pool record proposes its 2 nearest S1s (new pairs, not a feature)
  + name view     name-only search (records with an empty address)
For each: pair recall, complete-S1 recall (all true matches found), candidates
per S1, new true pairs vs the row above, and recall by slice (S2/S3, country,
script, S1 size). CPU works too (no exact rows); GPU is ~10x faster.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from business_entity_resolution.er.dataset import RETRIEVAL_COLUMNS, load_split  # noqa: E402
from business_entity_resolution.er.prepare import default_workers  # noqa: E402
from business_entity_resolution.er.retrieval import (  # noqa: E402
    RetrievalConfig, _in_sorted, build_resources, n_gpus, reverse_best, reverse_pairs, search)

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f}m] {msg}", flush=True)


def keys_of(found: dict, n: int, blockers) -> np.ndarray:
    parts = [found[b][0].astype(np.int64) * n + found[b][1] for b in blockers if b in found]
    return np.unique(np.concatenate(parts)) if parts else np.empty(0, np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(ROOT / "work"))
    ap.add_argument("--s1-sample", type=float, default=20000)
    ap.add_argument("--views", nargs="+", default=["full", "addr", "skel", "name"])
    ap.add_argument("--base-views", nargs="+", default=["full", "addr", "skel"])
    ap.add_argument("--scratch", default="/tmp/ber_scratch_rc")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--fit-sample", type=int, default=200_000)
    args = ap.parse_args()

    split = load_split(args.work, "train", s1_sample=args.s1_sample,
                       columns=RETRIEVAL_COLUMNS + ["non_ascii"])
    n, q = len(split.rec), split.query
    ti, tj = split.truth["i"].to_numpy(), split.truth["j"].to_numpy()
    tk = ti.astype(np.int64) * n + tj
    log(f"train: {n:,} records, {len(q):,} S1 queries, {len(tk):,} true pairs; GPUs: {n_gpus()}")
    cfg = RetrievalConfig(views=tuple(args.views), workers=args.workers, scratch=args.scratch,
                          fit_sample=args.fit_sample, exact_gpu=False)
    res = build_resources(split, cfg, log)

    rows = {}
    nt = np.bincount(np.searchsorted(q, ti), minlength=len(q))
    src_t = split.rec["source"].to_numpy()[tj]
    cc = split.rec["country_code"].to_numpy()[ti]
    xs = split.rec["non_ascii"].to_numpy()[tj] == 1
    size = np.select([nt[np.searchsorted(q, ti)] == 1, nt[np.searchsorted(q, ti)] <= 3], ["1", "2-3"], "4+")

    def measure(name, keys, prev):
        hit = _in_sorted(keys, tk)
        miss_s1 = np.unique(ti[~hit])
        r = {"pair_recall": round(float(hit.mean()), 4),
             "complete_s1_recall": round(1 - len(miss_s1) / len(q), 4),
             "cands_per_s1": round(len(keys) / len(q), 1),
             "new_true_pairs_vs_prev": int((hit & ~_in_sorted(prev, tk)).sum()) if prev is not None else None,
             "recall_S2": round(float(hit[src_t == 2].mean()), 4),
             "recall_S3": round(float(hit[src_t == 3].mean()), 4),
             "recall_cross_script": round(float(hit[xs].mean()), 4) if xs.any() else None}
        for c in np.unique(cc):
            r[f"recall_{split.countries[c]}"] = round(float(hit[cc == c].mean()), 4)
        for s in ("1", "2-3", "4+"):
            r[f"recall_s1size_{s}"] = round(float(hit[size == s].mean()), 4)
        rows[name] = r
        log(f"{name}: pair recall {r['pair_recall']}, complete-S1 {r['complete_s1_recall']}, "
            f"{r['cands_per_s1']}/S1, new true pairs {r['new_true_pairs_vs_prev']}")
        return keys

    base = [f"ann_{v}" for v in args.base_views] + ["rare", "exact"]
    extra = [f"ann_{v}" for v in args.views if v not in args.base_views]
    f_ivf = search(split, res, cfg, q, log)
    prev = measure("forward_ivf (current)", keys_of(f_ivf, n, base), None)
    cur = f_ivf
    if n_gpus() > 0:
        cfg_x = dataclasses.replace(cfg, exact_gpu=True)
        cur = search(split, res, cfg_x, q, log)
        prev = measure("forward_exact", keys_of(cur, n, base), prev)
    else:
        cfg_x = cfg
        log("no GPU: exact search skipped")
    pool = np.flatnonzero((res.src != 1) & np.isin(res.country, np.unique(res.country[q])))
    rev = reverse_best(split, res, cfg_x, pool)
    cur["rev"] = reverse_pairs(rev, q)
    prev = measure("+ reverse", keys_of(cur, n, base + ["rev"]), prev)
    if extra:
        prev = measure(f"+ {' '.join(v[4:] for v in extra)} view", keys_of(cur, n, base + ["rev"] + extra), prev)
    # what each blocker contributes on its own, in the final union
    allb = base + ["rev"] + extra
    per_b = {}
    for b in allb:
        kb = keys_of(cur, n, [b])
        others = keys_of(cur, n, [x for x in allb if x != b])
        hb, ho = _in_sorted(kb, tk), _in_sorted(others, tk)
        per_b[b] = {"recall_alone": round(float(hb.mean()), 4), "only_this_blocker": int((hb & ~ho).sum()),
                    "cands_per_s1": round(len(kb) / len(q), 1)}
    print("\n===== RECALL CHECK (paste this) =====")
    print(pd.DataFrame(rows).to_string())
    print("\nper blocker (in the final union):")
    print(pd.DataFrame(per_b).T.to_string())
    print(json.dumps({"s1_sample": len(q), "true_pairs": int(len(tk)), "minutes": round((time.time() - T0) / 60, 1)}))


if __name__ == "__main__":
    main()
