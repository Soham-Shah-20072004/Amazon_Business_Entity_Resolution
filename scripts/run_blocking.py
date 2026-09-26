#!/usr/bin/env python3
"""Step 2: candidate generation + recall report.

  # quick check on 50k random train S1s against the FULL S2/S3 pool
  python scripts/run_blocking.py --split train --s1-sample 50000 --tag b1

  # full test set (writes work/cands/test_<tag>.parquet)
  python scripts/run_blocking.py --split test --tag b1

Paste the printed REPORT block back into the chat.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from business_entity_resolution.er.dataset import RETRIEVAL_COLUMNS, load_split  # noqa: E402
from business_entity_resolution.er.prepare import default_workers  # noqa: E402
from business_entity_resolution.er.retrieval import (  # noqa: E402
    RetrievalConfig, blocking_report, generate_candidates)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(ROOT / "work"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--s1-sample", type=float, default=None,
                    help="random S1 queries: fraction (<1) or count; pool stays complete")
    ap.add_argument("--tag", default="b1")
    ap.add_argument("--views", nargs="+", default=list(RetrievalConfig.views))
    for f in fields(RetrievalConfig):
        if f.init and f.name not in ("views",):
            ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    args = ap.parse_args()
    if args.workers == RetrievalConfig.workers:
        args.workers = default_workers()
    cfg = RetrievalConfig(**{f.name: getattr(args, f.name) for f in fields(RetrievalConfig) if f.init
                             and f.name != "views"}, views=tuple(args.views))

    t = time.time()
    split = load_split(args.work, args.split, s1_sample=args.s1_sample, columns=RETRIEVAL_COLUMNS)
    print(f"[{time.strftime('%H:%M:%S')}] loaded {args.split}: {len(split.rec):,} records, "
          f"{len(split.query):,} S1 queries in {time.time() - t:.0f}s", flush=True)
    cand = generate_candidates(split, cfg, log=lambda s: print(s, flush=True))
    rep = blocking_report(split, cand, cfg)

    out = Path(args.work) / "cands"
    out.mkdir(parents=True, exist_ok=True)
    cand.to_parquet(out / f"{args.split}_{args.tag}.parquet", index=False)
    meta = {"split": args.split, "s1_sample": args.s1_sample, "config": asdict(cfg), "report": rep,
            "seconds": round(time.time() - t, 1)}
    (out / f"{args.split}_{args.tag}.json").write_text(json.dumps(meta, indent=2, default=str))
    print("\n===== REPORT (paste this) =====")
    print(json.dumps({"tag": args.tag, "split": args.split, "s1_sample": args.s1_sample,
                      "k_ann": cfg.k_ann, "k_rare": cfg.k_rare, "rare_max_df": cfg.rare_max_df,
                      "nprobe": cfg.nprobe, "minutes": round((time.time() - t) / 60, 1),
                      **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in rep.items()}},
                     indent=1))


if __name__ == "__main__":
    main()
