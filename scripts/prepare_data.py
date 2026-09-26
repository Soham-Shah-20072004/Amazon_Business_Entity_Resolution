#!/usr/bin/env python3
"""Step 1 (run once): TSVs -> normalised parquet cache under work/.

  python scripts/prepare_data.py --data-root dataset --out work
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from business_entity_resolution.er.prepare import default_workers, prepare_split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(ROOT / "dataset"))
    ap.add_argument("--out", default=str(ROOT / "work"))
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--workers", type=int, default=default_workers())
    args = ap.parse_args()
    for split in args.splits:
        t = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] preparing {split} with {args.workers} workers", flush=True)
        m = prepare_split(Path(args.data_root), split, Path(args.out), args.workers,
                          log=lambda s: print(s, flush=True))
        print(f"[{time.strftime('%H:%M:%S')}] {split} done in {time.time() - t:.0f}s: "
              f"{ {k: v['rows'] for k, v in m['tables'].items()} }", flush=True)


if __name__ == "__main__":
    main()
