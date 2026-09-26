"""One-time data preparation: fast TSV load -> parallel normalisation -> parquet.

Output (per split, under --out, default work/):
  <split>/records.parquet   one row per record (S1/S2/S3) with all text views
  train/truth.parquet       positive pairs (s1, cand) from the ground truth
  <split>/manifest.json     row counts, timings

Everything downstream reads these parquet files, so the slow text work
(unidecode on ~12M names + addresses per split) runs once.
"""

from __future__ import annotations

import csv
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from . import text as T

SRC_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_source(path: Path) -> pd.DataFrame:
    """C-engine TSV read; QUOTE_NONE so a stray '"' in an address can't swallow rows."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                     quoting=csv.QUOTE_NONE, engine="c")
    df.columns = [c.replace("﻿", "").strip() for c in df.columns]
    missing = [c for c in SRC_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    return df[SRC_COLS]


def count_lines(path: Path) -> int:
    with open(path, "rb") as fh:
        return sum(buf.count(b"\n") for buf in iter(lambda: fh.read(1 << 24), b""))


def normalise_frame(df: pd.DataFrame) -> pd.DataFrame:
    name_norm = [T.norm(x) for x in df["business_name"]]
    addr_norm = [T.norm(x) for x in df["business_address"]]
    name_ascii = [T.to_ascii(x) for x in name_norm]
    addr_ascii = [T.to_ascii(x) for x in addr_norm]
    name_canon = [T.canon_name(x) for x in name_ascii]
    addr_canon = [T.canon_addr(x) for x in addr_ascii]
    nums = [T.numbers(x) for x in addr_ascii]
    raw = (df["business_name"] + " " + df["business_address"]).tolist()
    return pd.DataFrame({
        "entity_id": df["entity_id"].to_numpy(),
        "country": df["country"].to_numpy(),
        "country_norm": [T.norm(x) for x in df["country"]],
        "name_norm": name_norm,
        "addr_norm": addr_norm,
        "name_ascii": name_ascii,
        "addr_ascii": addr_ascii,
        "name_canon": name_canon,
        "addr_canon": addr_canon,
        "name_core": [T.core_name(x) for x in name_canon],
        "name_acr": [T.acronym(x) for x in name_canon],
        "name_skel": [T.skeleton(x) for x in name_canon],
        "addr_skel": [T.skeleton(x) for x in addr_canon],
        "addr_nums": [" ".join(n) for n in nums],
        "addr_post": [" ".join(T.postcodes(n, a)) for n, a in zip(nums, addr_ascii)],
        "non_ascii": np.fromiter((any(ord(c) > 127 for c in s) for s in raw), np.int8, len(raw)),
    })


def _parallel_normalise(df: pd.DataFrame, workers: int, chunk: int = 100_000) -> pd.DataFrame:
    parts = [df.iloc[s:s + chunk] for s in range(0, len(df), chunk)]
    if workers <= 1 or len(parts) == 1:
        return pd.concat([normalise_frame(p) for p in parts], ignore_index=True)
    with Pool(workers) as pool:
        out = pool.map(normalise_frame, parts, chunksize=1)
    return pd.concat(out, ignore_index=True)


def prepare_split(data_root: Path, split: str, out: Path, workers: int, log=print) -> dict:
    t0 = time.time()
    src_dir = Path(data_root) / split
    dst = Path(out) / split
    dst.mkdir(parents=True, exist_ok=True)
    manifest = {"split": split, "tables": {}}
    frames = []
    for s in (1, 2, 3):
        path = src_dir / f"{split}_source{s}.tsv"
        t = time.time()
        raw = read_source(path)
        n_lines = count_lines(path) - 1
        if n_lines != len(raw):
            log(f"  WARNING {path.name}: {len(raw):,} rows parsed but {n_lines:,} data lines in file")
        norm = _parallel_normalise(raw, workers)
        norm.insert(1, "source", np.int8(s))
        frames.append(norm)
        manifest["tables"][f"source{s}"] = {"rows": len(raw), "lines": n_lines}
        log(f"  {path.name}: {len(raw):,} rows normalised in {time.time() - t:.0f}s")
    rec = pd.concat(frames, ignore_index=True)
    dup = rec["entity_id"].duplicated()
    if dup.any():
        raise ValueError(f"{split}: {int(dup.sum())} duplicate entity ids, e.g. "
                         f"{rec.loc[dup, 'entity_id'].head(3).tolist()}")
    rec.to_parquet(dst / "records.parquet", compression="zstd", index=False)
    manifest["countries"] = rec.groupby(["source", "country_norm"]).size() \
        .rename("n").reset_index().to_dict("records")

    gt_path = src_dir / f"{split}_ground_truth.tsv"
    if gt_path.exists():
        gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False,
                         quoting=csv.QUOTE_NONE, engine="c")
        pairs = gt.assign(cand=gt["matched_entity_ids"].str.split(",")) \
            .explode("cand")[["source1_entity_id", "cand"]] \
            .rename(columns={"source1_entity_id": "s1"})
        pairs["cand"] = pairs["cand"].str.strip()
        pairs = pairs[pairs["cand"] != ""].drop_duplicates()
        pairs.to_parquet(dst / "truth.parquet", compression="zstd", index=False)
        pd.DataFrame({"s1": gt["source1_entity_id"]}).to_parquet(dst / "gt_s1.parquet", index=False)
        manifest["truth"] = {"s1_rows": len(gt), "pairs": len(pairs),
                             "cands_with_2plus_owners": int(pairs["cand"].duplicated().sum())}
        log(f"  ground truth: {len(gt):,} S1 rows, {len(pairs):,} positive pairs, "
            f"{manifest['truth']['cands_with_2plus_owners']:,} candidate ids claimed by 2+ S1")
    manifest["seconds"] = round(time.time() - t0, 1)
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)
