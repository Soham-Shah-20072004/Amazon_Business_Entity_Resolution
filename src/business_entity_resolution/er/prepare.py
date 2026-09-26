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
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import text as T

SRC_COLS = ["entity_id", "business_name", "business_address", "country"]


def read_source(path: Path, chunk_rows: int = 400_000, max_rows: int | None = None):
    """Yield DataFrame chunks. C engine + QUOTE_NONE so a stray '"' in an
    address can't swallow rows; chunked so memory stays flat on small boxes."""
    reader = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                         quoting=csv.QUOTE_NONE, engine="c", chunksize=chunk_rows, nrows=max_rows)
    for k, df in enumerate(reader):
        df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
        if k == 0:
            missing = [c for c in SRC_COLS if c not in df.columns]
            if missing:
                raise ValueError(f"{path}: missing columns {missing}")
        yield df[SRC_COLS]


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


def _parallel_normalise(df: pd.DataFrame, pool, workers: int) -> pd.DataFrame:
    step = max(1, -(-len(df) // max(1, workers)))
    parts = [df.iloc[s:s + step] for s in range(0, len(df), step)]
    out = pool.map(normalise_frame, parts) if pool else [normalise_frame(p) for p in parts]
    return pd.concat(out, ignore_index=True)


def prepare_split(data_root: Path, split: str, out: Path, workers: int, log=print,
                  max_rows: int | None = None) -> dict:
    """Stream each source file in chunks -> normalise in parallel -> append to one
    parquet file. Peak memory is a few chunks, not the whole split."""
    t0 = time.time()
    src_dir = Path(data_root) / split
    dst = Path(out) / split
    dst.mkdir(parents=True, exist_ok=True)
    manifest = {"split": split, "tables": {}, "max_rows": max_rows}
    writer = None
    pool = mp.get_context("fork").Pool(workers) if workers > 1 else None
    try:
        for s in (1, 2, 3):
            path = src_dir / f"{split}_source{s}.tsv"
            t = time.time()
            n_rows = 0
            for chunk in read_source(path, max_rows=max_rows):
                norm = _parallel_normalise(chunk, pool, workers)
                norm.insert(1, "source", np.int8(s))
                table = pa.Table.from_pandas(norm, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(dst / "records.parquet", table.schema, compression="zstd")
                writer.write_table(table.cast(writer.schema))
                n_rows += len(chunk)
            n_lines = count_lines(path) - 1
            if max_rows is None and n_lines != n_rows:
                log(f"  WARNING {path.name}: {n_rows:,} rows parsed but {n_lines:,} data lines in file")
            manifest["tables"][f"source{s}"] = {"rows": n_rows, "lines": n_lines}
            log(f"  {path.name}: {n_rows:,} rows normalised in {time.time() - t:.0f}s")
    finally:
        if writer is not None:
            writer.close()
        if pool is not None:
            pool.close()
            pool.join()

    ids = pq.read_table(dst / "records.parquet", columns=["entity_id"]).column(0)
    n_unique = pc.count_distinct(ids).as_py()
    if n_unique != len(ids):
        raise ValueError(f"{split}: {len(ids) - n_unique} duplicate entity ids")
    meta = pq.read_table(dst / "records.parquet", columns=["source", "country_norm"])
    manifest["countries"] = meta.group_by(["source", "country_norm"]).aggregate(
        [([], "count_all")]).to_pylist()
    del ids, meta

    gt_path = src_dir / f"{split}_ground_truth.tsv"
    if gt_path.exists():
        gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False,
                         quoting=csv.QUOTE_NONE, engine="c")
        pairs = gt.assign(cand=gt["matched_entity_ids"].str.split(",")) \
            .explode("cand")[["source1_entity_id", "cand"]] \
            .rename(columns={"source1_entity_id": "s1"})
        pairs["cand"] = pairs["cand"].str.strip()
        pairs = pairs[pairs["cand"].notna() & (pairs["cand"] != "")].drop_duplicates()
        pairs.to_parquet(dst / "truth.parquet", compression="zstd", index=False)
        manifest["truth"] = {"s1_rows": len(gt), "pairs": len(pairs),
                             "cands_with_2plus_owners": int(pairs["cand"].duplicated().sum())}
        log(f"  ground truth: {len(gt):,} S1 rows, {len(pairs):,} positive pairs, "
            f"{manifest['truth']['cands_with_2plus_owners']:,} candidate ids claimed by 2+ S1")
    manifest["seconds"] = round(time.time() - t0, 1)
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)
