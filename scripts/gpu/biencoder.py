#!/usr/bin/env python3
"""Learned retrieval (bi-encoder) + sibling propagation, GPU.

A multilingual MiniLM is fine-tuned so that a record and its true S1 get close
vectors (contrastive, in-batch negatives: each batch holds distinct S1s, so an
S1's other true matches never act as negatives). Then every record is encoded
once and candidates are found three ways, per country and target source:
  fwd   each S1's top-k nearest records                      (S1 -> record)
  rev   each record's 2 nearest S1s                           (record -> S1)
  hop   near-copies of an S1's strongest candidate            (S1 -> A -> B, sibling propagation)

  python scripts/gpu/biencoder.py --work work --holdout-queries models/m1/queries.npy \
      --baseline-pairs models/m1/train_pairs.parquet --out bienc_out

The S1s in --holdout-queries (the Stage B training sample) are excluded from
fine-tuning, so the recall report on them is honest, and their candidates can
train the merge model (scripts/merge_bienc.py). Outputs:
  bienc_train.parquet / bienc_test.parquet   i, j, cos, fwd_rank, rev_rank, hop, hop_cos
  RECALL block: recall of each step on the held-out S1s, and how many true pairs
  it finds that the Stage B candidates (--baseline-pairs) missed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from business_entity_resolution.er.dataset import load_split  # noqa: E402

T0 = time.time()
COLS = ["entity_id", "source", "country_norm", "name_norm", "addr_norm"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f}m] {msg}", flush=True)


def texts_of(rec: pd.DataFrame) -> np.ndarray:
    name = rec["name_norm"].to_numpy(dtype=object)
    addr = rec["addr_norm"].to_numpy(dtype=object)
    return np.array([f"{n} | {a}" if a else n for n, a in zip(name, addr)], dtype=object)


class Encoder:
    def __init__(self, model_name: str, max_len: int):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.max_len = max_len
        self.amp = self.device.type == "cuda"
        n_gpu = torch.cuda.device_count() if self.amp else 0

        class Pool(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, input_ids, attention_mask):
                h = self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                mask = attention_mask.unsqueeze(-1).to(h.dtype)
                v = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
                return torch.nn.functional.normalize(v.float(), dim=-1)

        self.pool = Pool(self.model)
        self.net = torch.nn.DataParallel(self.pool) if n_gpu > 1 else self.pool

    def _enc(self, texts):
        e = self.tok(list(texts), truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
        return e["input_ids"].to(self.device, non_blocking=True), e["attention_mask"].to(self.device, non_blocking=True)

    def fit(self, a: np.ndarray, b: np.ndarray, bs: int, lr: float, scale: float = 20.0, seed: int = 0):
        torch = self.torch
        from transformers import get_linear_schedule_with_warmup
        self.net.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        steps = math.ceil(len(a) / bs)
        sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        order = np.random.default_rng(seed).permutation(len(a))
        labels = None
        run = 0.0
        for k in range(steps):
            idx = order[k * bs:(k + 1) * bs]
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                ea = self.net(*self._enc(a[idx]))
                eb = self.net(*self._enc(b[idx]))
            logits = scale * ea @ eb.T
            if labels is None or len(labels) != len(idx):
                labels = torch.arange(len(idx), device=logits.device)
            loss = (torch.nn.functional.cross_entropy(logits, labels)
                    + torch.nn.functional.cross_entropy(logits.T, labels)) / 2
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run = 0.98 * run + 0.02 * loss.item() if k else loss.item()
            if k % 200 == 0:
                log(f"    step {k}/{steps} loss {run:.4f}")

    def embed(self, texts: np.ndarray, out: np.ndarray, bs: int = 1024) -> None:
        """Writes unit vectors (float16) into `out` row by row; length-sorted batches."""
        torch = self.torch
        self.net.eval()
        lens = np.fromiter((len(t) for t in texts), np.int32, len(texts))
        order = np.argsort(lens)
        with torch.no_grad():
            for s in range(0, len(order), bs):
                idx = order[s:s + bs]
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                    v = self.net(*self._enc(texts[idx]))
                out[idx] = v.cpu().numpy().astype(np.float16)
                if (s // bs) % 2000 == 0:
                    log(f"    encoded {s + len(idx):,}/{len(texts):,}")


def knn(E: np.ndarray, base_rows: np.ndarray, query_rows: np.ndarray, k: int, chunk: int = 200_000):
    """Exact inner-product top-k of E[query_rows] among E[base_rows] (GPU if available)."""
    import faiss
    d = E.shape[1]
    k = min(k, len(base_rows))
    index = faiss.IndexFlatIP(d)
    if hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True
        index = faiss.index_cpu_to_all_gpus(index, co)
    for s in range(0, len(base_rows), 1_000_000):
        index.add(np.ascontiguousarray(E[base_rows[s:s + 1_000_000]], dtype=np.float32))
    D = np.empty((len(query_rows), k), np.float32)
    I = np.empty((len(query_rows), k), np.int64)
    for s in range(0, len(query_rows), chunk):
        D[s:s + chunk], I[s:s + chunk] = index.search(
            np.ascontiguousarray(E[query_rows[s:s + chunk]], dtype=np.float32), k)
    del index
    return D, np.where(I >= 0, base_rows[np.maximum(I, 0)], -1)


def rowdot(E: np.ndarray, a: np.ndarray, b: np.ndarray, chunk: int = 2_000_000) -> np.ndarray:
    out = np.empty(len(a), np.float32)
    for s in range(0, len(a), chunk):
        out[s:s + chunk] = np.einsum("ij,ij->i", E[a[s:s + chunk]].astype(np.float32),
                                     E[b[s:s + chunk]].astype(np.float32))
    return out


def retrieve(split, E: np.ndarray, queries: np.ndarray, k: int, rev_k: int, hop_seeds: int, hop_k: int) -> pd.DataFrame:
    src = split.rec["source"].to_numpy()
    cc = split.rec["country_code"].to_numpy()
    qset = np.zeros(len(src), bool)
    qset[queries] = True
    parts = []
    for c in np.unique(cc[queries]):
        q = queries[cc[queries] == c]
        s1c = np.flatnonzero((src == 1) & (cc == c))
        for tgt in (2, 3):
            pool = np.flatnonzero((src == tgt) & (cc == c))
            if len(pool) == 0:
                continue
            t = time.time()
            D, I = knn(E, pool, q, k)
            parts.append(pd.DataFrame({"i": np.repeat(q, I.shape[1]), "j": I.ravel(), "cos": D.ravel(),
                                       "fwd_rank": np.tile(np.arange(1, I.shape[1] + 1), len(q))}))
            # sibling propagation: near-copies of the S1's best candidates, among ALL pool records
            seeds = I[:, :hop_seeds]
            seed_i = np.repeat(q, seeds.shape[1])
            seed_j = seeds.ravel()
            ok = seed_j >= 0
            seed_i, seed_j = seed_i[ok], seed_j[ok]
            allpool = np.flatnonzero((src != 1) & (cc == c))
            useeds, inv = np.unique(seed_j, return_inverse=True)
            Dh, Ih = knn(E, allpool, useeds, hop_k + 1)
            hj = Ih[inv].ravel()
            hc = Dh[inv].ravel()
            hi = np.repeat(seed_i, Ih.shape[1])
            keep = (hj >= 0) & (hj != np.repeat(seed_j, Ih.shape[1]))
            parts.append(pd.DataFrame({"i": hi[keep], "j": hj[keep], "hop": np.int8(1), "hop_cos": hc[keep]}))
            log(f"    country {split.countries[c]} -> S{tgt}: fwd {len(q):,} x {len(pool):,}, "
                f"hop from {len(useeds):,} seeds in {time.time() - t:.0f}s")
        # reverse: every pool record of the country proposes its nearest S1s (among ALL S1)
        t = time.time()
        pool = np.flatnonzero((src != 1) & (cc == c))
        D, I = knn(E, s1c, pool, rev_k)
        ri = I.ravel()
        rj = np.repeat(pool, I.shape[1])
        rr = np.tile(np.arange(1, I.shape[1] + 1), len(pool))
        keep = (ri >= 0) & qset[np.maximum(ri, 0)]
        parts.append(pd.DataFrame({"i": ri[keep], "j": rj[keep], "cos": D.ravel()[keep], "rev_rank": rr[keep]}))
        log(f"    country {split.countries[c]}: reverse {len(pool):,} records x {len(s1c):,} S1 in {time.time() - t:.0f}s")
    df = pd.concat(parts, ignore_index=True)
    df = df[df["j"] >= 0]
    agg = df.groupby(["i", "j"], sort=False).agg(cos=("cos", "max"), fwd_rank=("fwd_rank", "min"),
                                                 rev_rank=("rev_rank", "min"), hop=("hop", "max"),
                                                 hop_cos=("hop_cos", "max")).reset_index()
    agg["fwd_rank"] = agg["fwd_rank"].fillna(99).astype(np.int16)
    agg["rev_rank"] = agg["rev_rank"].fillna(9).astype(np.int8)
    agg["hop"] = agg["hop"].fillna(0).astype(np.int8)
    agg["hop_cos"] = agg["hop_cos"].fillna(0).astype(np.float32)
    miss = agg["cos"].isna().to_numpy()           # pairs found only by propagation
    cos = agg["cos"].to_numpy(np.float32, na_value=0)
    cos[miss] = rowdot(E, agg["i"].to_numpy()[miss], agg["j"].to_numpy()[miss])
    agg["cos"] = cos
    agg["i"] = agg["i"].astype(np.int64)
    agg["j"] = agg["j"].astype(np.int64)
    return agg


def recall_report(split, pairs: pd.DataFrame, baseline: pd.DataFrame | None, queries: np.ndarray) -> dict:
    n = len(split.rec)
    t = split.truth[np.isin(split.truth["i"].to_numpy(), queries)]
    tk = np.sort(t["i"].to_numpy() * n + t["j"].to_numpy())

    def rec(mask):
        k = np.unique(pairs["i"].to_numpy()[mask] * n + pairs["j"].to_numpy()[mask])
        return k

    def hit(keys):
        pos = np.minimum(np.searchsorted(keys, tk), max(len(keys) - 1, 0))
        return (keys[pos] == tk) if len(keys) else np.zeros(len(tk), bool)

    f = pairs["fwd_rank"].to_numpy() < 99
    r = pairs["rev_rank"].to_numpy() < 9
    h = pairs["hop"].to_numpy() == 1
    out = {"held_out_s1": int(len(queries)), "true_pairs": int(len(tk))}
    for name, m in {"fwd": f, "fwd+rev": f | r, "fwd+rev+hop": f | r | h}.items():
        hk = hit(rec(m))
        miss_s1 = np.unique(t["i"].to_numpy()[~hk])
        out[f"recall_{name}"] = round(float(hk.mean()), 4)
        out[f"complete_s1_{name}"] = round(1 - len(miss_s1) / len(queries), 4)
        out[f"pairs_per_s1_{name}"] = round(int(m.sum()) / len(queries), 1)
    if baseline is not None:
        bk = np.sort(np.unique(baseline["i"].to_numpy() * n + baseline["j"].to_numpy()))
        hb = hit(bk)
        ha = hit(rec(f | r | h))
        out["stage_b_candidate_recall"] = round(float(hb.mean()), 4)
        out["true_pairs_stage_b_missed"] = int((~hb).sum())
        out["of_those_found_by_bienc"] = int((~hb & ha).sum())
        out["union_recall"] = round(float((hb | ha).mean()), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--holdout-queries", default=None, help="queries.npy of the Stage B model (train row ids)")
    ap.add_argument("--holdout-n", type=int, default=20000, help="random held-out S1s when no queries.npy is given")
    ap.add_argument("--baseline-pairs", default=None, help="Stage B train_pairs.parquet, for the recall comparison")
    ap.add_argument("--out", default="bienc_out")
    ap.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--max-pairs", type=int, default=1_200_000, help="training pairs (one per S1)")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--k", type=int, default=10, help="forward neighbours per S1 per source")
    ap.add_argument("--rev-k", type=int, default=2)
    ap.add_argument("--hop-seeds", type=int, default=2)
    ap.add_argument("--hop-k", type=int, default=3)
    ap.add_argument("--scratch", default="/tmp/bienc")
    ap.add_argument("--smoke", action="store_true", help="tiny run: few pairs, test limited to 20k S1")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    # ---- train split: fine-tune on S1s outside the held-out sample
    tr = load_split(args.work, "train", columns=COLS)
    rng = np.random.default_rng(0)
    if args.holdout_queries:
        hold = np.load(args.holdout_queries)
    else:
        hold = np.sort(rng.choice(tr.query, min(args.holdout_n, len(tr.query)), replace=False))
    t = tr.truth[~np.isin(tr.truth["i"].to_numpy(), hold)]
    t = t.iloc[rng.permutation(len(t))].drop_duplicates("i")          # one positive per S1
    t = t.head(2000 if args.smoke else args.max_pairs)
    txt = texts_of(tr.rec)
    log(f"train: {len(tr.rec):,} records; fine-tuning on {len(t):,} (S1, match) pairs; held out {len(hold):,} S1")
    enc = Encoder(args.model, args.max_len)
    log(f"device {enc.device}; e.g. '{txt[t['i'].iloc[0]]}'  <->  '{txt[t['j'].iloc[0]]}'")
    enc.fit(txt[t["i"].to_numpy()], txt[t["j"].to_numpy()], args.bs, args.lr)
    enc.model.save_pretrained(out / "model")
    enc.tok.save_pretrained(out / "model")

    # ---- train retrieval for the held-out S1s (recall report + training data for the merge model)
    E = np.lib.format.open_memmap(scratch / "E_train.npy", mode="w+", dtype=np.float16,
                                  shape=(len(txt), enc.model.config.hidden_size))
    log("encoding train records")
    enc.embed(txt, E)
    pairs = retrieve(tr, E, np.sort(hold), args.k, args.rev_k, args.hop_seeds, args.hop_k)
    pairs.to_parquet(out / "bienc_train.parquet", index=False)
    base = pd.read_parquet(args.baseline_pairs, columns=["i", "j"]) if args.baseline_pairs else None
    rep = recall_report(tr, pairs, base, np.sort(hold))
    del E
    (scratch / "E_train.npy").unlink(missing_ok=True)
    del tr, txt

    # ---- test split
    te = load_split(args.work, "test", columns=COLS)
    txt = texts_of(te.rec)
    q = np.flatnonzero(te.rec["source"].to_numpy() == 1)
    if args.smoke:
        q = np.sort(rng.choice(q, min(20_000, len(q)), replace=False))
    E = np.lib.format.open_memmap(scratch / "E_test.npy", mode="w+", dtype=np.float16,
                                  shape=(len(txt), enc.model.config.hidden_size))
    log("encoding test records")
    enc.embed(txt, E)
    tp = retrieve(te, E, q, args.k, args.rev_k, args.hop_seeds, args.hop_k)
    tp.to_parquet(out / "bienc_test.parquet", index=False)
    del E
    (scratch / "E_test.npy").unlink(missing_ok=True)
    rep.update({"test_s1": int(len(q)), "test_pairs_per_s1": round(len(tp) / len(q), 1),
                "minutes": round((time.time() - T0) / 60, 1)})
    (out / "bienc_summary.json").write_text(json.dumps(rep, indent=2))
    print("\n===== BI-ENCODER RECALL (paste this) =====")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
