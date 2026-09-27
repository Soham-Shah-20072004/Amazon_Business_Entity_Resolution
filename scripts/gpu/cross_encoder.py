#!/usr/bin/env python3
"""Cross-encoder reranker (GPU): fine-tune a multilingual transformer to read a
(S1, candidate) pair jointly and output P(match); its score becomes an extra
feature for the LightGBM matcher (scripts/run_combine.py).

  python scripts/gpu/cross_encoder.py \
      --pairs-dir  <Stage B models/<tag> dir: train_pairs.parquet, test_pairs/> \
      --train-records <work/train/records.parquet> --test-records <work/test/records.parquet> \
      --out ce_out

Input text per record: "<name> | <address>" in the original script (Hindi,
Gujarati, French accents kept - the model's SentencePiece vocabulary covers
them). A pair is fed as  [CLS] record A [SEP] record B [SEP]  so every token of
one record can attend to every token of the other.

Out-of-fold scheme (needed for stacking): train pairs are split in 2 halves by
S1; a model trained on one half scores the other half, and the test pairs get
the average of both models. Outputs ce_train.parquet / ce_test.parquet with
columns i, j, ce (probability).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

T0 = time.time()


def focus_set(df: pd.DataFrame, min_p: float, top_k: int) -> np.ndarray:
    """Same rule as er.evaluate.focus_set (kept local so this script runs standalone)."""
    rank = df.groupby("i", sort=False)["p"].rank(ascending=False, method="first").to_numpy()
    return (df["p"].to_numpy() >= min_p) & (rank <= top_k)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f}m] {msg}", flush=True)


def fold_of(i: np.ndarray, n_folds: int, seed: int = 42) -> np.ndarray:
    return ((i.astype(np.int64) * 2654435761 + seed) % 2_147_483_647 % n_folds).astype(np.int8)


def record_text(records: str, rows: np.ndarray) -> np.ndarray:
    """'<name> | <address>' (unicode-normalised, original script) for the given rows."""
    t = pq.read_table(records, columns=["name_norm", "addr_norm"])
    name = t.column(0).take(rows).to_pylist()
    addr = t.column(1).take(rows).to_pylist()
    return np.array([f"{n} | {a}" if a else n for n, a in zip(name, addr)], dtype=object)


def pair_texts(records: str, i: np.ndarray, j: np.ndarray):
    rows = np.unique(np.concatenate([i, j]))
    txt = record_text(records, rows)
    return txt[np.searchsorted(rows, i)], txt[np.searchsorted(rows, j)]


class Scorer:
    def __init__(self, model_name: str, max_len: int, device):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)
        self.device, self.max_len = device, max_len
        self.n_gpu = torch.cuda.device_count() if device.type == "cuda" else 0
        self.amp = device.type == "cuda"

    def _net(self):
        return self.torch.nn.DataParallel(self.model) if self.n_gpu > 1 else self.model

    def encode(self, a, b):
        enc = self.tok(list(a), list(b), truncation="longest_first", max_length=self.max_len,
                       padding=True, return_tensors="pt")
        return {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}

    def train(self, a, b, y, epochs: int, bs: int, lr: float, seed: int = 0):
        torch = self.torch
        from transformers import get_linear_schedule_with_warmup
        net = self._net()
        net.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        steps = epochs * math.ceil(len(y) / bs)
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        except (AttributeError, TypeError):           # older torch
            scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(seed)
        lens = np.fromiter((len(x) + len(z) for x, z in zip(a, b)), np.int32, len(a))
        step = 0
        for ep in range(epochs):
            # shuffle, then sort inside mega-batches by length -> little padding, still random
            order = rng.permutation(len(y))
            mega = bs * 50
            order = np.concatenate([o[np.argsort(lens[o])] for o in np.array_split(order, max(1, len(order) // mega))])
            batches = [order[s:s + bs] for s in range(0, len(order), bs)]
            rng.shuffle(batches)
            run = 0.0
            for k, idx in enumerate(batches):
                enc = self.encode(a[idx], b[idx])
                target = torch.tensor(y[idx], dtype=torch.float32, device=self.device)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                    logits = net(**enc).logits.squeeze(-1)
                loss = loss_fn(logits.float(), target)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                step += 1
                run = 0.98 * run + 0.02 * loss.item() if k else loss.item()
                if k % 200 == 0:
                    log(f"    epoch {ep + 1} step {k}/{len(batches)} loss {run:.4f}")

    def predict(self, a, b, bs: int) -> np.ndarray:
        torch = self.torch
        net = self._net()
        net.eval()
        lens = np.fromiter((len(x) + len(z) for x, z in zip(a, b)), np.int32, len(a))
        order = np.argsort(lens)
        out = np.empty(len(a), np.float32)
        with torch.no_grad():
            for s in range(0, len(order), bs):
                idx = order[s:s + bs]
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
                    logits = net(**self.encode(a[idx], b[idx])).logits.squeeze(-1)
                out[idx] = torch.sigmoid(logits.float()).cpu().numpy()
                if (s // bs) % 500 == 0:
                    log(f"    scored {s + len(idx):,}/{len(a):,}")
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--train-records", required=True)
    ap.add_argument("--test-records", required=True)
    ap.add_argument("--out", default="ce_out")
    ap.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--infer-bs", type=int, default=512)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--min-p", type=float, default=0.01, help="focus set: first-stage p threshold")
    ap.add_argument("--top-k", type=int, default=8, help="focus set: max pairs per S1")
    ap.add_argument("--limit-train", type=int, default=None, help="smoke test: use N train pairs")
    ap.add_argument("--limit-test", type=int, default=None, help="smoke test: score N test pairs")
    ap.add_argument("--reuse-dir", default=None,
                    help="earlier ce_out/: keep its scores, score only pairs it does not cover. If every "
                         "train pair is covered, one model is fine-tuned on all train pairs (no OOF needed)")
    args = ap.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device {device}, GPUs {torch.cuda.device_count()}, model {args.model}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pdir = Path(args.pairs_dir)

    tr = pd.read_parquet(pdir / "train_pairs.parquet", columns=["i", "j", "label", "p"])
    n_pos = int(tr["label"].sum())
    tr = tr[focus_set(tr, args.min_p, args.top_k)].reset_index(drop=True)
    log(f"focus set (p >= {args.min_p}, top {args.top_k}/S1) keeps {tr['label'].sum() / max(1, n_pos):.4f} "
        f"of the train positives")
    if args.limit_train:
        tr = tr.sample(min(args.limit_train, len(tr)), random_state=0).reset_index(drop=True)
    te = pd.concat([pd.read_parquet(f, columns=["i", "j", "p"]) for f in sorted((pdir / "test_pairs").glob("part-*.parquet"))],
                   ignore_index=True)
    n_te = len(te)
    te = te[focus_set(te, args.min_p, args.top_k)].reset_index(drop=True)
    log(f"test focus set: {len(te):,} of {n_te:,} pairs")
    if args.limit_test:
        te = te.head(args.limit_test)
    te_known = pd.DataFrame(columns=["i", "j", "ce"])
    train_known = None
    if args.reuse_dir:
        old_te = pd.read_parquet(Path(args.reuse_dir) / "ce_test.parquet")
        m = te.merge(old_te, on=["i", "j"], how="left")
        known = m["ce"].notna().to_numpy()
        te_known = m.loc[known, ["i", "j", "ce"]]
        te = te[~known].reset_index(drop=True)
        old_tr = pd.read_parquet(Path(args.reuse_dir) / "ce_train.parquet")
        mt = tr.merge(old_tr, on=["i", "j"], how="left")
        if mt["ce"].notna().all():
            train_known = mt["ce"].to_numpy(np.float32)
        log(f"reuse: {known.sum():,} test pairs already scored, {len(te):,} new; train scores "
            f"{'reused' if train_known is not None else 'recomputed (pairs differ)'}")
    log(f"train pairs {len(tr):,} (positives {tr['label'].mean():.1%}), test pairs {len(te):,}")
    ta, tb = pair_texts(args.train_records, tr["i"].to_numpy(), tr["j"].to_numpy())
    ea, eb = (pair_texts(args.test_records, te["i"].to_numpy(), te["j"].to_numpy()) if len(te)
              else (np.array([], dtype=object), np.array([], dtype=object)))
    log(f"texts ready, e.g. A='{ta[0]}'  B='{tb[0]}'")

    y = tr["label"].to_numpy().astype(np.float32)
    fold = fold_of(tr["i"].to_numpy(), args.folds)
    oof = np.zeros(len(tr), np.float32)
    test_score = np.zeros(len(te), np.float32)
    if train_known is not None:          # scores for train pairs exist: one model on all of them
        oof = train_known
        log(f"fine-tuning one model on all {len(tr):,} train pairs to score the {len(te):,} new test pairs")
        m = Scorer(args.model, args.max_len, device)
        m.train(ta, tb, y, args.epochs, args.bs, args.lr, seed=0)
        if len(te):
            test_score = m.predict(ea, eb, args.infer_bs)
        args.folds = 0
    for k in range(args.folds):
        fit, val = fold != k, fold == k
        log(f"fold {k}: fine-tuning on {fit.sum():,} pairs, scoring {val.sum():,} held-out pairs")
        m = Scorer(args.model, args.max_len, device)
        m.train(ta[fit], tb[fit], y[fit], args.epochs, args.bs, args.lr, seed=k)
        oof[val] = m.predict(ta[val], tb[val], args.infer_bs)
        auc = _auc(y[val], oof[val])
        log(f"fold {k}: held-out AUC {auc:.4f}, accuracy@0.5 {((oof[val] > 0.5) == (y[val] > 0.5)).mean():.4f}")
        if len(te):
            test_score += m.predict(ea, eb, args.infer_bs) / args.folds
        del m
        torch.cuda.empty_cache()

    pd.DataFrame({"i": tr["i"], "j": tr["j"], "ce": oof}).to_parquet(out / "ce_train.parquet", index=False)
    new_te = pd.DataFrame({"i": te["i"], "j": te["j"], "ce": test_score})
    all_te = pd.concat([te_known.astype(new_te.dtypes.to_dict()), new_te], ignore_index=True) if len(te_known) else new_te
    all_te.to_parquet(out / "ce_test.parquet", index=False)
    summary = {"model": args.model, "train_pairs": len(tr), "test_pairs": len(all_te), "test_pairs_scored_now": len(te),
               "oof_auc": _auc(y, oof), "minutes": round((time.time() - T0) / 60, 1)}
    (out / "ce_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n===== CROSS-ENCODER SUMMARY (paste this) =====")
    print(json.dumps(summary, indent=1))


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    pos, neg = p[y > 0.5], p[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


if __name__ == "__main__":
    main()
