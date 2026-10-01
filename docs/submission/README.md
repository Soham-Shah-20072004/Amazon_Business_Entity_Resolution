# Business Entity Resolution: reproduction guide

Amazon ML Challenge 2026. For every Source 1 (S1) record, find the Source 2 / Source 3
records of the same business. This folder holds the full pipeline that produced
`output/matching_results.tsv` (final matches) and `output/candidate_pairs.tsv`
(the last candidate set passed to the final matcher).

Methodology: see `Documentation_template.md` at the root of the ZIP.

## 1. Environment

| | Used for the submission |
|---|---|
| Machine | Kaggle notebooks: 4 vCPU, ~30 GB RAM; GPU steps on 2 × NVIDIA T4 (16 GB each) |
| Python | 3.10–3.12 |
| Disk | ~40 GB free in `/tmp` (memory-mapped vectors) plus ~15 GB for outputs |

```bash
pip install -r requirements.txt
# GPU machine only (exact nearest-neighbour search on the GPU):
pip uninstall -y faiss-cpu && pip install faiss-gpu-cu12
```

The cross-encoder step downloads the pretrained model
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 118M
parameters) from the Hugging Face Hub once. No other external data or lookups are used.

## 2. Data layout

```
<DATA>/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
<DATA>/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
```

All commands below run from this folder's `src/` directory:

```bash
cd src
export DATA=/path/to/dataset          # folder containing train/ and test/
export WORK=work                      # prepared data, models and cached pairs
export SCR=/tmp/ber_scratch           # memory-mapped vectors (large, temporary)
```

## 3. Pipeline (final submission = steps 1 → 6)

| Step | Command | Machine | Time (Kaggle) |
|---|---|---|---|
| 1. Prepare: parse TSVs, normalise text (NFKC, transliteration, canonical abbreviations), write parquet | `python scripts/prepare_data.py --data-root $DATA --out $WORK` | CPU | ~25 min |
| 2. Stage B train: blocking on 60,000 sampled train S1 against the full S2/S3 pool, pre-ranker + matcher (LightGBM, out-of-fold), decision rule | `python scripts/run_stage_b.py train --work $WORK --s1-sample 60000 --tag m1 --scratch $SCR` | CPU (GPU optional) | ~2–3 h |
| 3. Stage B predict: same blocking and models on all 1,732,544 test S1; caches all scored test pairs | `python scripts/run_stage_b.py predict --work $WORK --tag m1 --scratch $SCR --out output_m1` | CPU (GPU optional) | ~5 h CPU |
| 4. Cross-encoder: fine-tune multilingual MiniLM on the focus set (2-fold out-of-fold), score test focus pairs | `python scripts/gpu/cross_encoder.py --pairs-dir $WORK/models/m1 --train-records $WORK/train/records.parquet --test-records $WORK/test/records.parquet --out ce_out` | GPU | ~1 h 45 m |
| 5. Retrieval v2 (exact GPU search + reverse search + name view) with the step-2 models, then cross-encoder scores for the new pairs only | see 5a–5c below | GPU | ~3 h 30 m |
| 6. Stacked matcher + decision → output files | see 6 below | CPU or GPU | ~15 min |

**5a. Reuse the step-2 models under a new tag**
```bash
mkdir -p $WORK/models/m2
cp $WORK/models/m1/{prerank.txt,matcher.txt,meta.json,train_pairs.parquet,queries.npy,truth.parquet} $WORK/models/m2/
```
**5b. Predict with the improved retrieval**
```bash
python scripts/run_stage_b.py predict --work $WORK --tag m2 --scratch $SCR --out output_stage_b2 \
  --exact-gpu 1 --reverse-pool 1 --extra-views name
```
**5c. Cross-encoder scores for pairs not scored in step 4**
```bash
python scripts/gpu/cross_encoder.py --pairs-dir $WORK/models/m2 --train-records $WORK/train/records.parquet \
  --test-records $WORK/test/records.parquet --out ce_out_b2 --epochs 1 --reuse-dir ce_out
```
**6. Stacked matcher (LightGBM + cross-encoder + sibling features), decision rule, output files, format check**
```bash
python scripts/run_combine.py --pairs-dir $WORK/models/m2 \
  --extra-train ce_out_b2/ce_train.parquet --extra-test ce_out_b2/ce_test.parquet \
  --train-records $WORK/train/records.parquet --test-records $WORK/test/records.parquet \
  --test-dir $DATA/test --out output --no-ablate
```
Result: `src/output/matching_results.tsv` and `src/output/candidate_pairs.tsv` (copied to the ZIP's top-level `output/`). The script prints a
`COMBINE SUMMARY` (out-of-fold macro F0.5 0.9569 on the 60k held-out train S1) and runs the
format check (`scripts/check_submission.py`).

## 4. Same pipeline as Kaggle one-liners

Clone or copy this folder as `ber/` in a Kaggle notebook (Internet on), attach the dataset and
the previous notebooks' outputs, and run:

| Step | Notebook | Command |
|---|---|---|
| 1 | CPU | `bash ber/scripts/kaggle/stage_a.sh` |
| 2–3 | CPU or GPU T4 ×2 | `bash ber/scripts/kaggle/stage_b.sh` (attach step 1) |
| 4 | GPU T4 ×2 | `bash ber/scripts/kaggle/gpu_ce.sh` (attach steps 1, 2–3) |
| 5–6 | GPU T4 ×2 | `bash ber/scripts/kaggle/stage_b2.sh` (attach steps 1, 2–3, 4) |

Each script accepts `SMOKE=1` for a ~15–25 min end-to-end check on a data slice.

## 5. Diagnostics and experiments (not needed for the submission)

| Script | Purpose |
|---|---|
| `scripts/diagnose.py` | points lost per error type (false match / scored too low / never a candidate), error profiles, examples |
| `scripts/recall_check.py` | retrieval recall on real train labels per blocker (IVF vs exact, + reverse, + name view) |
| `scripts/inspect_country.py` | label-free per-country view of test predictions (France has no training labels) |
| `scripts/run_blocking.py` | blocking-only recall report |
| `scripts/gpu/biencoder.py`, `scripts/merge_bienc.py` | learned bi-encoder retrieval + sibling propagation and a gated merge (built, not part of the submitted result) |

## 6. Reproducibility notes

- Seeds are fixed (sampling 42, LightGBM 7 / 42, fold assignment by a hash of the S1 row id).
- The 60,000 training S1 are drawn with seed 42 from the S1 rows in file order, so the same
  data gives the same sample.
- GPU steps use fp16; scores can differ in the last decimals between GPU types. The decision
  thresholds are re-tuned on out-of-fold scores in every run.
- Folder layout: `src/business_entity_resolution/er/` holds the library (text normalisation,
  retrieval, features, models, metric); `src/scripts/` holds the command-line steps.
