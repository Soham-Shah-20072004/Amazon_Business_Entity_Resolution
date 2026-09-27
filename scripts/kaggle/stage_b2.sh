#!/usr/bin/env bash
# Stage B2 (GPU T4 x2 notebook): better SEARCH, reusing the trained m1 models (no retraining).
#   exact GPU search (instead of approximate) + reverse search (every S2/S3 record proposes its
#   2 nearest S1s) + name-only search view  ->  predict all test S1s  ->  cross-encoder for the
#   NEW pairs only  ->  stacked combine  ->  output/*.tsv
# Inputs to attach: the dataset, the Stage A notebook (work/{train,test}/records.parquet), the
# Stage B notebook (work/models/m1/...) and the GPU cross-encoder notebook (ce_out/).
#   SMOKE=1 bash ber/scripts/kaggle/stage_b2.sh        # ~20-25 min end-to-end check on a small slice
#   bash ber/scripts/kaggle/stage_b2.sh                # full run (use "Save & Run All")
#   RECALL_ONLY=1 bash ber/scripts/kaggle/stage_b2.sh  # recall check on real train labels (CPU ok, ~1 h)
# Optional env: EXTRA_VIEWS (default "name"), EPOCHS (cross-encoder, default 1), TAG (default m2)
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU=0
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then GPU=1; fi
if [ "$GPU" = "1" ]; then
  pip install -q faiss-gpu-cu12 unidecode rapidfuzz polars lightgbm 2>&1 | grep -v -i "warning\|notice" || true
  python -c "import faiss; assert faiss.get_num_gpus() > 0; print('faiss GPUs:', faiss.get_num_gpus())" || {
    echo "faiss-gpu not usable - falling back to faiss-cpu (approximate search on CPU, slower)"
    pip uninstall -y -q faiss-gpu-cu12 || true; pip install -q faiss-cpu; }
else
  pip install -q faiss-cpu unidecode rapidfuzz polars lightgbm 2>&1 | grep -v -i "warning\|notice" || true
fi
echo "machine: $(nproc) CPUs, GPU=$GPU"; free -g | head -2
TAG=${TAG:-m2}
SCR=/tmp/ber_scratch
first_existing() { while read -r p; do [ -e "$p" ] && { echo "$p"; return; }; done; }
S1=$(find -L /kaggle/input -name train_source1.tsv 2>/dev/null | head -1 || true)
ROOT=$(dirname "$(dirname "$S1")")
TEST_DIR=$(dirname "$(find -L /kaggle/input -name test_source1.tsv 2>/dev/null | head -1)")
PREV=$(find -L /kaggle/input -path "*work/test/records.parquet" 2>/dev/null | first_existing || true)
M1=$(find -L /kaggle/input -path "*models/m1/meta.json" 2>/dev/null | first_existing || true)
CE_OLD=$(find -L /kaggle/input -name ce_train.parquet 2>/dev/null | first_existing || true)
[ -n "$S1" ] || { echo "attach the competition dataset"; exit 1; }

if [ "${SMOKE:-0}" = "1" ]; then
  WORK=/tmp/ber_smoke_b2; OUT=/tmp/ber_smoke_b2_out
  [ -f "$WORK/test/records.parquet" ] || python scripts/prepare_data.py --data-root "$ROOT" --out "$WORK" --max-rows 300000
  # a model trained like m1 (no reverse, no name view), then predicted WITH the new search
  python scripts/run_stage_b.py train --work "$WORK" --s1-sample 5000 --tag smoke --scratch $SCR --fit-sample 100000
  rm -f $SCR/Z_train_*
  python scripts/recall_check.py --work "$WORK" --s1-sample 3000 --scratch /tmp/ber_scratch_rc --fit-sample 100000
  rm -rf /tmp/ber_scratch_rc
  python scripts/run_stage_b.py predict --work "$WORK" --tag smoke --scratch $SCR --out "$OUT" --chunk 10000 \
    --limit-s1 20000 --exact-gpu 1 --reverse-pool 1 --extra-views ${EXTRA_VIEWS:-name}
  rm -f $SCR/Z_test_*
  REUSE=(); [ -n "$CE_OLD" ] && REUSE=(--reuse-dir "$(dirname "$CE_OLD")")
  python scripts/gpu/cross_encoder.py --pairs-dir "$WORK/models/smoke" --train-records "$WORK/train/records.parquet" \
    --test-records "$WORK/test/records.parquet" --out /tmp/ce_smoke2 --epochs 1 --limit-train 3000 --limit-test 5000 "${REUSE[@]}"
  python scripts/run_combine.py --pairs-dir "$WORK/models/smoke" --extra-train /tmp/ce_smoke2/ce_train.parquet \
    --extra-test /tmp/ce_smoke2/ce_test.parquet --train-records "$WORK/train/records.parquet" \
    --test-records "$WORK/test/records.parquet" --out "$OUT/combined" --no-ablate
  echo "SMOKE OK (numbers on this slice mean nothing; this only checks the code paths)"
  exit 0
fi

[ -n "$PREV" ] || { echo "attach the Stage A notebook (prepared records)"; exit 1; }
WORK=/kaggle/working/work
mkdir -p "$WORK/models"
P=$(dirname "$(dirname "$PREV")"); ln -sfn "$P/train" "$WORK/train"; ln -sfn "$P/test" "$WORK/test"
echo "prepared data: $P"

if [ "${RECALL_ONLY:-0}" = "1" ]; then
  python scripts/recall_check.py --work "$WORK" --s1-sample "${S1_SAMPLE:-20000}" --scratch /tmp/ber_scratch_rc
  exit 0
fi

[ -n "$M1" ] || { echo "attach the Stage B notebook (work/models/m1)"; exit 1; }
[ -n "$CE_OLD" ] || { echo "attach the GPU cross-encoder notebook (ce_out/)"; exit 1; }
MD="$WORK/models/$TAG"; mkdir -p "$MD"
for f in prerank.txt matcher.txt meta.json train_pairs.parquet queries.npy truth.parquet; do
  cp "$(dirname "$M1")/$f" "$MD/"
done
echo "models: $(dirname "$M1") -> $MD; old cross-encoder scores: $(dirname "$CE_OLD")"

python scripts/run_stage_b.py predict --work "$WORK" --tag "$TAG" --scratch $SCR --out /kaggle/working/output_stage_b2 \
  --exact-gpu 1 --reverse-pool 1 --extra-views ${EXTRA_VIEWS:-name}
rm -f $SCR/Z_test_*
python scripts/gpu/cross_encoder.py --pairs-dir "$MD" --train-records "$WORK/train/records.parquet" \
  --test-records "$WORK/test/records.parquet" --out /kaggle/working/ce_out --epochs "${EPOCHS:-1}" \
  --reuse-dir "$(dirname "$CE_OLD")"
python scripts/run_combine.py --pairs-dir "$MD" --extra-train /kaggle/working/ce_out/ce_train.parquet \
  --extra-test /kaggle/working/ce_out/ce_test.parquet --train-records "$WORK/train/records.parquet" \
  --test-records "$WORK/test/records.parquet" --test-dir "$TEST_DIR" --out /kaggle/working/output --no-ablate
