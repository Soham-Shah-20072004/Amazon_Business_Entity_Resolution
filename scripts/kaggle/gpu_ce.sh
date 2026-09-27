#!/usr/bin/env bash
# GPU notebook (Accelerator: GPU T4 x2): cross-encoder -> stacked LightGBM -> output/*.tsv
# Inputs to attach: your dataset, the Stage A notebook (prepared records) and the Stage B
# notebook (models/<tag>/train_pairs.parquet + test_pairs/).
#   SMOKE=1 bash ber/scripts/kaggle/gpu_ce.sh    # ~10 min check on a few thousand pairs
#   bash ber/scripts/kaggle/gpu_ce.sh            # full run (use "Save & Run All")
# Optional env: TAG (Stage B tag, default m1), MODEL (Hugging Face id), EPOCHS (default 2)
set -euo pipefail
cd "$(dirname "$0")/../.."
pip install -q lightgbm rapidfuzz unidecode polars 2>&1 | grep -v -i "warning\|notice" || true
nvidia-smi --query-gpu=name,memory.total --format=csv || true
TAG=${TAG:-m1}

first_existing() { while read -r p; do [ -f "$p" ] && { echo "$p"; return; }; done; }
PAIRS=$(find -L /kaggle/input -path "*models/$TAG/train_pairs.parquet" 2>/dev/null | first_existing || true)
TRAIN_REC=$(find -L /kaggle/input -path "*work/train/records.parquet" 2>/dev/null | first_existing || true)
TEST_REC=$(find -L /kaggle/input -path "*work/test/records.parquet" 2>/dev/null | first_existing || true)
TEST_DIR=$(dirname "$(find -L /kaggle/input -name test_source1.tsv 2>/dev/null | head -1)")
for v in PAIRS TRAIN_REC TEST_REC; do
  [ -n "${!v}" ] || { echo "missing input for $v - attach the Stage A and Stage B notebooks"; exit 1; }
  echo "$v: ${!v}"
done
PDIR=$(dirname "$PAIRS")
CE_ARGS="--model ${MODEL:-sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2} --epochs ${EPOCHS:-2}"

if [ "${SMOKE:-0}" = "1" ]; then
  python scripts/gpu/cross_encoder.py --pairs-dir "$PDIR" --train-records "$TRAIN_REC" \
    --test-records "$TEST_REC" --out /tmp/ce_smoke $CE_ARGS --limit-train 20000 --limit-test 50000
  echo "(smoke run: combine skipped)"
  exit 0
fi
python scripts/gpu/cross_encoder.py --pairs-dir "$PDIR" --train-records "$TRAIN_REC" \
  --test-records "$TEST_REC" --out /kaggle/working/ce_out $CE_ARGS
python scripts/run_combine.py --pairs-dir "$PDIR" --extra-train /kaggle/working/ce_out/ce_train.parquet \
  --extra-test /kaggle/working/ce_out/ce_test.parquet --test-records "$TEST_REC" \
  --test-dir "$TEST_DIR" --out /kaggle/working/output
