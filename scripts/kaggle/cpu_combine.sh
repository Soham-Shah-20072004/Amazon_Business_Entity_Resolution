#!/usr/bin/env bash
# CPU notebook (no accelerator needed): diagnosis + combine on cached outputs, ~15-30 min.
# Inputs to attach: your dataset, the Stage A notebook (prepared records), the Stage B
# notebook (models/<tag>/...) and, if it has finished, the GPU cross-encoder notebook (ce_out/).
#   bash ber/scripts/kaggle/cpu_combine.sh            # diagnosis, then combine -> output/*.tsv
#   DIAG_ONLY=1 bash ber/scripts/kaggle/cpu_combine.sh   # diagnosis only (~5 min)
#   SKIP_DIAG=1 bash ber/scripts/kaggle/cpu_combine.sh   # combine only
#   INSPECT=1 bash ber/scripts/kaggle/cpu_combine.sh     # per-country test inspection only (~5 min)
#   ABLATE=0 ...  # train only the full stacked model, not the 2 comparison variants (~6 min less)
# Without a cross-encoder input, combine still adds sibling features + the per-S1 decision rule.
set -euo pipefail
cd "$(dirname "$0")/../.."
pip install -q lightgbm rapidfuzz unidecode polars 2>&1 | grep -v -i "warning\|notice" || true
TAG=${TAG:-m1}

first_existing() { while read -r p; do [ -f "$p" ] && { echo "$p"; return; }; done; }
PAIRS=$(find -L /kaggle/input -path "*models/$TAG/train_pairs.parquet" 2>/dev/null | first_existing || true)
TRAIN_REC=$(find -L /kaggle/input -path "*work/train/records.parquet" 2>/dev/null | first_existing || true)
TEST_REC=$(find -L /kaggle/input -path "*work/test/records.parquet" 2>/dev/null | first_existing || true)
CE_TRAIN=$(find -L /kaggle/input -name ce_train.parquet 2>/dev/null | first_existing || true)   # notebook output or uploaded dataset
TEST_DIR=$(dirname "$(find -L /kaggle/input -name test_source1.tsv 2>/dev/null | head -1)")
for v in PAIRS TRAIN_REC TEST_REC; do
  [ -n "${!v}" ] || { echo "missing input for $v - attach the Stage A and Stage B notebooks"; exit 1; }
  echo "$v: ${!v}"
done
PDIR=$(dirname "$PAIRS")

if [ "${INSPECT:-0}" = "1" ]; then
  python scripts/inspect_country.py --pairs-dir "$PDIR" --test-records "$TEST_REC" \
    --country "${COUNTRY:-France}" --out /kaggle/working/inspect
  exit 0
fi
if [ "${SKIP_DIAG:-0}" != "1" ]; then
  python scripts/diagnose.py --pairs-dir "$PDIR" --train-records "$TRAIN_REC" --out /kaggle/working/diag
fi
[ "${DIAG_ONLY:-0}" = "1" ] && exit 0

EXTRA=()
[ "${ABLATE:-1}" = "0" ] && EXTRA+=(--no-ablate)
if [ -n "$CE_TRAIN" ]; then
  echo "CE_TRAIN: $CE_TRAIN"
  EXTRA=(--extra-train "$CE_TRAIN" --extra-test "$(dirname "$CE_TRAIN")/ce_test.parquet")
else
  echo "no cross-encoder output attached: combine with sibling features only"
fi
python scripts/run_combine.py --pairs-dir "$PDIR" "${EXTRA[@]}" --train-records "$TRAIN_REC" \
  --test-records "$TEST_REC" --test-dir "$TEST_DIR" --out /kaggle/working/output
