#!/usr/bin/env bash
# Final merge (CPU notebook, ~30-45 min): B2 stacked result + bi-encoder candidates -> output/*.tsv
# Inputs: the dataset, the Stage A notebook (prepared records), the B2 notebook (work/models/m2,
# ce_out/, output/) and the bi-encoder notebook (bienc_out/). Do NOT attach stagec-gpu here.
set -euo pipefail
cd "$(dirname "$0")/../.."
pip install -q lightgbm rapidfuzz unidecode polars 2>&1 | grep -v -i "warning\|notice" || true
first_existing() { while read -r p; do [ -e "$p" ] && { echo "$p"; return; }; done; }
PREV=$(find -L /kaggle/input -path "*work/test/records.parquet" 2>/dev/null | first_existing || true)
M2=$(find -L /kaggle/input -path "*work/models/m2/meta.json" 2>/dev/null | first_existing || true)
BI=$(find -L /kaggle/input -name bienc_test.parquet 2>/dev/null | first_existing || true)
TEST_DIR=$(dirname "$(find -L /kaggle/input -name test_source1.tsv 2>/dev/null | head -1)")
for v in PREV M2 BI; do [ -n "${!v}" ] || { echo "missing input for $v"; exit 1; }; echo "$v: ${!v}"; done
W=$(dirname "$(dirname "$PREV")"); MD=$(dirname "$M2"); NB=${M2%/work/models/m2/meta.json}
STACK=$(find -L "$NB" -name stack_rule.json 2>/dev/null | head -1 || true)
if [ -z "$STACK" ]; then        # B2 ran before the combine saved its scores: recompute them (~15 min)
  python scripts/run_combine.py --pairs-dir "$MD" --extra-train "$NB/ce_out/ce_train.parquet" \
    --extra-test "$NB/ce_out/ce_test.parquet" --train-records "$W/train/records.parquet" \
    --test-records "$W/test/records.parquet" --out /kaggle/working/output_stack --no-ablate
  SD=/kaggle/working/output_stack
else
  SD=$(dirname "$STACK")
fi
python scripts/merge_bienc.py --work "$W" --stack-dir "$SD" --pairs-dir "$MD" --bienc-dir "$(dirname "$BI")" \
  --test-dir "$TEST_DIR" --out /kaggle/working/output_merged
