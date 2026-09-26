#!/usr/bin/env bash
# Stage A on Kaggle: install -> find dataset -> prepare -> blocking report.
#   SMOKE=1 bash ber/scripts/kaggle/stage_a.sh     # ~5 min check on a slice of the data
#   bash ber/scripts/kaggle/stage_a.sh             # full run (use "Save & Run All")
set -euo pipefail
cd "$(dirname "$0")/../.."
pip install -q faiss-cpu unidecode rapidfuzz 2>&1 | grep -v -i "warning\|notice" || true

S1=$(find -L /kaggle/input -name train_source1.tsv 2>/dev/null | head -1 || true)
if [ -z "$S1" ]; then            # dataset uploaded as a zip that Kaggle did not unpack
  Z=$(find -L /kaggle/input -name "*.zip" | head -1)
  [ -n "$Z" ] || { echo "No train_source1.tsv or .zip under /kaggle/input - add the dataset"; exit 1; }
  mkdir -p /tmp/ber_data && python -m zipfile -e "$Z" /tmp/ber_data
  S1=$(find /tmp/ber_data -name train_source1.tsv | head -1)
fi
ROOT=$(dirname "$(dirname "$S1")")
WORK=/kaggle/working/work
echo "dataset: $ROOT"
echo "machine: $(nproc) CPUs"; free -g | head -2; df -h /kaggle/working /tmp | tail -2

if [ "${SMOKE:-0}" = "1" ]; then
  WORK=/tmp/ber_smoke
  python scripts/prepare_data.py --data-root "$ROOT" --out "$WORK" --max-rows 300000
  python scripts/run_blocking.py --work "$WORK" --split train --s1-sample 5000 --tag smoke \
    --scratch /tmp/ber_scratch --fit-sample 100000
else
  [ -f "$WORK/test/records.parquet" ] || python scripts/prepare_data.py --data-root "$ROOT" --out "$WORK"
  python scripts/run_blocking.py --work "$WORK" --split train --s1-sample "${S1_SAMPLE:-50000}" \
    --tag "${TAG:-b1}" --scratch /tmp/ber_scratch
fi
