#!/usr/bin/env bash
# Bi-encoder retrieval + sibling propagation (GPU T4 x2 notebook), ~1.5-2 h.
# Inputs: the dataset, the Stage A notebook (work/{train,test}/records.parquet; prepared here if
# missing, +30 min) and the Stage B notebook (work/models/m1: queries.npy, train_pairs.parquet).
#   SMOKE=1 bash ber/scripts/kaggle/bienc.sh    # ~10 min check on a small slice
#   bash ber/scripts/kaggle/bienc.sh            # full run ("Save & Run All") -> /kaggle/working/bienc_out
set -euo pipefail
cd "$(dirname "$0")/../.."
pip install -q faiss-gpu-cu12 unidecode rapidfuzz polars 2>&1 | grep -v -i "warning\|notice" || true
python -c "import faiss; print('faiss GPUs:', faiss.get_num_gpus())" || { pip install -q faiss-cpu; }
nvidia-smi --query-gpu=name,memory.total --format=csv || true
first_existing() { while read -r p; do [ -e "$p" ] && { echo "$p"; return; }; done; }
S1=$(find -L /kaggle/input -name train_source1.tsv 2>/dev/null | head -1 || true)
[ -n "$S1" ] || { echo "attach the competition dataset"; exit 1; }
ROOT=$(dirname "$(dirname "$S1")")

if [ "${SMOKE:-0}" = "1" ]; then
  W=/tmp/ber_smoke_bi
  [ -f "$W/test/records.parquet" ] || python scripts/prepare_data.py --data-root "$ROOT" --out "$W" --max-rows 300000
  python scripts/gpu/biencoder.py --work "$W" --holdout-n 2000 --max-pairs 5000 --out /tmp/bienc_smoke --smoke
  echo "SMOKE OK (numbers on this slice mean nothing)"
  exit 0
fi

PREV=$(find -L /kaggle/input -path "*work/test/records.parquet" 2>/dev/null | first_existing || true)
if [ -n "$PREV" ]; then
  W=$(dirname "$(dirname "$PREV")"); echo "prepared data: $W"
else
  W=/tmp/ber_work; echo "no prepared data attached - preparing (~30 min)"
  python scripts/prepare_data.py --data-root "$ROOT" --out "$W"
fi
Q=$(find -L /kaggle/input -path "*models/m1/queries.npy" 2>/dev/null | first_existing || true)
[ -n "$Q" ] || { echo "attach the Stage B notebook (work/models/m1)"; exit 1; }
python scripts/gpu/biencoder.py --work "$W" --holdout-queries "$Q" \
  --baseline-pairs "$(dirname "$Q")/train_pairs.parquet" --out /kaggle/working/bienc_out
