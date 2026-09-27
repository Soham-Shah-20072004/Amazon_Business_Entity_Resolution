#!/usr/bin/env bash
# Stage B on Kaggle: train on a sample of train S1s -> predict all test S1s -> output/*.tsv
#   SMOKE=1 bash ber/scripts/kaggle/stage_b.sh   # ~15 min check on a slice of the data
#   bash ber/scripts/kaggle/stage_b.sh           # full run (use "Save & Run All")
# Optional env: TAG (default m1), S1_SAMPLE (default 60000), VIEWS (e.g. "full addr skel"),
#               NPROBE (IVF cells searched, default 32; lower = faster, higher = better recall)
#               B_ARGS extra retrieval flags for train AND predict, e.g.
#               TAG=m3 VIEWS="full addr skel name" B_ARGS="--reverse-pool 1 --exact-gpu 1" bash ...
set -euo pipefail
cd "$(dirname "$0")/../.."
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  FAISS=faiss-gpu-cu12; echo "GPU found: nearest-neighbour search will run on GPU"
else
  FAISS=faiss-cpu
fi
pip install -q $FAISS unidecode rapidfuzz polars 2>&1 | grep -v -i "warning\|notice" || true

S1=$(find -L /kaggle/input -name train_source1.tsv 2>/dev/null | head -1 || true)
if [ -z "$S1" ]; then
  Z=$(find -L /kaggle/input -name "*.zip" | head -1)
  [ -n "$Z" ] || { echo "No train_source1.tsv or .zip under /kaggle/input - add the dataset"; exit 1; }
  mkdir -p /tmp/ber_data && python -m zipfile -e "$Z" /tmp/ber_data
  S1=$(find /tmp/ber_data -name train_source1.tsv | head -1)
fi
ROOT=$(dirname "$(dirname "$S1")")
echo "dataset: $ROOT"; echo "machine: $(nproc) CPUs"; free -g | head -2
TAG=${TAG:-m1}
VIEW_ARGS="${VIEWS:+--views $VIEWS} ${NPROBE:+--nprobe $NPROBE}"
SCR=/tmp/ber_scratch

if [ "${SMOKE:-0}" = "1" ]; then
  WORK=/tmp/ber_smoke_b; OUT=/tmp/ber_smoke_out
  [ -f "$WORK/test/records.parquet" ] || python scripts/prepare_data.py --data-root "$ROOT" --out "$WORK" --max-rows 300000
  python scripts/run_stage_b.py train --work "$WORK" --s1-sample 5000 --tag smoke --scratch $SCR --fit-sample 100000 $VIEW_ARGS
  python scripts/run_stage_b.py predict --work "$WORK" --tag smoke --scratch $SCR --out $OUT --chunk 10000 --limit-s1 20000
  echo "(smoke run: submission check skipped - the data slice is incomplete by design)"
  exit 0
fi

WORK=/kaggle/working/work; OUT=/kaggle/working/output
mkdir -p "$WORK"
if [ ! -f "$WORK/test/records.parquet" ]; then
  PREV=$(find -L /kaggle/input -path "*work/test/records.parquet" 2>/dev/null | head -1 || true)
  if [ -n "$PREV" ]; then           # prepared data attached from an earlier notebook version
    P=$(dirname "$(dirname "$PREV")"); ln -sfn "$P/train" "$WORK/train"; ln -sfn "$P/test" "$WORK/test"
    echo "reusing prepared data from $P"
  else
    python scripts/prepare_data.py --data-root "$ROOT" --out "$WORK"
  fi
fi
python scripts/run_stage_b.py train --work "$WORK" --s1-sample "${S1_SAMPLE:-60000}" --tag "$TAG" --scratch $SCR $VIEW_ARGS ${B_ARGS:-}
rm -f $SCR/Z_train_*              # free disk before building the test vectors
python scripts/run_stage_b.py predict --work "$WORK" --tag "$TAG" --scratch $SCR --out "$OUT" ${B_ARGS:-}
cp "$WORK/models/$TAG/meta.json" "$OUT/meta_$TAG.json"
rm -f $SCR/Z_test_*
V=$(find -L /kaggle/input -name validate_submission.py 2>/dev/null | head -1 || true)
if [ -n "$V" ]; then
  python "$V" --matching "$OUT/matching_results.tsv" --candidate "$OUT/candidate_pairs.tsv" --test-dir "$ROOT/test"
else
  python scripts/check_submission.py --out "$OUT" --test-dir "$ROOT/test"
fi
