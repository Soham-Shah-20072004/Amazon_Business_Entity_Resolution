#!/usr/bin/env bash
# Build the challenge ZIP:
#   bash tools/make_submission.sh <team_name> [dir with matching_results.tsv + candidate_pairs.tsv] [dest dir]
# Layout produced (as required by the organisers):
#   <team>_submission/output/{matching_results.tsv,candidate_pairs.tsv}
#   <team>_submission/code/business_entity_resolution/{src/,README.md,requirements.txt}
#   <team>_submission/Documentation_template.md
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
TEAM=${1:?usage: make_submission.sh <team_name> [outputs dir] [dest dir]}
OUTSRC=${2:-}
DEST=$(mkdir -p "${3:-$PWD}" && cd "${3:-$PWD}" && pwd)
PKG="$DEST/${TEAM}_submission"
C="$PKG/code/business_entity_resolution"
rm -rf "$PKG" "$DEST/${TEAM}_submission.zip"
mkdir -p "$PKG/output" "$C/src/scripts"
cp -r "$REPO/src/business_entity_resolution" "$C/src/"
for f in prepare_data.py run_stage_b.py run_combine.py check_submission.py run_blocking.py diagnose.py \
         recall_check.py inspect_country.py merge_bienc.py; do
  cp "$REPO/scripts/$f" "$C/src/scripts/"
done
cp -r "$REPO/scripts/gpu" "$REPO/scripts/kaggle" "$C/src/scripts/"
find "$C" -name __pycache__ -prune -exec rm -rf {} +
cp "$REPO/docs/submission/README.md" "$REPO/docs/submission/requirements.txt" "$C/"
cp "$REPO/docs/submission/Documentation_template.md" "$PKG/"
# optional: fill the write-up's placeholders, e.g. MEMBERS="A, B, C" LB_FINAL=0.94 LB_V2=0.939 LB_V3=0.94
D="$PKG/Documentation_template.md"
python3 - "$D" "$TEAM" "${MEMBERS:-}" "${LB_FINAL:-}" "${LB_V2:-}" "${LB_V3:-}" <<'PY'
import sys
path, team, members, final, v2, v3 = sys.argv[1:]
s = open(path, encoding="utf-8").read()
for key, val in (("`<team name>`", team), ("`<names>`", members), ("`<public score>`", final),
                 ("`<lb v2>`", v2), ("`<lb v3>`", v3)):
    if val:
        s = s.replace(key, val)
open(path, "w", encoding="utf-8").write(s)
PY
if [ -n "$OUTSRC" ]; then
  cp "$OUTSRC/matching_results.tsv" "$OUTSRC/candidate_pairs.tsv" "$PKG/output/"
  python3 -m zipfile -c "$DEST/${TEAM}_submission.zip" "$PKG"
  echo "built $DEST/${TEAM}_submission.zip"
else
  echo "NOTE: no outputs given - put matching_results.tsv and candidate_pairs.tsv into $PKG/output/ and zip the folder"
fi
