#!/usr/bin/env bash
# One-time setup inside a SageMaker Studio JupyterLab space (File > New > Terminal).
#
#   curl -fsSL https://raw.githubusercontent.com/Soham-Shah-20072004/Amazon_Business_Entity_Resolution/soham/pipeline-v1/scripts/aws/setup_sagemaker.sh \
#     | bash -s -- s3://YOUR-BUCKET/dataset.zip
#
# The argument is one .zip in S3 or an S3 folder prefix with the extracted
# files. The space's execution role must be able to read the bucket (the
# default SageMaker role can read buckets whose name contains "sagemaker";
# otherwise attach AmazonS3ReadOnlyAccess to the role in IAM).
# Result: ~/ber (repo + .venv) and ~/ber/dataset -> folder with train/ and test/.
set -euo pipefail
DATA_S3=${1:?usage: setup_sagemaker.sh s3://bucket/dataset.zip-or-prefix}
BRANCH=${BRANCH:-soham/pipeline-v1}
REPO=https://github.com/Soham-Shah-20072004/Amazon_Business_Entity_Resolution.git

cd ~
if [ -d ber ]; then git -C ber pull -q; else git clone -q -b "$BRANCH" "$REPO" ber; fi
cd ber
python3 -m venv .venv
. .venv/bin/activate
pip install -q -U pip
pip install -q -r requirements.txt
command -v aws >/dev/null || pip install -q awscli

mkdir -p ~/data
if [[ "$DATA_S3" == *.zip ]]; then
  aws s3 cp --only-show-errors "$DATA_S3" ~/data/dataset.zip
  if command -v unzip >/dev/null; then unzip -q -o ~/data/dataset.zip -d ~/data/raw
  else python -m zipfile -e ~/data/dataset.zip ~/data/raw; fi
  rm -f ~/data/dataset.zip
else
  aws s3 sync --only-show-errors "$DATA_S3" ~/data/raw
fi
S1=$(find ~/data/raw -name train_source1.tsv | head -1)
[ -n "$S1" ] || { echo "train_source1.tsv not found under ~/data/raw"; exit 1; }
ln -sfn "$(dirname "$(dirname "$S1")")" ~/ber/dataset
echo "dataset -> $(readlink ~/ber/dataset)"
ls -la ~/ber/dataset/train ~/ber/dataset/test
echo "CPUs: $(nproc)"; free -g; df -h ~ | tail -1
echo "Setup done. Next:  cd ~/ber && . .venv/bin/activate"
