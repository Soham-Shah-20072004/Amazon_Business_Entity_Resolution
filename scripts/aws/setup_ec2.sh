#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 24.04 EC2 instance for the pipeline.
#
#   curl -fsSL https://raw.githubusercontent.com/Soham-Shah-20072004/Amazon_Business_Entity_Resolution/soham/pipeline-v1/scripts/aws/setup_ec2.sh \
#     | bash -s -- s3://YOUR-BUCKET/dataset.zip
#
# The argument is either one .zip in S3 or an S3 folder prefix holding the
# extracted files. The instance needs an IAM role that can read the bucket.
# Result: ~/ber (repo, branch soham/pipeline-v1, venv in .venv) and
# ~/ber/dataset -> folder containing train/ and test/.
set -euo pipefail
DATA_S3=${1:?usage: setup_ec2.sh s3://bucket/dataset.zip-or-prefix}
BRANCH=${BRANCH:-soham/pipeline-v1}
REPO=https://github.com/Soham-Shah-20072004/Amazon_Business_Entity_Resolution.git

sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-dev build-essential unzip git htop tmux >/dev/null
command -v aws >/dev/null || sudo snap install aws-cli --classic

cd ~
[ -d ber ] || git clone -q -b "$BRANCH" "$REPO" ber
cd ber
python3 -m venv .venv
. .venv/bin/activate
pip install -q -U pip
pip install -q -r requirements.txt

mkdir -p ~/data
if [[ "$DATA_S3" == *.zip ]]; then
  aws s3 cp --only-show-errors "$DATA_S3" ~/data/dataset.zip
  unzip -q -o ~/data/dataset.zip -d ~/data/raw
else
  aws s3 sync --only-show-errors "$DATA_S3" ~/data/raw
fi
S1=$(find ~/data/raw -name train_source1.tsv | head -1)
[ -n "$S1" ] || { echo "train_source1.tsv not found under ~/data/raw"; exit 1; }
ln -sfn "$(dirname "$(dirname "$S1")")" ~/ber/dataset
echo "dataset -> $(readlink ~/ber/dataset)"
ls -la ~/ber/dataset/train ~/ber/dataset/test
nproc; free -g
echo "Setup done. Next:  cd ~/ber && . .venv/bin/activate && tmux new -s ber"
