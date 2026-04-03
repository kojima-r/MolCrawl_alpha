#!/bin/bash
# Fine-tune the compounds GPT-2 (medium) model on ChEMBL.
#
# Prerequisites:
#   - compounds GPT-2 pretraining checkpoint must exist in
#       $LEARNING_SOURCE_DIR/compounds/gpt2-output/compounds-medium/
#   - ChEMBL training_ready_hf_dataset must be prepared via
#       workflows/01-compounds_chembl-prepare.sh
#
# Usage:
#   export LEARNING_SOURCE_DIR=<path>
#   bash workflows/03a-compounds_chembl-train-gpt2-medium.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

check_learning_source_dir
auto_select_gpu 15

LOG_DIR="${LEARNING_SOURCE_DIR}/compounds/chembl/logs"
mkdir -p "${LOG_DIR}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} PYTHONUNBUFFERED=1 \
nohup bash -c '$PYTHON molcrawl/gpt2/train.py \
    gpt2/configs/compounds/train_gpt2_chembl_medium.py' \
    > "${LOG_DIR}/compounds_chembl-train-gpt2-medium-$(date +%Y-%m-%d_%H-%M-%S).log" 2>&1 &

echo "GPT-2 fine-tuning running in background (GPU ${CUDA_VISIBLE_DEVICES})."
echo "Logs: ${LOG_DIR}/"
