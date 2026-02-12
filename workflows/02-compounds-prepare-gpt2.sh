#!/bin/bash

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir
mkdir -p ${LEARNING_SOURCE_DIR}/compounds/logs/
nohup bash -c 'python src/compounds/dataset/prepare_gpt2.py assets/configs/compounds.yaml --output_dataset_dir ${LEARNING_SOURCE_DIR}/compounds/organix13/compounds/training_ready_hf_dataset/gpt2' > \
    ${LEARNING_SOURCE_DIR}/compounds/logs/compounds-prepare-gpt2-`date +%Y-%m-%d_%H-%M-%S`.log 2>&1 &
