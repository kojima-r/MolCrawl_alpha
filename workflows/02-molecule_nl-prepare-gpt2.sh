#!/bin/bash

set -e

# Load common functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_functions.sh"

# Check LEARNING_SOURCE_DIR
check_learning_source_dir
mkdir -p ${LEARNING_SOURCE_DIR}/molecule_nl/logs/
nohup bash -c 'python src/molecule_related_nl/dataset/prepare_gpt2.py assets/configs/molecules_nl.yaml --output_dataset_dir ${LEARNING_SOURCE_DIR}/molecule_nl/training_ready_hf_dataset/gpt2' > \
    ${LEARNING_SOURCE_DIR}/molecule_nl/logs/molecule_nl-prepare-gpt2-`date +%Y-%m-%d_%H-%M-%S`.log 2>&1 &
