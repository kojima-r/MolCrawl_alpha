#!/bin/bash

set -e

# Check LEARNING_SOURCE_DIR
if [ -z "$LEARNING_SOURCE_DIR" ]; then
    echo "ERROR: LEARNING_SOURCE_DIR environment variable is not set."
    echo "Please set it before running this script:"
    echo "  export LEARNING_SOURCE_DIR='...'"
    exit 1
fi

echo "DatabaseDir: $LEARNING_SOURCE_DIR"
mkdir -p ${LEARNING_SOURCE_DIR}/logs
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 python gpt2/train.py ./gpt2/configs/molecule_nl/train_gpt2_medium_config.py' > \
    ${LEARNING_SOURCE_DIR}/logs/molecule_nl-train-medium-`date +%Y-%m-%d_%H-%M-%S`.log 2>&1 &