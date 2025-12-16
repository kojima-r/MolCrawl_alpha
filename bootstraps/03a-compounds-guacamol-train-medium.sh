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
mkdir -p ${LEARNING_SOURCE_DIR}/compounds/logs
nohup bash -c 'python gpt2/train.py gpt2/configs/compounds/train_gpt2_medium_config.py' > \
    ${LEARNING_SOURCE_DIR}/compounds/logs/compounds-train-medium-`date +%Y-%m-%d_%H-%M-%S`.log 2>&1 &