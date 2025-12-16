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
mkdir -p ${LEARNING_SOURCE_DIR}/molecule_nl/logs/
nohup bash -c 'python src/molecule_related_nl/dataset/prepare_gpt2.py assets/configs/molecules_nl.yaml' > \
    ${LEARNING_SOURCE_DIR}/molecule_nl/logs/molecule_nl-prepare-gpt2-`date +%Y-%m-%d_%H-%M-%S`.log 2>&1 &