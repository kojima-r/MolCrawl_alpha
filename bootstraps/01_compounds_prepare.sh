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
mkdir -p ${LEARNING_SOURCE_DIR}/compounds/logs/
nohup python scripts/preparation/preparation_script_compounds.py assets/configs/compounds.yaml --force \
> ${LEARNING_SOURCE_DIR}/compounds/logs/compounds-preparation-$(date +%Y-%m-%d_%H-%M-%S).log 2>&1 &