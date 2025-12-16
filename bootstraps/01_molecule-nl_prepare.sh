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
scripts/preparation/download_smolinstruct.sh
nohup python scripts/preparation/preparation_script_molecule_related_nat_lang.py assets/configs/molecules_nl.yaml\
> ${LEARNING_SOURCE_DIR}/molecule_nl/logs/molecule_related_nat_lang-preparation-$(date +%Y-%m-%d_%H-%M-%S).log 2>&1 &