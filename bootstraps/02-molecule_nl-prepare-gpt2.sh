#!/bin/bash
echo "DatabaseDir: $LEARNING_SOURCE_DIR"
mkdir -p logs
nohup bash -c 'python src/molecule_related_nl/dataset/prepare_gpt2.py assets/configs/molecules_nl.yaml' > \
    logs/molecule_nl-prepare-gpt2-`date +%Y-%m-%d_%H-%M-%S`.log 2>&1 &
