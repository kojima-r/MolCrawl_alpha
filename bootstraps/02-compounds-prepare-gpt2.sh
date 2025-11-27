#!/bin/bash
echo "DatabaseDir: $LEARNING_SOURCE_DIR"
mkdir -p logs
nohup bash -c 'python src/compounds/dataset/prepare_gpt2.py assets/configs/compounds.yaml' > \
    logs/compounds-prepare-gpt2-`date +%Y-%m-%d_%H-%M-%S`.log 2>&1 &
