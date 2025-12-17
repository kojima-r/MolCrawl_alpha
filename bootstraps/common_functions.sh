#!/bin/bash
# Common functions for bootstrap scripts

# Check if LEARNING_SOURCE_DIR environment variable is set
# Usage: check_learning_source_dir
check_learning_source_dir() {
    if [ -z "$LEARNING_SOURCE_DIR" ]; then
        echo "ERROR: LEARNING_SOURCE_DIR environment variable is not set."
        echo "Please set it before running this script:"
        echo "  export LEARNING_SOURCE_DIR='...'"
        exit 1
    fi
    echo "DatabaseDir: $LEARNING_SOURCE_DIR"
}
