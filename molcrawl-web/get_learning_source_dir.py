#!/usr/bin/env python3
"""
Script to retrieve LEARNING_SOURCE_DIR from environment variable and output as JSON.
Self-contained with no external package dependencies.
Requires the LEARNING_SOURCE_DIR environment variable.
"""

import json
import os
import sys

try:
    learning_source_dir = os.environ.get("LEARNING_SOURCE_DIR")
    if not learning_source_dir:
        print(
            json.dumps({"error": "LEARNING_SOURCE_DIR environment variable is not set"}),
            file=sys.stderr,
        )
        sys.exit(1)

    # This script is located directly under molcrawl-web/, so one level up is the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    absolute_path = os.path.join(project_root, learning_source_dir)

    result = {
        "learning_source_dir": learning_source_dir,
        "project_root": project_root,
        "absolute_path": absolute_path,
    }

    print(json.dumps(result))

except Exception as e:
    print(json.dumps({"error": f"Unexpected error: {e}"}), file=sys.stderr)
    sys.exit(1)
