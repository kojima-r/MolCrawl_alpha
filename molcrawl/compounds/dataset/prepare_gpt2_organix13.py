"""
GPT-2 training dataset preparation script (all compound dataset integrated version)

Supports v2 dataset configuration and from separate tokenized parquet files.
Convert to HuggingFace Dataset format for GPT-2 training.

Output path: {compounds_dir}/organix13/compounds/training_ready_hf_dataset
- GPT-2 training configuration (train_gpt2_config.py) references this path.

How to use:
  LEARNING_SOURCE_DIR=learning_source_YYYYMMDD python src/compounds/dataset/prepare_gpt2_organix13.py assets/configs/compounds.yaml
"""

import os
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from molcrawl.compounds.dataset.dataset_config import (
    DATASET_DEFINITIONS,
    CompoundDatasetType,
)
from molcrawl.compounds.utils.config import CompoundConfig
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


def prepare_gpt2_dataset(compounds_dir: str):
    """
    Integrate v2's individual tokenized parquet and create a GPT-2 training dataset.

    Processing details:
    1. Load parquet of all datasets from tokenized/ directory
    2. Extract and integrate the tokens column
    3. Rename tokens → input_ids
    4. Split into train/valid/test (80/10/10)
    5. Save in HuggingFace Dataset format

        Args:
    compounds_dir: compounds directorypath of
    """
    compounds_path = Path(compounds_dir)

    # Target all datasets excluding GuacaMol which is not for GPT-2
    target_datasets = [dt for dt in DATASET_DEFINITIONS.keys() if dt != CompoundDatasetType.GUACAMOL]

    # Collect tokenized parquet
    all_tokens = []
    loaded_datasets = []

    for dataset_type in target_datasets:
        info = DATASET_DEFINITIONS[dataset_type]
        tokenized_path = info.get_tokenized_path(compounds_path)

        if not tokenized_path.exists():
            print(f"  ⚠ {info.name}: Tokenized data not found at {tokenized_path}, skipping")
            continue

        print(f"  Loading {info.name} from {tokenized_path}...")
        table = pq.read_table(tokenized_path, columns=["tokens"])
        df = table.to_pandas()
        print(f"    → {len(df)} samples")

        all_tokens.append(df)
        loaded_datasets.append(info.name)

    if not all_tokens:
        raise FileNotFoundError(
            f"No tokenized datasets found in {compounds_path / 'tokenized'}\n\n"
            f"Please run the preparation script first:\n"
            f"  LEARNING_SOURCE_DIR={os.environ.get('LEARNING_SOURCE_DIR', 'learning_source_YYYYMMDD')} "
            f"python src/preparation/preparation_script_compounds.py assets/configs/compounds.yaml"
        )

    # Integration
    print(f"\nCombining {len(loaded_datasets)} datasets: {loaded_datasets}")
    combined_df = pd.concat(all_tokens, ignore_index=True)
    print(f"Total combined samples: {len(combined_df)}")

    # Rename tokens → input_ids (because GPT-2 learning expects input_ids)
    combined_df = combined_df.rename(columns={"tokens": "input_ids"})

    # Split into train/valid/test (80/10/10)
    train_df, temp_df = train_test_split(combined_df, test_size=0.2, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"Split: Train={len(train_df)}, Valid={len(valid_df)}, Test={len(test_df)}")

    # Create HuggingFace Dataset
    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, preserve_index=False),
            "valid": Dataset.from_pandas(valid_df, preserve_index=False),
            "test": Dataset.from_pandas(test_df, preserve_index=False),
        }
    )

    # Save to the legacy output path for backward compatibility with GPT-2 training configs
    # (train_gpt2_config.py references: compounds/organix13/compounds/training_ready_hf_dataset)
    output_path = compounds_path / "organix13" / "compounds" / "training_ready_hf_dataset"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving dataset to: {output_path}")
    dataset.save_to_disk(str(output_path))

    # Print statistics
    print("\nDataset statistics:")
    for split in ["train", "valid", "test"]:
        print(f"  {split}: {len(dataset[split])} samples")
    print("\nThis path matches train_gpt2_config.py → dataset_dir parameter.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Prepare GPT-2 training dataset from tokenized compounds data")
    parser.add_argument(
        "config",
        help="Path to compounds config file (e.g. assets/configs/compounds.yaml)",
    )
    args = parser.parse_args()

    # config is received for compatibility, but v2 uses tokenized data directly
    # vocab_path / max_length is not required
    _ = CompoundConfig.from_file(args.config)

    # Get compounds directory from LEARNING_SOURCE_DIR
    learning_source_dir = os.environ.get("LEARNING_SOURCE_DIR")
    if not learning_source_dir:
        raise ValueError(
            "LEARNING_SOURCE_DIR environment variable is not set.\n"
            "Please set it before running this script:\n"
            "  export LEARNING_SOURCE_DIR='learning_source_YYYYMMDD'"
        )

    compounds_dir = Path(learning_source_dir) / "compounds"
    print(f"Using compounds directory: {compounds_dir}")

    prepare_gpt2_dataset(str(compounds_dir))
