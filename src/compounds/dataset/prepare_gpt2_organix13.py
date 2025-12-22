from argparse import ArgumentParser
import os
import sys
from pathlib import Path

# プロジェクトルートのsrcディレクトリをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
from compounds.utils.config import CompoundConfig
from compounds.utils.tokenizer import CompoundsTokenizer
from datasets import Dataset, DatasetDict


def tokenize_batch_dataset(compounds_dir, vocab_path, max_length):
    """
    Tokenize OrganiX13 parquet data for GPT-2 training.
    
    Args:
        compounds_dir: Base directory for compounds data (from LEARNING_SOURCE_DIR)
        vocab_path: Path to vocabulary file
        max_length: Maximum token length
    """
    tokenizer = CompoundsTokenizer(
        vocab_path,
        max_length,
    )

    # OrganiX13 parquet files
    organix13_dir = Path(compounds_dir) / "organix13"
    parquet_dir = Path(compounds_dir) / "parquet_files"
    
    # Check if parquet files exist
    if not parquet_dir.exists():
        raise FileNotFoundError(
            f"Parquet files directory not found: {parquet_dir}\n\n"
            f"Please run the preparation script first:\n"
            f"  LEARNING_SOURCE_DIR={os.environ.get('LEARNING_SOURCE_DIR', 'learning_20251210')} "
            f"python scripts/preparation/preparation_script_compounds.py assets/configs/compounds.yaml"
        )

    dataset_dic = {}
    for split in ["train", "valid", "test"]:
        parquet_file = parquet_dir / f"{split}.parquet"
        
        if not parquet_file.exists():
            raise FileNotFoundError(
                f"Parquet file not found: {parquet_file}\n"
                f"Expected location: {parquet_dir}/{split}.parquet"
            )
        
        print(f"Loading {split} data from: {parquet_file}")
        df = pd.read_parquet(parquet_file)
        
        # Tokenize SMILES
        if "smiles" not in df.columns:
            raise ValueError(f"'smiles' column not found in {parquet_file}")
        
        print(f"Tokenizing {len(df)} SMILES for {split} split...")
        df["tokens"] = df["smiles"].apply(tokenizer.tokenize_text)
        print(f"Example decoded: {tokenizer.decode(df['tokens'].iloc[0])}")
        dataset_dic[split] = df

    # Create HuggingFace Dataset
    d = {
        "train": Dataset.from_dict(
            {"input_ids": dataset_dic["train"]["tokens"].to_numpy()}
        ),
        "valid": Dataset.from_dict(
            {"input_ids": dataset_dic["valid"]["tokens"].to_numpy()}
        ),
        "test": Dataset.from_dict(
            {"input_ids": dataset_dic["test"]["tokens"].to_numpy()}
        ),
    }

    dataset = DatasetDict(d)

    # Save to compounds directory structure (OrganiX13 specific path)
    output_path = organix13_dir / "compounds" / "training_ready_hf_dataset"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving dataset to: {output_path}")
    print(f"Match this path to the train_gpt2_config.py->dataset_dir parameter.")
    dataset.save_to_disk(str(output_path))
    
    # Print statistics
    print("\nDataset statistics:")
    for split in ["train", "valid", "test"]:
        print(f"  {split}: {len(dataset[split])} samples")


if __name__ == "__main__":
    number_sample = None

    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = CompoundConfig.from_file(args.config).data_preparation
    context_length = cfg.max_length

    # Get compounds directory from LEARNING_SOURCE_DIR
    learning_source_dir = os.environ.get("LEARNING_SOURCE_DIR")
    if not learning_source_dir:
        raise ValueError(
            "LEARNING_SOURCE_DIR environment variable is not set.\n"
            "Please set it before running this script:\n"
            "  export LEARNING_SOURCE_DIR='learning_20251210'"
        )
    
    compounds_dir = Path(learning_source_dir) / "compounds"
    print(f"Using compounds directory: {compounds_dir}")

    tokenize_batch_dataset(str(compounds_dir), cfg.vocab_path, cfg.max_length)

