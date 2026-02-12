from functools import partial
from argparse import ArgumentParser
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# プロジェクトルートのsrcディレクトリをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# データセットキャッシュ設定を読み込み（configs/cache.yamlから）
from utils.cache_config import setup_cache_env
setup_cache_env()

from datasets import DatasetDict
import numpy as np

from molecule_related_nl.utils.tokenizer import MoleculeNatLangTokenizer
from molecule_related_nl.utils.config import MoleculeNLConfig
from molecule_related_nl.utils.general import read_dataset


def concatenate_texts(examples: Dict[str, List[List[int]]], eos_token_id: int) -> Dict[str, List[int]]:
    concatenated_ids: List[int] = []
    for input_ids, output_ids in zip(examples["input_ids"], examples["output_ids"]):
        concatenated_ids.extend(input_ids + output_ids)
    return {"input_ids": concatenated_ids}


def create_chunks(examples: Dict[str, List[int]], context_length: int) -> Dict[str, List[List[int]]]:
    concatenated_ids: List[int] = examples["input_ids"]

    # Calculate the total number of chunks
    total_length = len(concatenated_ids)
    num_chunks = total_length // context_length

    # Truncate the concatenated_ids to a multiple of context_length
    total_length = num_chunks * context_length
    concatenated_ids = concatenated_ids[:total_length]

    # Split into chunks
    input_ids: List[List[int]] = [
        concatenated_ids[i : i + context_length] for i in range(0, total_length, context_length)
    ]

    return {"input_ids": input_ids}


def tokenize_batch_dataset(
    parquet_path: str,
    context_length: int,
    number_sample: int,
    output_dataset_dir: Optional[str],
) -> str:

    tokenize_dataset = DatasetDict(read_dataset(parquet_path))
    
    # Handle validation/valid split naming
    if "validation" in tokenize_dataset and "valid" not in tokenize_dataset:
        tokenize_dataset["valid"] = tokenize_dataset["validation"]
        del tokenize_dataset["validation"]
    elif "valid" not in tokenize_dataset and "validation" not in tokenize_dataset:
        raise KeyError("Neither 'valid' nor 'validation' split found in dataset")

    tokenize_dataset["train"] = tokenize_dataset["train"].select(
        np.random.choice(len(tokenize_dataset["train"]), int(number_sample * 0.8), replace=False)
    )
    tokenize_dataset["valid"] = tokenize_dataset["valid"].select(
        np.random.choice(len(tokenize_dataset["valid"]), int(number_sample * 0.1), replace=False)
    )
    tokenize_dataset["test"] = tokenize_dataset["test"].select(
        np.random.choice(len(tokenize_dataset["test"]), int(number_sample * 0.1), replace=False)
    )

    tokenizer = MoleculeNatLangTokenizer()

    concatenated_dataset = tokenize_dataset.map(
        partial(concatenate_texts, eos_token_id=tokenizer.tokenizer.eos_token_id),
        batched=True,
        batch_size=-1,
        remove_columns=tokenize_dataset["train"].column_names,
    )

    chunked_dataset = concatenated_dataset.map(
        partial(create_chunks, context_length=context_length),
        batched=True,
        batch_size=-1,
    )

    # GPT-2用はBERTと分けて保存する
    default_dataset_dir: Path = Path(parquet_path).parent / "training_ready_hf_dataset" / "gpt2"
    dataset_dir: Path = Path(output_dataset_dir) if output_dataset_dir else default_dataset_dir
    path_dataset = str(dataset_dir)
    print(f"Saving dataset to: {path_dataset}. Match this path to the train_gpt2_config.py->dataset_dir parameter.")
    chunked_dataset.save_to_disk(path_dataset)
    return path_dataset


if __name__ == "__main__":
    number_sample: int = 50000
    context_length: int = 1024

    parser = ArgumentParser()
    parser.add_argument("config")
    parser.add_argument(
        "--output_dataset_dir",
        type=str,
        default=None,
        help="GPT-2用の出力ディレクトリ（未指定なら output_dir/training_ready_hf_dataset/gpt2）",
    )
    args = parser.parse_args()
    cfg = MoleculeNLConfig.from_file(args.config).data_preparation
    
    # 相対パスを絶対パスに変換
    from config.paths import PROJECT_ROOT, LEARNING_SOURCE_DIR
    save_path = os.path.join(PROJECT_ROOT, LEARNING_SOURCE_DIR, cfg.save_path)

    tokenize_batch_dataset(save_path, context_length, number_sample, args.output_dataset_dir)
