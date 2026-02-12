from argparse import ArgumentParser
import os
import sys
from pathlib import Path
from functools import partial
from typing import Dict, List, Optional

# プロジェクトルートのsrcディレクトリをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# データセットキャッシュ設定を読み込み（configs/cache.yamlから）
try:
    # 任意のキャッシュ設定。存在しない環境でも学習は継続できる。
    from utils.cache_config import setup_cache_env
except ModuleNotFoundError:
    setup_cache_env = None

if setup_cache_env is not None:
    setup_cache_env()
else:
    # cache_configが無い環境でも動作は可能
    print("WARNING: utils.cache_config not found. Continuing without cache setup.")

# from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict
import sentencepiece as spm

from genome_sequence.utils.config import GenomeSequenceConfig


def tokenize_function(examples: Dict[str, List[str]], tokenizer) -> Dict[str, List[int]]:
    return {"input_ids": tokenizer.encode(examples["text"]).ids}


def concatenate_texts(examples: Dict[str, List[List[int]]], eos_token_id: int) -> Dict[str, List[int]]:
    concatenated_ids: List[int] = []
    for input_ids in examples["input_ids"]:
        concatenated_ids.extend(input_ids + [eos_token_id])
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
    output_dir: str,
    context_length: int,
    number_sample: int,
    output_dataset_dir: Optional[str],
    num_proc: int,
) -> str:
    data = (
        load_dataset(
            "parquet",
            data_files=[str(Path(output_dir) / "parquet_files")],
            cache_dir=str(Path(output_dir) / "hf_cache"),
            split="train",
        )
        .shuffle()
        .select(range(number_sample))
    )

    tokenized_datasets = data.train_test_split(test_size=0.2)
    valid_test_split = tokenized_datasets["test"].train_test_split(test_size=0.5)
    tokenized_datasets = DatasetDict(
        {"train": tokenized_datasets["train"], "valid": valid_test_split["train"], "test": valid_test_split["test"]}
    )

    tokenizer = spm.SentencePieceProcessor(model_file=str(Path(output_dir) / "spm_tokenizer.model"))
    # tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

    concatenated_dataset = tokenized_datasets.map(
        partial(concatenate_texts, eos_token_id=tokenizer.eos_id()),
        # partial(concatenate_texts, eos_token_id=tokenizer.eos_token_id),
        batched=True,
        batch_size=-1,
        remove_columns=["num_tokens"],
        num_proc=num_proc,
    )

    chunked_dataset = concatenated_dataset.map(
        partial(create_chunks, context_length=context_length),
        batched=True,
        batch_size=context_length * 10,
        num_proc=num_proc,
    )

    # GPT-2用はBERTと分けて保存する
    default_dataset_dir: Path = Path(output_dir) / "training_ready_hf_dataset" / "gpt2"
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
    parser.add_argument(
        "--num_proc",
        type=int,
        default=8,
        help="datasets.map の並列プロセス数（デフォルト: 8）",
    )
    args = parser.parse_args()
    cfg = GenomeSequenceConfig.from_file(args.config).data_preparation

    tokenize_batch_dataset(cfg.output_dir, context_length, number_sample, args.output_dataset_dir, args.num_proc)
