from argparse import ArgumentParser
import os
import matplotlib.pyplot as plt
import numpy as np

import logging
import logging.config

from pathlib import Path

from core.base import read_parquet, save_parquet, multiprocess_tokenization, setup_logging
from compounds.utils.tokenizer import CompoundsTokenizer, ScaffoldsTokenizer
from compounds.utils.config import CompoundConfig
from compounds.utils.general import download_datasets

from config.paths import COMPOUNDS_DIR

logger = logging.getLogger(__name__)

def run_statistics(table_row, column_name):
    series_length = []
    for i in table_row:
        if i.is_valid:
            series_length.append(len(i))

    plt.hist(series_length, bins=np.arange(0, 200, 1))
    plt.xlabel("Length of tokenized {}".format(column_name))
    plt.title("Distribution of tokenized {} lengths".format(column_name))
    plt.savefig("assets/img/compounds_tokenized_{}_lengths_dist.png".format(column_name))
    plt.close()
    logger.info(msg="Saved distribution of tokenized {} lengths to assets/img/compounds_tokenized_{}_lengths_dist.png".format(column_name, column_name))

    return {
        "Number of Samples for {}".format(column_name): len(series_length),
        "Number of Tokens for {}".format(column_name): sum(series_length),
    }



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--force", action="store_true", help="Force re-download and reprocessing even if files exist")
    args = parser.parse_args()
    cfg = CompoundConfig.from_file(args.config).data_preparation
    organix13_dataset_path = COMPOUNDS_DIR + "/organix13"
    os.path.exists(organix13_dataset_path) or os.makedirs(organix13_dataset_path)

    setup_logging(COMPOUNDS_DIR + "/compounds_logs")

    # マーカーファイル・出力ファイル
    download_marker = Path(organix13_dataset_path) / "download_complete.marker"
    tokenized_marker = Path(organix13_dataset_path) / "tokenized_complete.marker"
    stats_marker = Path(organix13_dataset_path) / "stats_complete.marker"
    processed_parquet = Path(organix13_dataset_path) / "OrganiX13_tokenized.parquet"

    # 1. データダウンロード
    if not args.force and download_marker.exists():
        logger.info("Dataset download already completed. Skipping download step.")
    else:
        logger.info("Downloading datasets...")
        os.path.exists(cfg.raw_data_path) or os.makedirs(cfg.raw_data_path)
        download_datasets(cfg.raw_data_path, organix13_dataset_path)
        download_marker.touch()
        logger.info("Download completed.")

    # 2. トークナイズ
    if not args.force and tokenized_marker.exists() and processed_parquet.exists():
        logger.info("Tokenization already completed. Skipping tokenization step.")
        organix13_dataset = read_parquet(file_path=str(processed_parquet))
    else:
        organix13_dataset = read_parquet(file_path=os.path.join(organix13_dataset_path, "OrganiX13.parquet"))
        mol_tokenizer = CompoundsTokenizer(
            cfg.vocab_path,
            cfg.max_length,
        )
        logger.info(msg="Tokenizing SMILES...")
        processed_organix13 = multiprocess_tokenization(
            mol_tokenizer.bulk_tokenizer_parquet, organix13_dataset, column_name="smiles", new_column_name="tokens", processes=2
        )
        scaffolds_tokenizer = ScaffoldsTokenizer(
            cfg.vocab_path,
            cfg.max_length,
        )
        logger.info(msg="Tokenizing Scaffolds...")
        processed_organix13 = multiprocess_tokenization(
            scaffolds_tokenizer.bulk_tokenizer_parquet,
            processed_organix13,
            column_name="smiles",
            new_column_name="scaffold_tokens",
        )
        logger.info(msg="Tokenizing done.")
        save_parquet(table=processed_organix13, file_path=processed_parquet)
        tokenized_marker.touch()
        organix13_dataset = processed_organix13

    # 3. 統計処理
    if not args.force and stats_marker.exists():
        logger.info("Statistics already computed. Skipping statistics step.")
    else:
        logger.info(msg="Computing Statistics...")
        statistics = {
            **run_statistics(organix13_dataset["tokens"], "SMILES"),
            **run_statistics(organix13_dataset["scaffold_tokens"], "Scaffolds")
        }
        for key, value in statistics.items():
            logger.info(msg="{}: {}".format(key, value))
        stats_marker.touch()

    logger.info(msg="Saving processed dataset to {}.".format(COMPOUNDS_DIR))
    # 保存はtokenized_marker作成時に実施済み
