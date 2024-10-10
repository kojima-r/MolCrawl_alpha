from argparse import ArgumentParser
import os
import json
import logging

from pathlib import Path

from compounds.dataset.organix13.opv.prepare_opv import OPV
from compounds.dataset.organix13.download import download_datasets_from_repo

from utils.base import read_parquet, save_parquet, multiprocess_tokenization
from compounds.utils.tokenizer import CompoundsTokenizer, ScaffoldsTokenizer
from compounds.utils.config import CompoundConfig
from compounds.dataset.organix13.zinc.download_and_convert_to_parquet import download_zinc_files, convert_zinc_to_parquet
from compounds.dataset.organix13.combine_all import combine_all


logger = logging.getLogger(__name__)


def setup_logging(output_dir: str, logging_config: str = "assets/logging_config.json"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(logging_config, "r") as file:
        config = json.load(file)
    logging_file = f"{output_dir}/logging.log"
    config["handlers"]["file"]["filename"] = logging_file
    if os.path.exists(logging_file):
        os.remove(logging_file)
    logging.config.dictConfig(config=config)

def download_datasets(raw_data_dir: str, output_dir: str):
    download_zinc_files()
    convert_zinc_to_parquet(raw_data_dir)
    OPV(raw_data_dir)
    download_datasets_from_repo(raw_data_dir)
    combine_all(raw_data_dir, output_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = CompoundConfig.from_file(args.config).data_preparation
    
    setup_logging(Path(cfg.save_path).parent)

    try:
        # os.path.exists(cfg.raw_data_path) or os.makedirs(cfg.raw_data_path)
        # download_datasets(cfg.raw_data_path, cfg.organix13_dataset)

        organix13_dataset = read_parquet(
            file_path=os.path.join(cfg.organix13_dataset, "OrganiX13.parquet")
        )

        mol_tokenizer = CompoundsTokenizer(
            cfg.vocab_path,
            cfg.max_length,
        )

        logger.info(msg="Tokenizing SMILES...")
        processed_organix13 = multiprocess_tokenization(mol_tokenizer.bulk_tokenizer_parquet, organix13_dataset, column_name="smiles", new_column_name="tokens")

        scaffolds_tokenizer = ScaffoldsTokenizer(
            cfg.vocab_path,
            cfg.max_length,
        )

        logger.info(msg="Tokenizing Scaffolds...")
        processed_organix13 = scaffolds_tokenizer.bulk_tokenizer_parquet(organix13_dataset, column_name="smiles", new_column_name="scaffold_tokens")
        
        logger.info(msg="Tokenizing done.")

        logger.info(
            msg="Saving processed dataset to {}.".format(cfg.save_path)
        )

        save_parquet(
            table=processed_organix13,
            file_path=cfg.save_path
        )

    except Exception as exception_handle:
        logger.error(
            msg=exception_handle
        )

        raise
