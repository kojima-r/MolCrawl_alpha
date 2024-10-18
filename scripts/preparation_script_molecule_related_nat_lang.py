from argparse import ArgumentParser
import os
import logging
import logging.config

from pathlib import Path

from utils.base import setup_logging
from molecule_related_nl.dataset.download import download_hf_dataset
from molecule_related_nl.utils.config import MoleculeNLConfig
from molecule_related_nl.utils.general import read_dataset, count_number_of_tokens, save_dataset

from molecule_related_nl.utils.tokenizer import MoleculeNatLangTokenizer


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = MoleculeNLConfig.from_file(args.config).data_preparation
    
    logging_dir = Path(cfg.save_path).parent / "molecule_related_natural_language_logs"
    os.path.exists(logging_dir) or os.makedirs(logging_dir)
    setup_logging(logging_dir)

    os.path.exists(cfg.dataset) or os.makedirs(cfg.dataset)

    logger.info(msg="Downloading Dataset...")
    download_hf_dataset(cfg.dataset, cfg.hf_path)

    dataset = read_dataset(cfg.dataset)

    tokenizer = MoleculeNatLangTokenizer()

    logger.info(msg="Tokenizing Scaffolds...")
    
    token_counts = {}
    for split in dataset.keys():
        dataset[split] = dataset[split].map(tokenizer.tokenize_dict)
        token_counts[split] = count_number_of_tokens(dataset[split])


    logger.info(msg="Dataset Statistics:")
    for split in token_counts.keys():
        logger.info(msg=f"{split}: {token_counts[split]}")
        logger.info(msg=f"Number of examples: {len(dataset[split])}")

    logger.info(msg="Total number of tokens: {}".format(sum(token_counts.values())))
    logger.info(msg="Total number of examples: {}".format(sum([len(dataset[split]) for split in dataset.keys()])))

    logger.info(msg="Saving processed dataset to {}.".format(cfg.save_path))
    save_dataset(dataset, cfg.save_path)
