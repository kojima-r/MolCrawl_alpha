from argparse import ArgumentParser, Namespace
from logging import Formatter, Logger, StreamHandler, getLogger

from src.utilities.common import read_parquet, save_parquet
from src.utilities.molecules_utilities import MoleculesTokenizer


def get_script_arguments() -> Namespace:
    """
    Get the script arguments.

    :returns: The script arguments.
    """

    argument_parser = ArgumentParser()

    argument_parser.add_argument(
        "-o13",
        "--organix13-dataset",
        type=str,
        required=True,
        help="Path to the root folder of the organix13 dataset."
    )

    argument_parser.add_argument(
        "-sp",
        "--save-path",
        type=str,
        required=True,
        help="Path to save the processed and tokenized dataset."
    )

    argument_parser.add_argument(
        "-vp",
        "--vocab-path",
        type=str,
        required=True,
        help="Path to the vocabulary."
    )
    
    argument_parser.add_argument(
        "-ml",
        "--max-length",
        type=str,
        required=False,
        help="Max length of the tokenized sequences.",
        default=256
    )

    return argument_parser.parse_args()


def get_script_logger() -> Logger:
    """
    Get the script logger.

    :returns: The script logger.
    """

    logger = getLogger(
        name="organix13_processing"
    )

    logger.setLevel(
        level="INFO"
    )

    formatter = Formatter(
        fmt="[{asctime:s}] {levelname:s}: \"{message:s}\"",
        style="{"
    )

    stream_handler = StreamHandler()

    stream_handler.setLevel(
        level="INFO"
    )

    stream_handler.setFormatter(
        fmt=formatter
    )

    logger.addHandler(
        hdlr=stream_handler
    )

    return logger


def save_dataset(dataset, file_path):
    dataset.to_parquet(
        file_path
    )

if __name__ == "__main__":
    script_logger = get_script_logger()

    try:
        script_arguments = get_script_arguments()

        organix13_dataset = read_parquet(
            file_path=script_arguments.organix13_dataset
        )

        tokenizer = MoleculesTokenizer(
            script_arguments.vocab_path,
            script_arguments.max_length,
            script_logger
        )

        processed_organix13 = tokenizer.bulk_tokenizer_parquet(organix13_dataset, "smiles")
        
        script_logger.info(
            msg="Saving processed dataset to {}.".format(script_arguments.save_path)
        )

        save_parquet(
            table=processed_organix13,
            file_path=script_arguments.save_path
        )

    except Exception as exception_handle:
        script_logger.error(
            msg=exception_handle
        )

        raise
