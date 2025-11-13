import os
import logging
from datasets import Dataset, DatasetDict
from pathlib import Path

logger = logging.getLogger(__name__)


def read_dataset(dataset_path: str):
    """
    Read dataset from disk, supporting both split directories and DatasetDict format

    Args:
        dataset_path: Path to the dataset directory

    Returns:
        dict or DatasetDict: Dictionary of splits with Dataset objects
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    # Try to load as DatasetDict first (if it was saved with save_to_disk)
    try:
        logger.info(f"Attempting to load dataset as DatasetDict from {dataset_path}")
        dataset_dict = DatasetDict.load_from_disk(str(dataset_path))
        logger.info(f"Successfully loaded DatasetDict with splits: {list(dataset_dict.keys())}")
        return dataset_dict
    except Exception as e:
        logger.debug(f"Not a DatasetDict format: {e}")

    # Fall back to loading individual split directories
    splits = {}
    try:
        for folder in os.listdir(dataset_path):
            folder_path = dataset_path / folder
            if folder_path.is_dir():
                # Skip cache and metadata directories
                if folder.startswith(".") or folder == "hf_cache":
                    continue
                try:
                    logger.info(f"Loading split: {folder}")
                    splits[folder] = Dataset.load_from_disk(str(folder_path))
                    logger.info(f"Loaded {folder} with {len(splits[folder])} samples")
                except Exception as split_error:
                    logger.warning(f"Failed to load split {folder}: {split_error}")

        if not splits:
            raise ValueError(f"No valid dataset splits found in {dataset_path}")

        return splits
    except Exception as e:
        logger.error(f"Failed to read dataset from {dataset_path}: {e}")
        raise


def save_dataset(dataset, dataset_path: str):
    """
    Save dataset to disk

    Args:
        dataset: Dictionary of Dataset objects or DatasetDict
        dataset_path: Path to save the dataset
    """
    dataset_path = Path(dataset_path)
    os.makedirs(dataset_path, exist_ok=True)

    logger.info(f"Saving dataset to {dataset_path}")

    # If it's a DatasetDict, we can save it directly
    if isinstance(dataset, DatasetDict):
        dataset.save_to_disk(str(dataset_path))
        logger.info(f"Saved DatasetDict with {len(dataset)} splits")
        return

    # Otherwise, save each split separately
    for split in dataset.keys():
        split_path = dataset_path / split
        logger.info(f"Saving {split} split to {split_path}")
        dataset[split].save_to_disk(str(split_path))
        logger.info(f"Saved {split} with {len(dataset[split])} samples")
