import os
from datasets import Dataset

def read_dataset(dataset_path: str):
    splits = {}
    for folder in os.listdir(dataset_path):
        splits[folder] = Dataset.load_from_disk(os.path.join(dataset_path, folder))

    return splits

def count_number_of_tokens(dataset):
    number_of_tokens = 0

    def internal_count(x):
        nonlocal number_of_tokens
        number_of_tokens += len(x["input_ids"]) + len(x["output_ids"])
        return x
        
    dataset.map(internal_count)

    return number_of_tokens

def save_dataset(dataset, dataset_path: str):
    for split in dataset.keys():
        dataset[split].save_to_disk(os.path.join(dataset_path, split))
