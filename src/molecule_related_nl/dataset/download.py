import os

from pathlib import Path
from datasets import load_dataset


def download_hf_dataset(save_path_parent):
    
    os.path.exists(save_path_parent) or os.makedirs(save_path_parent)

    data = load_dataset('osunlp/SMolInstruct', trust_remote_code=True)

    save_path = Path(save_path_parent) / Path('osunlp/SMolInstruct')

    for split in data.keys():
        data[split].save_to_disk(str(save_path / Path(split)))


if __name__ == "__main__":
    download_hf_dataset("src/molecule_related_nl/assets/raw_data")
