from datasets import load_from_disk
import torch


class ProteinSequenceDataset:

    def __init__(self, dataset_dir, split):
        super().__init__()
        self.data = load_from_disk(dataset_dir)[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]["input_ids"])
