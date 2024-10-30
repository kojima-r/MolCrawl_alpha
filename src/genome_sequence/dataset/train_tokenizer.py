"""https://github.com/MAGICS-LAB/DNABERT_2/issues/74"""

from typing import List
from pathlib import Path
from argparse import ArgumentParser

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

import numpy as np

from genome_sequence.utils.config import GenomeSequenceConfig


def read_file(file_path: str) -> List[str]:
    with open(file_path, "r") as file:
        return file.readlines()


def train_tokenizer(output_dir, vocab_size):
    path_dir = Path(output_dir) / "raw_files"

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        show_progress=True,
        vocab_size=vocab_size,
        min_frequency=2,
    )

    tokenizer.pre_tokenizer = Whitespace()

    files = [str(p) for p in path_dir.glob("*.raw")]
    np.random.seed(42)
    np.random.permutation(files)
    files = files[:35]
    tokenizer.train(files, trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    # tokenizer.train_from_iterator(line_iterator, trainer)
    tokenizer.save(str(Path(output_dir) / "tokenizer.json"))


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = GenomeSequenceConfig.from_file(args.config).data_preparation

    train_tokenizer(cfg.output_dir, cfg.vocab_size)


# import argparse
# from tokenizers import Tokenizer
# from tokenizers.models import BPE
# from tokenizers.trainers import BpeTrainer
# import os
# import glob
# import json
# from multiprocessing import Pool, cpu_count
# from pathlib import Path


# def main(args):
#     # paths = [str(x) for x in Path('/home/zhihan/dnabert_xl/splits').glob('**/*.fa')]
#     paths = ["/home/user/local-private-zhihan/data/DNABERT-2/tokenizer/train.txt"]
#     postfix = "_multi"

#     vocab_size = args.vocab_size
#     tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

#     tokenizer.pre_tokenizer = Whitespace()
#     tokenizer.train(paths, trainer)
#     tokenizer.post_processor = TemplateProcessing(
#         single="[CLS] $A [SEP]",
#         pair="[CLS] $A [SEP] $B:1 [SEP]:1",
#         special_tokens=[
#             ("[CLS]", tokenizer.token_to_id("[CLS]")),
#             ("[SEP]", tokenizer.token_to_id("[SEP]")),
#         ],
#     )

#     print("train finish")

#     tokenizer_dir = args.tokenizer_dir + str(vocab_size) + postfix
#     if not os.path.exists(tokenizer_dir):
#         os.makedirs(tokenizer_dir)
#     tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))

#     # generate and save tokenizer config
#     tokenizer_config = {"tokenizer_class": "PreTrainedTokenizerFast",
#                         "unk_token": "[UNK]",
#                         "cls_token": "[CLS]",
#                         "sep_token": "[SEP]",
#                         "pad_token": "[PAD]",
#                         "mask_token": "[MASK]"}
#     with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w") as f:
#         json.dump(tokenizer_config, f)

#     # generate and save model config
#     with open(os.path.join("data", "config.json"), "r") as f:
#         model_config = json.load(f)
#     model_config['vocab_size'] = vocab_size
#     with open(os.path.join(tokenizer_dir, "config.json"), "w") as f:
#         json.dump(model_config, f)

#     print("tokenizer saved")
