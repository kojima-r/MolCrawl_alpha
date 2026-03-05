"""
Sample from a trained model
"""

import os
from contextlib import nullcontext

import numpy as np
import torch
from molcrawl.gpt2.model import GPT, GPTConfig


def cut_after_eos(seq, bos_id, eos_id):
    """Returns from the first occurrence of BOS up to the first EOS. If there is no EOS, to the end."""
    # seq assumes a list of ints
    try:
        start = seq.index(bos_id)
    except ValueError:
        start = 0
    if eos_id is None:
        return seq[start:]
    for i in range(start + 1, len(seq)):
        if seq[i] == eos_id:
            return seq[start : i + 1]
    return seq[start:]


# Special Tokens
start_instruction = None
end_instruction = None
eos_token = None

dataset_params: dict[str, object] = {}
# -----------------------------------------------------------------------------
init_from = "resume"  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = "out"  # ignored if init_from is not 'resume'
start = "\n"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
seed = 1337
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = (
    "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
)  # 'float32' or 'bfloat16' or 'float16'
compile = False  # use PyTorch 2.0 to compile the model to be faster
exec(open("gpt2/configurator.py").read())  # overrides from command line or config file
tokenizer, batch_size = None, 1
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model

# init from a model saved in a specific directory
ckpt_path = os.path.join(out_dir, "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint["model_args"])
model = GPT(gptconf)
state_dict = checkpoint["model"]
unwanted_prefix = "_orig_mod."
for k, _ in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)

num_samples = 100000
max_new_tokens = 128  # Maximum token length per molecule
temperature = 1.0
top_k = None

# 1) Determine BOS/EOS/PAD (The golden rule is to use separate IDs for BOS and EOS)
# If you are already using start_instruction/eos_token, give it priority
bos_id = start_instruction

# PAD is often required for generate. If undefined, assign either BOS/EOS
pad_id = getattr(tokenizer, "pad_token_id", None)
if pad_id is None:
    pad_id = eos_token if eos_token is not None else bos_id

# 2) BOS-only input for batch ([B, 1])
# start_ids = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)

# {'C': 942539, 'O': 213055, 'N': 80235, 'c': 13219, 'F': 9771, '[O-]': 725, 'S': 2445, 'I': 197, 'Cl': 7474, 'Br': 1825, '[H]': 302, '[C-]': 536, '[N-]': 679, 'B': 52, '[NH3+]': 16, '[Se]': 4, '[BH3-]': 9, '[F-]': 1, 'n': 2, '[N]': 1, '[S-]': 2, '[Cl-]': 1, '[CH]': 2, '[SeH]': 5, '[SeH2]': 1, '[B]': 2, '[CH2]': 1, '[Br-]': 1, '[NH2+]': 1, '[NH-]': 1}
first_token_freq = {
    16: 942539,
    19: 213055,
    23: 80235,
    15: 13219,
    27: 9771,
    36: 725,
    34: 2445,
    48: 197,
    28: 7474,
    37: 1825,
    63: 302,
    86: 536,
    61: 679,
    54: 52,
    107: 16,
    126: 4,
    92: 9,
    89: 1,
    25: 2,
    177: 1,
    112: 2,
    57: 1,
    83: 2,
    249: 5,
    476: 1,
    167: 2,
    158: 1,
    85: 1,
    116: 1,
    161: 1,
}

ids = np.fromiter(first_token_freq.keys(), dtype=np.int64)
counts = np.fromiter(first_token_freq.values(), dtype=np.float64)
p = counts / counts.sum()  # Normalization
p /= p.sum()  # Set it to 1 again just to be sure


smiles_list: list[str] = []
while len(smiles_list) < num_samples:
    # --- 2) Bulk sampling of first tokens
    # second_ids = rng.choice(ids, size=batch_size, p=p) # Use rng if reproducibility is required
    second_ids = np.random.choice(ids, size=batch_size, p=p)

    start_ids = torch.tensor(
        np.column_stack([np.full(batch_size, bos_id, dtype=np.int64), second_ids]),
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad(), ctx:
        out = model.generate(
            start_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_token,
            pad_token_id=pad_id,
        )
        # print(out) # for debugging

    for seq in out.tolist():
        gen_ids = cut_after_eos(seq, bos_id, eos_token)
        s = tokenizer.decode(gen_ids, skip_special_tokens=True)  # type: ignore[attr-defined]
        # If the tokenizer outputs spaces separated, use the following line
        s = s.replace(" ", "")
        smiles_list.append(s)
        if len(smiles_list) >= num_samples:
            break

    if len(smiles_list) / 256 % 10 == 0:
        print(f"{len(smiles_list)}/{num_samples} samples generated")

with open(f"{out_dir}/generated_compounds.txt", "w") as f:
    for s in smiles_list:
        f.write(f"{s}\n")
