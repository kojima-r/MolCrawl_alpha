#!/usr/bin/env python3
"""
Script to download and test models from Hugging Face Hub

This script provides the following functionality:
1. Download the model from Hugging Face Hub
2. Confirm checkpoint loading
3. Verification of model structure
4. Simple reasoning test
5. Text generation test (optional)

How to use:
    python test_huggingface_download.py <repo_id> [options]

example:
    python test_huggingface_download.py deskull/rna-small-gpt2 --test-generate
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# add project path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent


try:
    import torch
except ImportError:
    print("[ERROR] PyTorch is not installed")
    print("Installation: pip install torch")
    sys.exit(1)

try:
    from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files
except ImportError:
    print("[ERROR] huggingface_hub is not installed")
    print("Installation: pip install huggingface_hub")
    sys.exit(1)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Download and test models from Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
example:
  # Basic test (download and loading confirmation)
  python test_huggingface_download.py deskull/rna-small-gpt2

  # with generation tests
  python test_huggingface_download.py deskull/rna-small-gpt2 --test-generate

  # specify local cache
  python test_huggingface_download.py deskull/rna-small-gpt2 --cache-dir ./models
        """,
    )

    parser.add_argument("repo_id", type=str, help="Hugging Face Hub repository ID")
    parser.add_argument("--revision", type=str, default="main", help="branch/tag/commit")
    parser.add_argument("--cache-dir", type=str, help="Cache directory to download to")
    parser.add_argument("--checkpoint-file", type=str, default="ckpt.pt", help="checkpoint file name")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cpu, cuda, auto)")
    parser.add_argument("--test-generate", action="store_true", help="Run text generation test")
    parser.add_argument(
        "--domain",
        type=str,
        choices=["rna", "genome", "protein_sequence", "compounds", "molecule_nat_lang"],
        help="Model domain (for tokenizer selection)",
    )
    parser.add_argument("--max-tokens", type=int, default=50, help="Maximum number of tokens to generate")
    parser.add_argument("--verbose", "-v", action="store_true", help="verbose output")

    return parser.parse_args()


def get_device(device_arg: str) -> str:
    """Determine device"""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_arg


def list_repo_contents(repo_id: str, revision: str = "main") -> list:
    """List repository contents"""
    print(f"\n[INFO] Checking repository contents: {repo_id}")
    try:
        files = list_repo_files(repo_id, revision=revision)
        print(f"[INFO] File list ({len(files)} items):")
        for f in files:
            print(f"  - {f}")
        return files
    except Exception as e:
        print(f"[ERROR] Failed to retrieve repository contents: {e}")
        return []


def download_checkpoint(repo_id: str, checkpoint_file: str, cache_dir: str = None, revision: str = "main") -> str:
    """Download checkpoint file"""
    print(f"\n[INFO] Downloading checkpoint: {checkpoint_file}")

    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=checkpoint_file,
            cache_dir=cache_dir,
            revision=revision,
        )
        print(f"[SUCCESS] Download completed: {local_path}")
        return local_path
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return None


def download_all_files(repo_id: str, cache_dir: str = None, revision: str = "main") -> str:
    """Download entire repository"""
    print("\n[INFO] Downloading entire repository...")

    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            revision=revision,
        )
        print(f"[SUCCESS] Download completed: {local_dir}")
        return local_dir
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return None


def load_checkpoint(checkpoint_path: str, device: str, verbose: bool = False):
    """Read checkpoint"""
    print("\n[INFO] Loading checkpoint...")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Check the contents of the checkpoint
        print("[INFO] Checkpoint key:")
        for key in checkpoint.keys():
            if key == "model":
                print(f"  - {key}: (state_dict with {len(checkpoint[key])} parameters)")
            elif isinstance(checkpoint[key], dict):
                print(f"  - {key}: (dict with {len(checkpoint[key])} items)")
            else:
                print(f"  - {key}: {checkpoint[key]}")

        # get model settings
        model_args = checkpoint.get("model_args", {})
        if model_args:
            print("\n[INFO] Model settings:")
            for k, v in model_args.items():
                print(f"  - {k}: {v}")

        return checkpoint

    except Exception as e:
        print(f"[ERROR] Failed to read checkpoint: {e}")
        import traceback

        traceback.print_exc()
        return None


def load_model(checkpoint: dict, device: str):
    """Build model from checkpoints"""
    print("\n[INFO] Building model...")

    try:
        # import GPT model
        from gpt2.model import GPT, GPTConfig

        model_args = checkpoint.get("model_args", {})
        if not model_args:
            print("[ERROR] model_args not included in checkpoint")
            return None

        # create GPTConfig
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

        # load state dictionary
        state_dict = checkpoint["model"]

        # remove unnecessary prefixes
        unwanted_prefix = "_orig_mod."
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # display model information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("[SUCCESS] Model construction completed")
        print(f" - Total number of parameters: {total_params:,}")
        print(f" - Number of trainable parameters: {trainable_params:,}")
        print(f" - memory usage: {total_params * 4 / (1024**2):.2f} MB (float32)")

        return model

    except Exception as e:
        print(f"[ERROR] Failed to build model: {e}")
        import traceback

        traceback.print_exc()
        return None


def load_tokenizer(domain: str):
    """Load tokenizer for domain"""
    print(f"\n[INFO] Loading tokenizer (domain: {domain})...")

    try:
        if domain == "rna":
            from rna.utils.bert_tokenizer import create_bert_rna_tokenizer

            tokenizer = create_bert_rna_tokenizer()
        elif domain == "genome":
            from genome_sequence.utils.tokenizer import create_genome_tokenizer

            tokenizer = create_genome_tokenizer()
        elif domain == "protein_sequence":
            from protein_sequence.utils.bert_tokenizer import create_bert_protein_tokenizer

            tokenizer = create_bert_protein_tokenizer()
        elif domain == "compounds":
            from compounds.utils.tokenizer import CompoundsTokenizer

            vocab_file = str(PROJECT_ROOT / "assets" / "molecules" / "vocab.txt")
            tokenizer = CompoundsTokenizer(vocab_file, 256)
        elif domain == "molecule_nat_lang":
            from molecule_nat_lang.utils.tokenizer import MoleculeNatLangTokenizer

            tokenizer = MoleculeNatLangTokenizer()
        else:
            print(f"[WARNING] Unknown domain: {domain}")
            return None

        print("[SUCCESS] Tokenizer loading complete")
        return tokenizer

    except Exception as e:
        print(f"[WARNING] Failed to load tokenizer: {e}")
        return None


def test_forward_pass(model, device: str, vocab_size: int = 1000, seq_len: int = 128):
    """Forward path test"""
    print("\n[INFO] Testing forward path...")

    try:
        # create random input tensor
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)

        with torch.no_grad():
            # forward pass
            logits, loss = model(input_ids)

        print("[SUCCESS] Forward pass successful")
        print(f" - Input shape: {input_ids.shape}")
        print(f" - Output shape: {logits.shape}")
        print(f" - Output statistics: mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")

        return True

    except Exception as e:
        print(f"[ERROR] Forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_generate(model, tokenizer, device: str, max_tokens: int = 50):
    """Text generation test"""
    print("\n[INFO] Testing text generation...")

    try:
        # Prepare initial token
        if tokenizer is not None:
            # If you have a tokenizer, use the appropriate starting token
            if hasattr(tokenizer, "encode"):
                # Sample input (change depending on your domain)
                sample_inputs = ["A", "G", "C", "U"]  # For RNA
                try:
                    input_ids = tokenizer.encode(sample_inputs[0])
                    if isinstance(input_ids, list):
                        input_ids = torch.tensor([input_ids], device=device)
                    else:
                        input_ids = input_ids.unsqueeze(0).to(device)
                except Exception:
                    input_ids = torch.tensor([[1]], device=device)  # Fallback
            else:
                input_ids = torch.tensor([[1]], device=device)
        else:
            # If there is no tokenizer, start with a random token
            input_ids = torch.randint(1, 100, (1, 1), device=device)

        print(f" - starting token: {input_ids.tolist()}")

        # generate
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=0.8,
                top_k=40,
            )

        print("[SUCCESS] Text generation successful")
        print(f" - Number of generated tokens: {generated.shape[1]}")
        print(f" - Generated token ID: {generated[0, :20].tolist()}...")  # first 20 tokens

        # decode (if tokenizer is present)
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            try:
                decoded = tokenizer.decode(generated[0].tolist())
                print(f" - Decoded result: {decoded[:100]}...")  # First 100 characters
            except Exception as e:
                print(f" - decoding error: {e}")

        return True

    except Exception as e:
        print(f"[ERROR] Text generation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_tests(
    repo_id: str,
    revision: str = "main",
    cache_dir: str = None,
    checkpoint_file: str = "ckpt.pt",
    device: str = "auto",
    test_generate: bool = False,
    domain: str = None,
    max_tokens: int = 50,
    verbose: bool = False,
) -> dict:
    """Run all tests"""

    results = {
        "repo_id": repo_id,
        "success": True,
        "tests": {},
    }

    device = get_device(device)
    print(f"\n{'=' * 60}")
    print("Hugging Face Hub model test")
    print(f"{'=' * 60}")
    print(f"Repository ID: {repo_id}")
    print(f"device: {device}")
    print(f"{'=' * 60}")

    # 1. Check repository contents
    files = list_repo_contents(repo_id, revision)
    results["tests"]["list_files"] = len(files) > 0
    if not files:
        results["success"] = False
        return results

    # Find checkpoint file
    pt_files = [f for f in files if f.endswith(".pt")]
    if not pt_files:
        print("[ERROR] .pt file not found")
        results["success"] = False
        return results

    # specified file or ckpt.pt or first .pt file
    if checkpoint_file in files:
        target_file = checkpoint_file
    elif "ckpt.pt" in files:
        target_file = "ckpt.pt"
    else:
        target_file = pt_files[0]
    print(f"\n[INFO] Checkpoint to use: {target_file}")

    # 2. Download Checkpoint
    checkpoint_path = download_checkpoint(repo_id, target_file, cache_dir, revision)
    results["tests"]["download"] = checkpoint_path is not None
    if not checkpoint_path:
        results["success"] = False
        return results

    # 3. Read checkpoint
    checkpoint = load_checkpoint(checkpoint_path, device, verbose)
    results["tests"]["load_checkpoint"] = checkpoint is not None
    if not checkpoint:
        results["success"] = False
        return results

    # 4. Build the model
    model = load_model(checkpoint, device)
    results["tests"]["load_model"] = model is not None
    if not model:
        results["success"] = False
        return results

    # Model settingsfromvocab_sizeget
    model_args = checkpoint.get("model_args", {})
    vocab_size = model_args.get("vocab_size", 1000)
    block_size = model_args.get("block_size", 128)

    # 5. Forward path test
    forward_ok = test_forward_pass(model, device, vocab_size, min(block_size, 128))
    results["tests"]["forward_pass"] = forward_ok
    if not forward_ok:
        results["success"] = False

    # 6. Generated test (optional)
    if test_generate:
        tokenizer = None
        if domain:
            tokenizer = load_tokenizer(domain)

        generate_ok = test_generate_func(model, tokenizer, device, max_tokens)
        results["tests"]["generate"] = generate_ok
        if not generate_ok:
            results["success"] = False

    return results


# Rename so that test_generate and function name do not overlap
test_generate_func = test_generate


def print_summary(results: dict):
    """View test results summary"""
    print(f"\n{'=' * 60}")
    print("Test results summary")
    print(f"{'=' * 60}")

    for test_name, passed in results["tests"].items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")

    print(f"{'=' * 60}")
    if results["success"]:
        print("[SUCCESS] All tests passed!")
    else:
        print("[FAILED] Some tests failed")
    print(f"{'=' * 60}")


def main():
    """Main function"""
    args = parse_args()

    results = run_tests(
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
        checkpoint_file=args.checkpoint_file,
        device=args.device,
        test_generate=args.test_generate,
        domain=args.domain,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
    )

    print_summary(results)

    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
