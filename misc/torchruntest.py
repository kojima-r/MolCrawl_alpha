"""
Check GPU environment and DDP settings
"""

import os

import torch


def check_environment():
    print("=== GPU environment check ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPUnumber: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} (Memory: {props.total_memory / 1e9:.1f}GB)")

    print("\n=== Check environment variables ===")
    ddp_vars = ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
    for var in ddp_vars:
        value = os.environ.get(var, "Not set")
        print(f"{var}: {value}")

    print("\n=== DDP determination ===")
    ddp = int(os.environ.get("RANK", -1)) != -1
    print(f"DDP execution: {ddp}")

    if ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        print(f"Local rank: {local_rank}")
        if local_rank >= torch.cuda.device_count():
            print(f"⚠️ Error: LOCAL_RANK({local_rank}) >= Number of GPUs({torch.cuda.device_count()})")
            return False

    return True


if __name__ == "__main__":
    success = check_environment()
    if not success:
        print("\n❌ There is a problem with the environment settings")
        exit(1)
    else:
        print("\n✅ Environment settings are normal")
