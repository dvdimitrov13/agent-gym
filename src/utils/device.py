import torch


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype() -> torch.dtype:
    device = get_device()
    if device == "cuda":
        return torch.bfloat16
    # MPS float16/bfloat16 support is incomplete for generation
    return torch.float32
