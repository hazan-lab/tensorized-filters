import torch

def get_device() -> torch.device:
    """Get appropriate compute device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
