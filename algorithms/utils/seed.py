"""Random seed helpers for reproducible experiments."""
import os
import random
import numpy as np

def set_random_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and optionally PyTorch if installed."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # type: ignore
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except AttributeError:
        pass
