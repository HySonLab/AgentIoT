"""Global seed control.

Every stochastic component in the pipeline takes its seed from here so that
"N independent runs with different seeds" actually produces independent runs.
The previous implementation hard-coded random_state=42 inside model
constructors, which made the reported seed-robustness study vacuous.
"""

import os
import random

import numpy as np

try:
    import torch
except ImportError:  # torch is optional for data-only work
    torch = None


def set_seeds(seed: int) -> None:
    """Seed python, numpy, and torch RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def spawn_rng(seed: int) -> np.random.Generator:
    """Independent generator for a component, derived from the run seed."""
    return np.random.default_rng(seed)
