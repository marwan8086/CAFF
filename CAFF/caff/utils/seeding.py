"""
caff/utils/seeding.py
=====================
Deterministic seeding helper. Paper §8.4 mandates reproducibility
across {42, 1337, 2024}; this module ensures all RNG sources
(torch CPU/CUDA, numpy, python) are seeded identically.
"""

from __future__ import annotations

import os
import random
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set seeds for all RNG sources used by CAFF.

    Parameters
    ----------
    seed : int
        The seed value. Paper uses {42, 1337, 2024}.
    deterministic : bool
        If True, also set cuDNN to deterministic mode. This makes
        runs bit-reproducible at the cost of ~10% throughput.

    Notes
    -----
    Sets:
      - random.seed
      - numpy.random.seed
      - torch.manual_seed (CPU)
      - torch.cuda.manual_seed_all (all GPUs)
      - PYTHONHASHSEED env var (for hash randomization)
      - cuDNN deterministic flags (if deterministic=True)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # CUBLAS workspace config required for full determinism
        # in matmul ops on CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            # torch < 1.8
            pass

    logger.info(f"Global seed set to {seed} (deterministic={deterministic})")