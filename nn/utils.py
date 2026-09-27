"""Shared utilities."""

import numpy as np


def accuracy(predictions, targets):
    """Fraction of correct class predictions (handles one-hot or integer targets)."""
    predictions = np.asarray(predictions).ravel()
    targets = np.asarray(targets)
    if targets.ndim == 2:
        targets = targets.argmax(axis=1)
    return float(np.mean(predictions == targets))


def set_seed(seed):
    """Seed the module-level default RNG used by ``Sequential``."""
    return np.random.default_rng(seed)
