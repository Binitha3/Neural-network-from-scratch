"""Datasets and batching.

``load_mnist`` downloads the canonical ``mnist.npz`` once (mirrored by
Google/TensorFlow), normalises pixels to [0, 1] and caches it under
``data/`` so training runs offline after the first download. The toy
generators produce 2-D binary problems chosen for the decision-boundary
visuals: two interleaved half-moons, and two-arm spirals that a linear
model cannot separate at all — the classic showcase for hidden layers.
"""

from pathlib import Path

import numpy as np

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def load_mnist(data_dir="data", flatten=True, download=True):
    """Returns (X_train, y_train, X_test, y_test).

    X: float32 in [0, 1], shape (n, 784) if flatten else (n, 28, 28).
    y: int64 labels in 0..9.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "mnist.npz"

    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} not found; pass download=True")
        import urllib.request
        print(f"downloading MNIST to {path} (~11 MB, first run only)...")
        urllib.request.urlretrieve(MNIST_URL, path)

    with np.load(path) as npz:
        X_train = npz["x_train"].astype(np.float32) / 255.0
        y_train = npz["y_train"].astype(np.int64)
        X_test = npz["x_test"].astype(np.float32) / 255.0
        y_test = npz["y_test"].astype(np.int64)

    if flatten:
        X_train = X_train.reshape(len(X_train), -1)
        X_test = X_test.reshape(len(X_test), -1)
    return X_train, y_train, X_test, y_test


def make_moons(n_samples=400, noise=0.15, rng=None):
    """Two interleaved half-circles: linearly inseparable but easy for an MLP."""
    rng = rng if rng is not None else np.random.default_rng()
    n = n_samples // 2
    theta1 = rng.uniform(0, np.pi, n)
    theta2 = rng.uniform(0, np.pi, n)
    moons0 = np.stack([np.cos(theta1), np.sin(theta1)], axis=1)
    moons1 = np.stack([1 - np.cos(theta2), 0.5 - np.sin(theta2)], axis=1)
    X = np.vstack([moons0, moons1]) + rng.normal(0, noise, (2 * n, 2))
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    return X.astype(np.float32), y


def make_spirals(n_samples=400, classes=2, noise=0.1, turns=1.75, rng=None):
    """Two-arm spiral — the goto demo that a network learns curved boundaries."""
    rng = rng if rng is not None else np.random.default_rng()
    n = n_samples // classes
    xs, ys = [], []
    for c in range(classes):
        r = np.linspace(0.05, 1.0, n)
        theta = np.linspace(0, turns * 2 * np.pi, n) + c * (2 * np.pi / classes)
        pts = np.stack([r * np.sin(theta), r * np.cos(theta)], axis=1)
        xs.append(pts + rng.normal(0, noise, (n, 2)))
        ys.append(np.full(n, c))
    return np.vstack(xs).astype(np.float32), np.concatenate(ys)


def one_hot(y, num_classes):
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1.0
    return out


def train_val_split(X, y, val_frac=0.2, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    idx = rng.permutation(len(X))
    cut = int(len(X) * (1.0 - val_frac))
    tr, va = idx[:cut], idx[cut:]
    return X[tr], y[tr], X[va], y[va]


class BatchIterator:
    """Yields shuffled mini-batches of (x, y) each pass; shuffle=False for eval."""

    def __init__(self, batch_size=32, shuffle=True, rng=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, X, y):
        n = len(X)
        order = self.rng.permutation(n) if self.shuffle else np.arange(n)
        for start in range(0, n, self.batch_size):
            idx = order[start:start + self.batch_size]
            yield X[idx], y[idx]
