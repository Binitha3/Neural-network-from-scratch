"""
nn — a small neural-network library built on NumPy only.

Implements forward propagation, backpropagation and gradient-descent-style
optimisers from first principles. No autograd, no frameworks.

    >>> from nn import Sequential, Dense, ReLU, SoftmaxCrossEntropy, Adam
    >>> model = Sequential([Dense(2, 32), ReLU(), Dense(32, 3)])
    >>> loss_fn = SoftmaxCrossEntropy()
    >>> opt = Adam(model, lr=1e-3)
    >>> model.train_on_batch(X, y, loss_fn, opt)
    1.0986...
"""

from .activations import ReLU, Sigmoid, Tanh, LeakyReLU
from .layers import Dense, Dropout, Flatten
from .losses import SoftmaxCrossEntropy, MSELoss, BinaryCrossEntropy
from .model import Sequential
from .optimizers import SGD, SGDMomentum, Adam
from .data import (BatchIterator, load_mnist, make_spirals, make_moons,
                   one_hot, train_val_split)
from .utils import accuracy, set_seed

__version__ = "1.0.0"

__all__ = [
    "ReLU", "Sigmoid", "Tanh", "LeakyReLU",
    "Dense", "Dropout", "Flatten",
    "SoftmaxCrossEntropy", "MSELoss", "BinaryCrossEntropy",
    "Sequential", "SGD", "SGDMomentum", "Adam",
    "BatchIterator", "load_mnist", "make_spirals", "make_moons",
    "one_hot", "train_val_split",
    "accuracy", "set_seed",
]
