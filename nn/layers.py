"""Trainable layers.

The central invariant of the library: every layer guarantees
``backward(dL/dy) -> dL/dx`` and exposes learnable parameters through the
``params()`` / ``grads()`` dictionaries, keyed by the *same* names. The
optimiser updates ``layer.params()[name] -= lr * layer.grads()[name]`` —
this is the whole contract that makes composition work.
"""

import numpy as np


class Dense:
    """Fully-connected layer: y = x @ W + b.

    Weights use He (Kaiming) initialisation, the standard choice when the
    layer feeds a ReLU-family activation: W ~ N(0, sqrt(2 / fan_in)),
    biases start at zero.
    """

    def __init__(self, in_features, out_features, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        self.in_features = in_features
        self.out_features = out_features
        self.W = rng.normal(0.0, np.sqrt(2.0 / in_features),
                            size=(in_features, out_features))
        self.b = np.zeros(out_features)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x, training=True):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_out):
        """Chain rule for y = xW + b:
           dL/dW = x^T @ grad_out,  dL/db = sum over batch,  dL/dx = grad_out @ W^T.
        """
        self.dW = self.x.T @ grad_out
        self.db = grad_out.sum(axis=0)
        return grad_out @ self.W.T

    def params(self):
        return {"W": self.W, "b": self.b}

    def grads(self):
        return {"W": self.dW, "b": self.db}


class Dropout:
    """Inverted dropout: active only while ``training``.

    Drops each unit with probability p and scales survivors by 1/(1-p) at
    train time, so activation magnitudes stay calibrated and *no rescaling
    is needed at inference*. In eval mode it is the identity.
    """

    def __init__(self, p=0.5, rng=None):
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout probability must be in [0, 1), got {p}")
        self.p = p
        self.rng = rng if rng is not None else np.random.default_rng()

    def forward(self, x, training=True):
        if not training or self.p == 0.0:
            self.mask = None
            return x
        self.mask = (self.rng.random(x.shape) >= self.p) / (1.0 - self.p)
        return x * self.mask

    def backward(self, grad_out):
        if self.mask is None:
            return grad_out
        return grad_out * self.mask

    def params(self):
        return {}

    def grads(self):
        return {}


class Flatten:
    """Reshapes (batch, ...) -> (batch, -1). Common before Dense heads."""

    def forward(self, x, training=True):
        self.x_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_out):
        return grad_out.reshape(self.x_shape)

    def params(self):
        return {}

    def grads(self):
        return {}
