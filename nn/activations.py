"""Activation functions.

Every activation is a layer: ``forward`` transforms the input and caches what
the backward pass needs; ``backward`` applies the chain rule to turn the
upstream gradient into a gradient w.r.t. the layer input.
"""

import numpy as np


class ReLU:
    """Rectified Linear Unit: f(x) = max(0, x), f'(x) = 1[x > 0]."""

    def forward(self, x, training=True):
        self.mask = (x > 0).astype(x.dtype)
        return x * self.mask

    def backward(self, grad_out):
        return grad_out * self.mask

    def params(self):
        return {}

    def grads(self):
        return {}


class LeakyReLU:
    """Leaky ReLU: f(x) = x if x > 0 else alpha * x (alpha = 0.01)."""

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x, training=True):
        self.mask = (x > 0).astype(x.dtype)
        return x * self.mask + self.alpha * x * (1.0 - self.mask)

    def backward(self, grad_out):
        return grad_out * (self.mask + self.alpha * (1.0 - self.mask))

    def params(self):
        return {}

    def grads(self):
        return {}


class Sigmoid:
    """Sigmoid: f(x) = 1 / (1 + exp(-x)), f'(x) = f(x) * (1 - f(x))."""

    def forward(self, x, training=True):
        self.out = 1.0 / (1.0 + np.exp(-x))
        return self.out

    def backward(self, grad_out):
        return grad_out * self.out * (1.0 - self.out)

    def params(self):
        return {}

    def grads(self):
        return {}


class Tanh:
    """Hyperbolic tangent: f(x) = tanh(x), f'(x) = 1 - tanh(x)^2."""

    def forward(self, x, training=True):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad_out):
        return grad_out * (1.0 - self.out ** 2)

    def params(self):
        return {}

    def grads(self):
        return {}
