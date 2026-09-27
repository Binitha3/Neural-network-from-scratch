"""Optimisers: plain SGD, SGD + momentum, and Adam.

All three consume the model's ``params()`` / ``grads()`` dictionaries and
mutate the parameter arrays in place.

Subtlety worth knowing (and a classic aliasing bug): a layer's ``backward``
typically *rebinds* its gradient attributes (``self.dW = ...``), replacing
the old array. An optimiser that cached references to those arrays at
construction would silently read stale zeros forever. So instead of caching,
each step re-resolves the ``(param, grad)`` pairs through the model — cheap
(dictionary rebuild per step) and correct for any layer implementation.
The gradient-descent-reduces-loss test exists precisely to catch this class
of bug end to end.
"""

import numpy as np


class SGD:
    """Vanilla gradient descent: theta <- theta - lr * grad."""

    def __init__(self, model, lr=0.1):
        self.model = model
        self.lr = lr

    def _pairs(self):
        return list(self.model.params_and_grads())

    def step(self):
        for param, grad in self._pairs():
            param -= self.lr * grad

    def zero_grad(self):
        for _, grad in self._pairs():
            grad[...] = 0.0


class SGDMomentum:
    """SGD with classical momentum: v <- mu*v + grad; theta <- theta - lr*v.

    Momentum accumulates a velocity across steps, damping oscillation across
    ravines of the loss surface and accelerating along consistent descent
    directions.
    """

    def __init__(self, model, lr=0.05, momentum=0.9):
        self.model = model
        self.lr, self.mu = lr, momentum
        self.vel = [np.zeros_like(p) for p, _ in model.params_and_grads()]

    def step(self):
        for (param, grad), v in zip(self.model.params_and_grads(), self.vel):
            v *= self.mu
            v += grad
            param -= self.lr * v

    def zero_grad(self):
        for _, grad in self.model.params_and_grads():
            grad[...] = 0.0


class Adam:
    """Adam (Kingma & Ba, 2015) with bias-corrected moment estimates.

    m_t = beta1*m + (1-beta1)*g         (1st moment / mean)
    v_t = beta2*v + (1-beta2)*g^2       (2nd moment / uncentered variance)
    m_hat = m_t / (1 - beta1**t)        (bias correction, t = step count)
    v_hat = v_t / (1 - beta2**t)
    theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.model = model
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.m = [np.zeros_like(p) for p, _ in model.params_and_grads()]
        self.v = [np.zeros_like(p) for p, _ in model.params_and_grads()]
        self.t = 0

    def step(self):
        self.t += 1
        b1t = 1.0 - self.b1 ** self.t
        b2t = 1.0 - self.b2 ** self.t
        for (param, grad), m, v in zip(self.model.params_and_grads(),
                                       self.m, self.v):
            m *= self.b1
            m += (1.0 - self.b1) * grad
            v *= self.b2
            v += (1.0 - self.b2) * (grad * grad)
            m_hat = m / b1t
            v_hat = v / b2t
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for _, grad in self.model.params_and_grads():
            grad[...] = 0.0
