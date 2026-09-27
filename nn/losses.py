"""Loss functions.

Each loss exposes ``forward(pred, target) -> float`` and
``backward() -> dL/dpred``. The forward pass caches everything the analytic
backward pass needs, so the two always agree with each other — which is
exactly what the numerical gradient checker in ``tests/`` verifies.
"""

import numpy as np


class SoftmaxCrossEntropy:
    """Softmax + cross-entropy fused into a numerically stable op.

    Keeping them together lets the backward pass collapse to the classic
    identity ``(softmax(y) - one_hot(t)) / batch`` — the subtraction of two
    numbers of similar size, which avoids the overflow of a naive
    ``exp`` and the log(0) of a naive log. Handles both one-hot and integer
    class targets.
    """

    def forward(self, logits, target):
        shifted = logits - logits.max(axis=1, keepdims=True)  # max-shift trick
        exp = np.exp(shifted)
        self.probs = exp / exp.sum(axis=1, keepdims=True)

        if target.ndim == 2:                      # one-hot -> indices
            target = target.argmax(axis=1)
        self.target = target
        n = logits.shape[0]
        correct = self.probs[np.arange(n), target]
        return float(np.mean(-np.log(correct + 1e-12)))

    def backward(self):
        n = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(n), self.target] -= 1.0
        return grad / n

    def predicted_classes(self):
        return self.probs.argmax(axis=1)


class BinaryCrossEntropy:
    """BCE over sigmoid probabilities with the fused, overflow-safe form.

    Uses ``np.logaddexp(0, -z)`` for the positive term and
    ``np.logaddexp(0, z)`` for the negative one — mathematically
    ``log(1+e^-z)`` and ``log(1+e^z)``, but stable for arbitrarily large |z|.
    """

    def forward(self, z, target):
        self.target = target
        self.z = z
        pos = np.logaddexp(0.0, -z)
        neg = np.logaddexp(0.0, z)
        return float(np.mean(target * pos + (1.0 - target) * neg))

    def backward(self):
        # dL/dz for sigmoid(z) fused with BCE = sigmoid(z) - t, per element.
        s = 1.0 / (1.0 + np.exp(-self.z))
        return (s - self.target) / self.z.shape[0]

    def predicted_classes(self):
        return (self.z > 0).astype(int).ravel()


class MSELoss:
    """Mean-squared error over all elements: mean((pred - target)^2)."""

    def forward(self, pred, target):
        self.diff = pred - target
        return float(np.mean(self.diff ** 2))

    def backward(self):
        return 2.0 * self.diff / self.diff.size
