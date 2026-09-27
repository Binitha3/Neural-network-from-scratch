"""Demo: analytic backward gradients vs. brute-force finite differences.

Run:  python scripts/gradient_check_demo.py

Prints max/mean abs error between the two for each parameter of a small
network — the numbers a reviewer wants to see next to "implemented
backpropagation by hand".
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from nn import Dense, ReLU, SoftmaxCrossEntropy, Sequential


def numerical_grad_wrt(param, value_fn, eps=1e-5):
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        orig = param[i]
        param[i] = orig + eps
        fp = value_fn()
        param[i] = orig - eps
        fm = value_fn()
        param[i] = orig
        grad[i] = (fp - fm) / (2.0 * eps)
        it.iternext()
    return grad


def main():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(16, 3))
    y = rng.integers(0, 2, size=16)

    model = Sequential([Dense(3, 8), ReLU(), Dense(8, 2)])
    loss_fn = SoftmaxCrossEntropy()

    logits = model.forward(x)
    loss_fn.forward(logits, y)
    model.backward(loss_fn.backward())

    print(f"{'param':<8} {'shape':<12} {'max abs err':>12} {'mean abs err':>13}")
    for li, layer in enumerate(model.layers):
        for name, param in layer.params().items():
            numeric = numerical_grad_wrt(
                param, lambda: SoftmaxCrossEntropy().forward(model.forward(x), y))
            analytic = layer.grads()[name]
            err = np.abs(analytic - numeric)
            print(f"layer{li}.{name:<4} {str(param.shape):<12} "
                  f"{err.max():>12.3e} {err.mean():>13.3e}")

    print("\nErrors at the 1e-9 .. 1e-7 level = backward pass is exact "
          "up to float rounding.")


if __name__ == "__main__":
    main()
