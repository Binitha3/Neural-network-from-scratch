"""Numerical gradient checking: the acid test for hand-derived backprop.

For every differentiable quantity we compare the analytic gradient
(backward pass) against a brute-force central finite difference of the loss:

    dL/dx_i ~= [L(x + eps) - L(x - eps)] / (2 * eps)

If our chain-rule derivations are wrong anywhere, these tests fail. They are
the reason this repo can claim backpropagation is *correct*, not just that it
runs. Inputs are chosen away from ReLU kinks so finite differences don't
straddle a discontinuity.
"""

import unittest

import numpy as np

from nn import (Dense, ReLU, LeakyReLU, Sigmoid, Tanh, Dropout, Flatten,
                SoftmaxCrossEntropy, BinaryCrossEntropy, MSELoss, Sequential)


def numerical_grad(value_fn, x, eps=1e-5):
    """Central-difference gradient of scalar value_fn() w.r.t. array x."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index
        orig = x[i]
        x[i] = orig + eps
        fp = value_fn()
        x[i] = orig - eps
        fm = value_fn()
        x[i] = orig
        grad[i] = (fp - fm) / (2.0 * eps)
        it.iternext()
    return grad


class GradientCheck(unittest.TestCase):
    TOL = 1e-5

    def setUp(self):
        self.rng = np.random.default_rng(0)

    def check(self, analytic, numeric, msg):
        np.testing.assert_allclose(analytic, numeric, atol=self.TOL,
                                   err_msg=msg)

    # ---- activations -------------------------------------------------- #
    def test_relu(self):
        x = self.rng.choice([-0.8, -0.5, -0.2, 0.2, 0.5, 0.9], size=(4, 6))
        g = self.rng.standard_normal(x.shape)
        relu = ReLU()
        relu.forward(x)
        analytic = relu.backward(g)
        self.check(analytic,
                   numerical_grad(lambda: (ReLU().forward(x) * g).sum(), x),
                   "ReLU backward")

    def test_leaky_relu(self):
        x = self.rng.choice([-0.8, -0.5, -0.2, 0.2, 0.5, 0.9], size=(4, 6))
        g = self.rng.standard_normal(x.shape)
        leaky = LeakyReLU(alpha=0.1)
        leaky.forward(x)
        analytic = leaky.backward(g)
        self.check(analytic,
                   numerical_grad(
                       lambda: (LeakyReLU(alpha=0.1).forward(x) * g).sum(), x),
                   "LeakyReLU backward")

    def test_sigmoid(self):
        x = self.rng.uniform(-3, 3, size=(4, 6))
        g = self.rng.standard_normal(x.shape)
        sig = Sigmoid()
        sig.forward(x)
        analytic = sig.backward(g)
        self.check(analytic,
                   numerical_grad(lambda: (Sigmoid().forward(x) * g).sum(), x),
                   "Sigmoid backward")

    def test_tanh(self):
        x = self.rng.uniform(-3, 3, size=(4, 6))
        g = self.rng.standard_normal(x.shape)
        t = Tanh()
        t.forward(x)
        analytic = t.backward(g)
        self.check(analytic,
                   numerical_grad(lambda: (Tanh().forward(x) * g).sum(), x),
                   "Tanh backward")

    # ---- dense layer --------------------------------------------------- #
    def test_dense_weight_input_bias(self):
        x = self.rng.uniform(-1, 1, size=(5, 4))
        seed_W = self.rng.normal(size=(4, 3))
        seed_b = self.rng.normal(size=3)
        gout = self.rng.standard_normal((5, 3))

        layer = Dense(4, 3)
        layer.W = seed_W.copy()
        layer.b = seed_b.copy()

        layer.forward(x)
        analytic_dx = layer.backward(gout)

        def value():
            tmp = Dense(4, 3)
            tmp.W, tmp.b = seed_W, seed_b
            return float((tmp.forward(x) * gout).sum())

        self.check(analytic_dx, numerical_grad(value, x), "Dense dX")
        self.check(layer.dW, numerical_grad(value, seed_W), "Dense dW")
        self.check(layer.db, numerical_grad(value, seed_b), "Dense db")

    # ---- losses --------------------------------------------------------- #
    def test_softmax_cross_entropy(self):
        logits = self.rng.normal(size=(6, 5))
        y = self.rng.integers(0, 5, size=6)
        loss = SoftmaxCrossEntropy()
        loss.forward(logits, y)
        self.check(loss.backward(),
                   numerical_grad(lambda: SoftmaxCrossEntropy().forward(logits, y),
                                  logits),
                   "SoftmaxCE dLogits")

    def test_softmax_cross_entropy_one_hot_target(self):
        logits = self.rng.normal(size=(6, 5))
        y = np.eye(5)[self.rng.integers(0, 5, size=6)]
        loss = SoftmaxCrossEntropy()
        loss.forward(logits, y)
        self.check(loss.backward(),
                   numerical_grad(lambda: SoftmaxCrossEntropy().forward(logits, y),
                                  logits),
                   "SoftmaxCE dLogits (one-hot)")

    def test_binary_cross_entropy(self):
        z = self.rng.uniform(-4, 4, size=(8, 1))
        t = self.rng.integers(0, 2, size=(8, 1)).astype(float)
        loss = BinaryCrossEntropy()
        loss.forward(z, t)
        self.check(loss.backward(),
                   numerical_grad(lambda: BinaryCrossEntropy().forward(z, t), z),
                   "BCE dZ")

    def test_mse(self):
        pred = self.rng.normal(size=(8, 3))
        target = self.rng.normal(size=(8, 3))
        loss = MSELoss()
        loss.forward(pred, target)
        self.check(loss.backward(),
                   numerical_grad(lambda: MSELoss().forward(pred, target), pred),
                   "MSE dPred")

    # ---- whole model ------------------------------------------------------ #
    def test_full_model_input_gradient(self):
        x = self.rng.uniform(-1, 1, size=(7, 4))
        y = self.rng.integers(0, 3, size=7)
        model = Sequential([Dense(4, 16), ReLU(), Dense(16, 3)])
        loss = SoftmaxCrossEntropy()

        logits = model.forward(x)
        loss.forward(logits, y)
        analytic = model.backward(loss.backward())

        def value():
            m = Sequential([Dense(4, 16), ReLU(), Dense(16, 3)])
            m.layers[0].W = model.layers[0].W
            m.layers[0].b = model.layers[0].b
            m.layers[2].W = model.layers[2].W
            m.layers[2].b = model.layers[2].b
            return SoftmaxCrossEntropy().forward(m.forward(x), y)

        self.check(analytic, numerical_grad(value, x), "model dX")

    def test_full_model_end_to_end_gradient_descent_reduces_loss(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(64, 3))
        W_true = 3.0 * rng.normal(size=(3, 2))  # scale up for a wide margin
        y = (x @ W_true).argmax(axis=1)
        model = Sequential([Dense(3, 8), Tanh(), Dense(8, 2)])
        loss_fn = SoftmaxCrossEntropy()
        from nn import SGDMomentum
        opt = SGDMomentum(model, lr=0.2, momentum=0.9)
        first = model.train_on_batch(x, y, loss_fn, opt)
        last = first
        for _ in range(100):
            last = model.train_on_batch(x, y, loss_fn, opt)
        self.assertLess(last, first / 10.0, "loss should drop by an order of magnitude")
        self.assertLess(last, 0.05, "loss should approach zero on separable data")


if __name__ == "__main__":
    unittest.main()
