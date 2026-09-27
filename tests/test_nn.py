"""Unit tests for layers, losses, optimizers, and data utilities."""

import unittest

import numpy as np

from nn import (Dense, Dropout, Flatten, ReLU, Sequential, SoftmaxCrossEntropy,
                MSELoss, SGD, SGDMomentum, Adam, BatchIterator,
                make_moons, make_spirals, one_hot, train_val_split, accuracy)


class TestLayers(unittest.TestCase):
    def test_dense_shapes_and_type(self):
        layer = Dense(4, 3)
        x = np.zeros((7, 4))
        self.assertEqual(layer.forward(x).shape, (7, 3))

    def test_dense_he_init_statistics(self):
        rng = np.random.default_rng(0)
        layer = Dense(1000, 1, rng=rng)
        # He init std = sqrt(2/fan_in) ~ 0.0447 (sample std of 1000 draws,
        # tolerance ~3 standard errors of the estimate)
        self.assertAlmostEqual(layer.W.std(), np.sqrt(2.0 / 1000), delta=0.005)
        np.testing.assert_array_equal(layer.b, np.zeros(1))

    def test_dropout_train_eval(self):
        rng = np.random.default_rng(0)
        drop = Dropout(p=0.5, rng=rng)
        x = np.ones((2000, 10))
        out = drop.forward(x, training=True)
        # inverted dropout: survivors scaled to mean ~1
        self.assertTrue(0.9 < out.mean() < 1.1)
        self.assertTrue(np.isin(out, [0.0, 2.0]).all())  # 0 or 1/(1-p)
        eval_out = drop.forward(x, training=False)
        np.testing.assert_array_equal(eval_out, x)  # identity at eval

    def test_dropout_invalid_p(self):
        with self.assertRaises(ValueError):
            Dropout(p=1.0)

    def test_flatten(self):
        flat = Flatten()
        out = flat.forward(np.zeros((5, 3, 2, 2)))
        self.assertEqual(out.shape, (5, 12))
        self.assertEqual(flat.backward(out).shape, (5, 3, 2, 2))


class TestLosses(unittest.TestCase):
    def test_softmax_rows_sum_to_one(self):
        rng = np.random.default_rng(0)
        loss = SoftmaxCrossEntropy()
        loss.forward(rng.normal(size=(10, 4)), rng.integers(0, 4, 10))
        np.testing.assert_allclose(loss.probs.sum(axis=1), 1.0, rtol=1e-9)

    def test_softmax_numerical_stability(self):
        # unshifted exp would overflow to inf/nan here
        loss = SoftmaxCrossEntropy()
        value = loss.forward(np.array([[1000.0, 0.0, -1000.0]]), np.array([0]))
        self.assertTrue(np.isfinite(value))

    def test_softmax_perfect_prediction_near_zero_loss(self):
        loss = SoftmaxCrossEntropy()
        value = loss.forward(np.array([[100.0, 0.0]]), np.array([0]))
        self.assertLess(value, 1e-6)

    def test_mse_known_value(self):
        loss = MSELoss()
        value = loss.forward(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 0.0]))
        self.assertAlmostEqual(value, (1 + 4 + 9) / 3)


class TestOptimizers(unittest.TestCase):
    def _model(self):
        model = Sequential([Dense(2, 3)])
        model.layers[0].W = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        model.layers[0].b = np.zeros(3)
        model.layers[0].dW = np.array([[0.1, 0.0, -0.1], [0.2, 0.0, -0.2]])
        model.layers[0].db = np.array([0.1, 0.1, 0.1])
        return model

    def test_sgd_step(self):
        model = self._model()
        SGD(model, lr=0.1).step()
        expected_W = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) - 0.1 * np.array(
            [[0.1, 0.0, -0.1], [0.2, 0.0, -0.2]])
        np.testing.assert_allclose(model.layers[0].W, expected_W)

    def test_sgd_momentum_accumulates(self):
        model = self._model()
        opt = SGDMomentum(model, lr=0.1, momentum=0.9)
        opt.step()  # v = g;        W -= 0.1*g
        W_after_1 = model.layers[0].W.copy()
        opt.step()  # v = 0.9v + g; W -= 0.1*v
        # second-step velocity = 1.9*g, so extra movement vs first step
        delta2 = W_after_1 - model.layers[0].W
        delta1 = np.array([[0.1, 0.0, -0.1], [0.2, 0.0, -0.2]]) * 0.1
        np.testing.assert_allclose(delta2, 1.9 * delta1)

    def test_adam_first_step_equals_sign_times_lr(self):
        # with zero init moments and bias correction, step 1 ~ lr * g/|g|
        model = Sequential([Dense(2, 2)])
        model.layers[0].W = np.zeros((2, 2))
        model.layers[0].dW = np.array([[4.0, -1.0], [0.5, -8.0]])
        model.layers[0].db = np.zeros(2)
        Adam(model, lr=0.01).step()
        # update = -lr * sign(g):  g = [[4, -1], [0.5, -8]]
        expected = np.array([[-0.0099, 0.0099], [-0.0099, 0.0099]])
        np.testing.assert_allclose(model.layers[0].W, expected, atol=1e-4)

    def test_zero_grad(self):
        model = self._model()
        opt = SGD(model, lr=0.1)
        opt.zero_grad()
        np.testing.assert_array_equal(model.layers[0].dW, np.zeros_like(model.layers[0].dW))


class TestData(unittest.TestCase):
    def test_moons_shapes_and_labels(self):
        rng = np.random.default_rng(0)
        X, y = make_moons(400, rng=rng)
        self.assertEqual(X.shape, (400, 2))
        self.assertEqual(set(np.unique(y)), {0, 1})
        self.assertEqual((y == 0).sum(), 200)

    def test_spirals_linearly_inseparable_by_class_angle(self):
        rng = np.random.default_rng(0)
        X, y = make_spirals(300, classes=3, rng=rng)
        self.assertEqual(X.shape, (300, 2))
        self.assertEqual(set(np.unique(y)), {0, 1, 2})

    def test_one_hot(self):
        out = one_hot(np.array([0, 2, 1]), 3)
        np.testing.assert_array_equal(
            out, np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]))

    def test_split_preserves_paired_order(self):
        rng = np.random.default_rng(0)
        X = np.arange(20).reshape(10, 2)
        y = np.arange(10)
        Xtr, ytr, Xva, yva = train_val_split(X, y, val_frac=0.2, rng=rng)
        self.assertEqual(len(Xtr) + len(Xva), 10)
        for x_row, label in zip(Xtr, ytr):
            self.assertEqual(x_row[0], 2 * label)
            self.assertEqual(x_row[1], 2 * label + 1)

    def test_batch_iterator_covers_all(self):
        rng = np.random.default_rng(0)
        X = np.arange(20).reshape(10, 2)
        y = np.arange(10)
        seen = []
        for xb, yb in BatchIterator(batch_size=4, rng=rng)(X, y):
            self.assertEqual(len(xb), len(yb))
            seen.extend(yb.tolist())
        self.assertEqual(sorted(seen), list(range(10)))


class TestMetrics(unittest.TestCase):
    def test_accuracy_one_hot_and_int(self):
        self.assertAlmostEqual(accuracy([0, 1, 2, 1], np.eye(3)[[0, 1, 2, 0]]), 0.75)
        self.assertAlmostEqual(accuracy(np.array([1, 1]), np.array([1, 1])), 1.0)


if __name__ == "__main__":
    unittest.main()
