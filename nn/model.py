"""The Sequential model: forward, backward, and parameter plumbing.

The forward/backward pair is the whole backpropagation algorithm in
miniature:

  forward :  x_0 -> [layer_1.forward] -> ... -> [layer_n.forward] -> y
  backward:  dL/dy -> [layer_n.backward] -> ... -> [layer_1.backward] -> dL/dx_0

Each ``backward`` call applies the chain rule for one op and hands the result
to the previous layer. All bookkeeping (what to cache, how to differentiate)
was decided when each layer was written — there is no autograd graph, no
dynamic tracing. That is the point of building this from scratch.
"""

import numpy as np

from .utils import accuracy


class Sequential:
    def __init__(self, layers):
        self.layers = list(layers)
        self.rng = np.random.default_rng()
        for layer in self.layers:
            if hasattr(layer, "rng") and isinstance(getattr(layer, "rng"), np.random.Generator):
                layer.rng = self.rng

    # ------------------------------------------------------------------ #
    # core algorithm                                                      #
    # ------------------------------------------------------------------ #
    def forward(self, x, training=True):
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    # ------------------------------------------------------------------ #
    # training                                                            #
    # ------------------------------------------------------------------ #
    def train_on_batch(self, x, y, loss_fn, optimizer):
        """One optimisation step; returns the loss for this batch."""
        logits = self.forward(x, training=True)
        loss = loss_fn.forward(logits, y)
        optimizer.zero_grad()
        self.backward(loss_fn.backward())
        optimizer.step()
        return loss

    def fit(self, X, y, epochs, batch_size, loss_fn, optimizer,
            X_val=None, y_val=None, verbose=True):
        """Full training loop with optional validation each epoch."""
        history = []
        n = X.shape[0]
        for epoch in range(1, epochs + 1):
            order = self.rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                epoch_loss += self.train_on_batch(
                    X[idx], y[idx], loss_fn, optimizer) * len(idx)
            epoch_loss /= n
            row = {"epoch": epoch, "loss": epoch_loss}

            if X_val is not None:
                val_loss = loss_fn.forward(self.predict(X_val), y_val)
                row["val_loss"] = float(val_loss)
                row["val_acc"] = accuracy(
                    loss_fn.predicted_classes(), y_val if y_val.ndim == 2 else y_val)

            history.append(row)
            if verbose:
                msg = f"epoch {epoch:>3}/{epochs}  loss {row['loss']:.4f}"
                if "val_loss" in row:
                    msg += f"  val_loss {row['val_loss']:.4f}  val_acc {row['val_acc']:.4f}"
                print(msg)
        return history

    def predict(self, X):
        return self.forward(X, training=False)

    # ------------------------------------------------------------------ #
    # parameter plumbing for optimisers                                   #
    # ------------------------------------------------------------------ #
    def params_and_grads(self):
        for layer in self.layers:
            params = layer.params()
            grads = layer.grads()
            for name in params:
                yield params[name], grads[name]

    def parameters_count(self):
        return sum(p.size for p, _ in self.params_and_grads())
