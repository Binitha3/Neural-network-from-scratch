"""Demo: watch the decision boundary form over training — as an animated GIF.

Run:  python scripts/toy_visualization.py

Trains a small MLP on the two-moons and two-spiral toy problems, snapshots
the decision boundary every N steps, and writes an animated GIF per dataset
to ``outputs/``. Also plots the loss curves. This is the visual that makes
the backprop work feel alive — great as the README hero image.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from nn import (Dense, ReLU, SGDMomentum, Sequential, SoftmaxCrossEntropy,
                make_moons, make_spirals, accuracy)

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)


def grid_points(X, h=0.02):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return np.c_[xx.ravel(), yy.ravel()], xx, yy


def train_with_snapshots(X, y, epochs=120, lr=0.1, snapshot_every=4, seed=0):
    rng = np.random.default_rng(seed)
    model = Sequential([
        Dense(2, 32, rng=rng), ReLU(),
        Dense(32, 32, rng=rng), ReLU(),
        Dense(32, 2, rng=rng),
    ])
    loss_fn = SoftmaxCrossEntropy()
    opt = SGDMomentum(model, lr=lr, momentum=0.9)

    snapshots, losses = [], []
    n = len(X)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, 32):
            idx = order[start:start + 32]
            epoch_loss += model.train_on_batch(
                X[idx], y[idx], loss_fn, opt) * len(idx)
        losses.append(epoch_loss / n)

        if epoch % snapshot_every == 0 or epoch == 1 or epoch == epochs:
            grid, xx, yy = grid_points(X)
            probs = model.predict(grid)[:, 1]
            acc = accuracy(model.predict(X)[:, 1] > 0.5, y)
            snapshots.append((epoch, probs.reshape(xx.shape), losses[-1], acc))
    return model, snapshots, losses


def save_gif(X, y, snapshots, grid_xx, grid_yy, title, path, fps=6):
    colors = np.array(["#3b82f6", "#f97316"])  # blue / orange
    frames = []
    for epoch, grid_probs, loss, acc in snapshots:
        fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=110)
        ax.contourf(grid_xx, grid_yy, grid_probs, levels=20, cmap="RdBu", alpha=0.75)
        ax.scatter(X[:, 0], X[:, 1], c=colors[y], edgecolors="k", linewidths=0.4, s=18)
        ax.set_title(f"{title}\nepoch {epoch}  |  loss {loss:.3f}  |  acc {acc:.1%}",
                     fontsize=10)
        ax.set_xlim(grid_xx.min(), grid_xx.max())
        ax.set_ylim(grid_yy.min(), grid_yy.max())
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)

    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
    print(f"saved {path}")


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    X, y = make_moons(400, noise=0.15, rng=rng)
    _, snapshots, losses = train_with_snapshots(X, y, epochs=120, lr=0.1)
    grid, grid_xx, grid_yy = grid_points(X)
    save_gif(X, y, snapshots, grid_xx, grid_yy,
             "Two moons — MLP decision boundary", OUT / "moons.gif")

    Xs, ys = make_spirals(400, classes=2, noise=0.08, turns=1.75, rng=rng)
    _, snapshots_s, losses_s = train_with_snapshots(Xs, ys, epochs=250, lr=0.05)
    grid_s, grid_xx_s, grid_yy_s = grid_points(Xs)
    save_gif(Xs, ys, snapshots_s, grid_xx_s, grid_yy_s,
             "Two spirals — MLP decision boundary", OUT / "spirals.gif")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=110)
    for ax, losses_, name in zip(axes, [losses, losses_s], ["two moons", "two spirals"]):
        ax.plot(losses_)
        ax.set_title(f"training loss — {name}")
        ax.set_xlabel("epoch"); ax.set_ylabel("softmax cross-entropy")
    fig.tight_layout()
    fig.savefig(OUT / "toy_loss_curves.png")
    print(f"saved {OUT / 'toy_loss_curves.png'}")
