"""Train an MLP on MNIST using only NumPy — the flagship result of this repo.

Run:  python scripts/train_mnist.py

784 -> 128 -> 64 -> 10 MLP, softmax cross-entropy, Adam. Trains for a few
epochs and writes a training-dynamics figure (loss + validation accuracy)
and a confusion matrix to ``outputs/``. On a laptop CPU this reaches
~97-98% test accuracy in under two minutes.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nn import (Adam, Dense, Dropout, ReLU, Sequential, SoftmaxCrossEntropy,
                accuracy, load_mnist, set_seed)

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)


def confusion_matrix(y_true, y_pred, k=10):
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def main(epochs=8, batch_size=64, lr=1e-3):
    set_seed(42)
    X_train, y_train, X_test, y_test = load_mnist()

    model = Sequential([
        Dense(784, 128), ReLU(),
        Dropout(0.1),
        Dense(128, 64), ReLU(),
        Dropout(0.1),
        Dense(64, 10),
    ])
    loss_fn = SoftmaxCrossEntropy()
    opt = Adam(model, lr=lr)

    print(f"MLP 784-128-64-10  |  {model.parameters_count():,} parameters  "
          f"|  {epochs} epochs, batch {batch_size}, Adam lr={lr}")
    start = time.time()

    # --- manual training loop so we can also track test metrics per epoch --
    history = []
    rng = np.random.default_rng(42)
    n = len(X_train)
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        epoch_loss = 0.0
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            epoch_loss += model.train_on_batch(
                X_train[idx], y_train[idx], loss_fn, opt) * len(idx)
        train_loss = epoch_loss / n

        test_logits = model.predict(X_test)
        test_loss = loss_fn.forward(test_logits, y_test)
        test_acc = accuracy(loss_fn.predicted_classes(), y_test)

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "test_loss": test_loss, "test_acc": test_acc})
        print(f"epoch {epoch}/{epochs}  train_loss {train_loss:.4f}  "
              f"test_loss {test_loss:.4f}  test_acc {test_acc:.4f}")

    elapsed = time.time() - start
    print(f"\ndone in {elapsed:.1f}s  |  final test accuracy: "
          f"{history[-1]['test_acc']:.2%}")

    # --- figures --------------------------------------------------------- #
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=110)
    axes[0].plot(ep, [h["train_loss"] for h in history], label="train")
    axes[0].plot(ep, [h["test_loss"] for h in history], label="test")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("softmax cross-entropy")
    axes[0].set_title("MNIST loss"); axes[0].legend()
    axes[1].plot(ep, [h["test_acc"] for h in history], color="#16a34a")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
    axes[1].set_title("test accuracy")
    fig.tight_layout()
    fig.savefig(OUT / "mnist_training.png")
    print(f"saved {OUT / 'mnist_training.png'}")

    cm = confusion_matrix(y_test, loss_fn.predicted_classes())
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=110)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"MNIST confusion matrix (acc {history[-1]['test_acc']:.2%})")
    for i in range(10):
        for j in range(10):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, shrink=0.85)
    fig.tight_layout()
    fig.savefig(OUT / "mnist_confusion_matrix.png")
    print(f"saved {OUT / 'mnist_confusion_matrix.png'}")

    with open(OUT / "mnist_results.json", "w") as f:
        json.dump({"final_test_accuracy": history[-1]["test_acc"],
                   "epochs": epochs, "batch_size": batch_size, "lr": lr,
                   "parameters": model.parameters_count(),
                   "seconds": round(elapsed, 1), "history": history}, f, indent=2)
    print(f"saved {OUT / 'mnist_results.json'}")


if __name__ == "__main__":
    main()
