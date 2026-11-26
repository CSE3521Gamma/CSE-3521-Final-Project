"""M3 model comparison (ML Engineer).

Combines the classical models (results/baseline/baseline_results.json, written
by baseline_models.py) with the feedforward neural network
(results/feature_engineering/deep_model_results_2.json, the final 5-epoch run
written by deep_model.py) into one comparison table, one confusion matrix per
model, and a loss-curve plot for both neural network runs.

Run from the repo root after baseline_models.py and deep_model.py:
    python src/model_comparison.py

Outputs go to results/comparison/.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from paths import BASELINE_RESULTS, FE_RESULTS, ROOT

OUT = ROOT / "results" / "comparison"
OUT.mkdir(parents=True, exist_ok=True)
LABELS = ["Negative/Neutral", "Positive"]
DEEP_RUNS = [("Run 1 (10 epochs)", "deep_model_results.json"),
             ("Run 2 (5 epochs, final)", "deep_model_results_2.json")]
DEEP_NAME = "Feedforward NN (TF-IDF)"


def load_deep(filename):
    for folder in (FE_RESULTS, BASELINE_RESULTS):
        path = folder / filename
        if path.exists():
            return json.loads(path.read_text())
    raise FileNotFoundError(f"{filename} not found in {FE_RESULTS} or {BASELINE_RESULTS}")


def main():
    classical = json.loads((BASELINE_RESULTS / "baseline_results.json").read_text())
    deep = load_deep(DEEP_RUNS[-1][1])

    rows, matrices = [], {}
    for name, r in classical.items():
        rows.append({"Model": name, "Accuracy": r["accuracy"],
                     "Precision (Neg)": r["neg_precision"], "Recall (Neg)": r["neg_recall"],
                     "F1 (Neg)": r["neg_f1"], "Macro F1": r["macro_f1"],
                     "Training time (s)": r["train_time_sec"]})
        matrices[name] = r["confusion_matrix"]
    rows.append({"Model": DEEP_NAME, "Accuracy": deep["accuracy"],
                 "Precision (Neg)": deep["precision"], "Recall (Neg)": deep["recall"],
                 "F1 (Neg)": deep["f1"], "Macro F1": deep["macro_f1"],
                 "Training time (s)": deep["train_time_sec"]})
    matrices[DEEP_NAME] = deep["confusion_matrix"]

    table = pd.DataFrame(rows)
    table.round(4).to_csv(OUT / "model_comparison.csv", index=False)
    print(table.round(4).to_string(index=False))

    # One confusion matrix per model
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (name, cm) in zip(axes.ravel(), matrices.items()):
        ConfusionMatrixDisplay(np.array(cm), display_labels=LABELS).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(name, fontsize=10)
    for ax in axes.ravel()[len(matrices):]:
        ax.axis("off")
    fig.suptitle("Confusion matrices on the 10,000-review test set")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrices_all_models.png", dpi=200)
    plt.close(fig)

    # Training vs. validation loss for both neural network runs
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (title, filename) in zip(axes, DEEP_RUNS):
        run = load_deep(filename)
        epochs = range(1, len(run["train_loss"]) + 1)
        ax.plot(epochs, run["train_loss"], marker="o", label="Training loss")
        ax.plot(epochs, run["val_loss"], marker="o", label="Validation loss")
        ax.set_title(f"Feedforward NN - {title}")
        ax.set_xlabel("Epoch")
        ax.set_xticks(list(epochs))
        ax.legend()
    axes[0].set_ylabel("Weighted binary cross-entropy loss")
    fig.tight_layout()
    fig.savefig(OUT / "deep_loss_curves.png", dpi=200)
    plt.close(fig)
    print("Saved comparison outputs to", OUT)


if __name__ == "__main__":
    main()
