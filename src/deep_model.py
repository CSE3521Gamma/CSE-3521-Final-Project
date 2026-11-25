"""Deep Model for M3 using feedforward neural network on TF-IDF features."""
import json, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold

from baseline_models import load, metrics
from paths import BASELINE_RESULTS

SEED = 44
EPOCHS = 10
BATCH = 256
torch.manual_seed(SEED)
np.random.seed(SEED)
dev = "cuda" if torch.cuda.is_available() else "cpu"

X, y_train, y_test, split, raw, cleaned, groups = load()
Xtr_all, Xte = X["tfidf"]

# Validation set carved from TRAINING data only, keeping duplicate reviews together
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
tr_idx, val_idx = next(cv.split(Xtr_all, y_train, groups))
Xtr, ytr = Xtr_all[tr_idx], y_train[tr_idx]
Xval, yval = Xtr_all[val_idx], y_train[val_idx]
print("train", Xtr.shape, "val", Xval.shape, "test", Xte.shape)


def batches(X, y, shuffle):
    idx = np.random.permutation(X.shape[0]) if shuffle else np.arange(X.shape[0])
    for i in range(0, len(idx), BATCH):
        b = idx[i:i + BATCH]
        xb = torch.tensor(X[b].toarray(), dtype=torch.float32, device=dev)
        yb = torch.tensor(y[b], dtype=torch.float32, device=dev)
        yield xb, yb


model = nn.Sequential(
    nn.Linear(Xtr.shape[1], 256), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(64, 1),
).to(dev)

# Downweight the majority class (Positive)
pos_weight = torch.tensor((ytr == 0).sum() / (ytr == 1).sum(), dtype=torch.float32, device=dev)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


def run_epoch(X, y, train):
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in batches(X, y, shuffle=train):
            out = model(xb).squeeze(1)
            loss = loss_fn(out, yb)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += loss.item() * len(yb)
            n += len(yb)
    return total / n


train_loss, val_loss = [], []
best_val, best_state = float("inf"), None
t0 = time.time()
for ep in range(1, EPOCHS + 1):
    tl = run_epoch(Xtr, ytr, train=True)
    vl = run_epoch(Xval, yval, train=False)
    train_loss.append(tl)
    val_loss.append(vl)
    if vl < best_val:
        best_val = vl
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    print(f"epoch {ep:2d}  train_loss={tl:.4f}  val_loss={vl:.4f}", flush=True)
train_time = time.time() - t0

# Use the weights from the epoch with the lowest validation loss
model.load_state_dict(best_state)

# Final evaluation on the test set
model.eval()
probs = []
with torch.no_grad():
    for xb, _ in batches(Xte, y_test, shuffle=False):
        probs.append(torch.sigmoid(model(xb).squeeze(1)).cpu().numpy())
probs = np.concatenate(probs)
y_pred = (probs >= 0.5).astype(int)

m = metrics(y_test, y_pred)
result = {
    "name": "Feedforward NN (TF-IDF)",
    "accuracy": m["accuracy"],
    "precision": m["neg_precision"],
    "recall": m["neg_recall"],
    "f1": m["neg_f1"],
    "macro_f1": m["macro_f1"],
    "train_time_sec": train_time,
    "confusion_matrix": m["confusion_matrix"],
    "train_loss": train_loss,
    "val_loss": val_loss,
}
with open(BASELINE_RESULTS / "deep_model_results.json", "w") as f:
    json.dump(result, f, indent=2, default=float)
torch.save(best_state, BASELINE_RESULTS / "deep_model.pt")

# Loss curve
plt.figure(figsize=(6, 4))
plt.plot(range(1, EPOCHS + 1), train_loss, marker="o", label="Train loss")
plt.plot(range(1, EPOCHS + 1), val_loss, marker="o", label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Feedforward NN loss per epoch")
plt.legend()
plt.tight_layout()
plt.savefig(BASELINE_RESULTS / "deep_loss_curve.png", dpi=200)

# Save per-review predictions for error analysis
import pandas as pd
pd.DataFrame({"Id": raw["Id"].values[split["test_idx"]],
              "Score": raw["Score"].values[split["test_idx"]],
              "prob_positive": probs.round(3),
              "correct": y_pred == y_test}
             ).to_csv(BASELINE_RESULTS / "deep_test_predictions.csv", index=False)

print(f"\nTest accuracy={m['accuracy']:.4f}  neg_f1={m['neg_f1']:.4f}  "
      f"macro_f1={m['macro_f1']:.4f}  train_time={train_time:.1f}s")