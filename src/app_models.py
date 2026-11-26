"""Fitted models served by the Streamlit app.

baseline_models.py tunes and evaluates the classical models but does not save
them. This module refits the tuned models (hyperparameters read from
results/baseline/baseline_results.json) on the saved training features and
stores them in models/ so the app can load them quickly.

The app calls load_or_train(), which trains a model the first time it is
needed if models/ does not have it yet, so nothing in models/ has to be
committed. Run this file to (re)build every model and check that its test
scores match the M2 baseline results:

    python src/app_models.py
"""

import json

import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from paths import BASELINE_RESULTS, FEATURES, MODELS

# key -> (name in baseline_results.json, feature set, model builder)
SPECS = {
    "logreg": ("Logistic Regression (TF-IDF)", "tfidf",
               lambda p: LogisticRegression(max_iter=2000, class_weight="balanced", **p)),
    "nb": ("Multinomial Naive Bayes (BoW)", "bow",
           lambda p: MultinomialNB(**p)),
    "knn": ("KNN (TF-IDF)", "tfidf",
            lambda p: KNeighborsClassifier(metric="cosine", algorithm="brute", **p)),
}


def baseline_results():
    return json.loads((BASELINE_RESULTS / "baseline_results.json").read_text())


def train(key):
    name, rep, build = SPECS[key]
    model = build(baseline_results()[name]["best_params"])
    model.fit(load_npz(FEATURES / f"X_train_{rep}.npz"), np.load(FEATURES / "y_train.npy"))
    return model


def model_path(key):
    return MODELS / f"{key}.joblib"


def load_or_train(key, save=True):
    path = model_path(key)
    if path.exists():
        try:
            return joblib.load(path)
        except Exception:
            pass  # e.g. saved with a different scikit-learn version: retrain below
    model = train(key)
    if save:
        try:
            joblib.dump(model, path, compress=3)
        except OSError:
            pass  # read-only file system: the model still works from memory
    return model


def main():
    results = baseline_results()
    y_test = np.load(FEATURES / "y_test.npy")
    for key, (name, rep, _) in SPECS.items():
        model = train(key)
        joblib.dump(model, model_path(key), compress=3)
        y_pred = model.predict(load_npz(FEATURES / f"X_test_{rep}.npz"))
        acc, macro = accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average="macro")
        expected = results[name]["macro_f1"]
        status = "OK" if abs(macro - expected) < 1e-3 else f"MISMATCH (expected {expected:.4f})"
        print(f"{name:32s} acc={acc:.4f} macro_f1={macro:.4f} {status}")
    print("Saved models to", MODELS)


if __name__ == "__main__":
    main()
