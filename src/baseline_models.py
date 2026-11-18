"""M2 baseline results (ML Engineer).

Trains classical models on Deb's saved BoW / TF-IDF matrices and reports
accuracy, precision, recall, and F1 on the 20% held-out test set, compared
against a majority-class baseline.

Models
  - Majority class      (always predicts Positive)
  - KNN on TF-IDF        (the model we planned in M1)
  - Multinomial NB on BoW (Lesson 9)
  - Logistic Regression on TF-IDF (also gives the top-10 informative words)

Hyperparameters are tuned with 5-fold cross-validation on the TRAINING set
only. Folds are grouped by review body so duplicate reviews never sit on both
sides of a fold (same rule Deb used for the train/test split). The test set
is only touched once, for the final numbers.

Run from the repo root:
    python src/baseline_models.py
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from feature_engineering import clean_text
from paths import BASELINE_RESULTS, FEATURES, PROCESSED, RAW_DATA

SEED = 44
OUT = BASELINE_RESULTS
LABELS = ["Negative/Neutral", "Positive"]  # 0, 1


def load():
    X = {k: (load_npz(FEATURES / f"X_train_{k}.npz"), load_npz(FEATURES / f"X_test_{k}.npz"))
         for k in ("bow", "tfidf")}
    y_train = np.load(FEATURES / "y_train.npy")
    y_test = np.load(FEATURES / "y_test.npy")
    split = np.load(PROCESSED / "split_indices_seed44.npz")
    raw = pd.read_csv(RAW_DATA)
    cleaned = pd.read_csv(PROCESSED / "cleaned_reviews_seed44.csv")
    # Group id = cleaned review body, so CV folds keep duplicates together.
    bodies = raw["Text"].map(clean_text)
    groups = pd.factorize(bodies)[0][split["train_idx"]]
    return X, y_train, y_test, split, raw, cleaned, groups


def metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1],
                                                 zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "neg_precision": p[0], "neg_recall": r[0], "neg_f1": f[0],
        "pos_precision": p[1], "pos_recall": r[1], "pos_f1": f[1],
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def tune(model, grid, X, y, groups, folds=5, jobs=-1):
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    search = GridSearchCV(model, grid, scoring="f1_macro", cv=cv, n_jobs=jobs)
    search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_params_, search.best_score_


def main():
    X, y_train, y_test, split, raw, cleaned, groups = load()
    Xb_tr, Xb_te = X["bow"]
    Xt_tr, Xt_te = X["tfidf"]

    specs = [
        ("Majority class (always Positive)", "bow",
         DummyClassifier(strategy="most_frequent"), None),
        ("KNN (TF-IDF)", "tfidf",
         KNeighborsClassifier(metric="cosine", algorithm="brute"),
         {"n_neighbors": [5, 15, 35], "weights": ["uniform", "distance"]}),
        ("Multinomial Naive Bayes (BoW)", "bow",
         MultinomialNB(), {"alpha": [0.1, 0.3, 1.0]}),
        ("Logistic Regression (TF-IDF)", "tfidf",
         LogisticRegression(max_iter=2000, class_weight="balanced"),
         {"C": [0.5, 1, 2, 4, 8]}),
    ]

    results, preds, fitted = {}, {}, {}
    for name, rep, model, grid in specs:
        Xtr, Xte = (Xb_tr, Xb_te) if rep == "bow" else (Xt_tr, Xt_te)
        if grid:
            # KNN builds large distance matrices, so use fewer folds/workers.
            is_knn = isinstance(model, KNeighborsClassifier)
            model, params, cv_f1 = tune(model, grid, Xtr, y_train, groups,
                                        folds=3 if is_knn else 5,
                                        jobs=2 if is_knn else -1)
        else:
            model.fit(Xtr, y_train)
            params, cv_f1 = {}, None
        y_pred = model.predict(Xte)
        results[name] = {"features": rep, "best_params": params,
                         "cv_macro_f1": cv_f1, **metrics(y_test, y_pred)}
        preds[name], fitted[name] = y_pred, model
        print(f"{name:35s} acc={results[name]['accuracy']:.4f} "
              f"neg_f1={results[name]['neg_f1']:.4f} "
              f"macro_f1={results[name]['macro_f1']:.4f} {params}", flush=True)

    # Results table
    rows = []
    for name, r in results.items():
        rows.append({"Model": name, "Accuracy": r["accuracy"],
                     "Neg Precision": r["neg_precision"], "Neg Recall": r["neg_recall"],
                     "Neg F1": r["neg_f1"], "Pos Precision": r["pos_precision"],
                     "Pos Recall": r["pos_recall"], "Pos F1": r["pos_f1"],
                     "Macro F1": r["macro_f1"]})
    pd.DataFrame(rows).round(4).to_csv(OUT / "baseline_results.csv", index=False)
    with open(OUT / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    # Confusion matrices
    names = list(results)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    for ax, name in zip(axes.ravel(), names):
        ConfusionMatrixDisplay(np.array(results[name]["confusion_matrix"]),
                               display_labels=LABELS).plot(ax=ax, cmap="Blues",
                                                           colorbar=False, values_format="d")
        ax.set_title(name, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrices_all.png", dpi=200)
    plt.close(fig)

    best = max((n for n in names if not n.startswith("Majority")),
               key=lambda n: results[n]["macro_f1"])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(np.array(results[best]["confusion_matrix"]),
                           display_labels=LABELS).plot(ax=ax, cmap="Blues",
                                                       colorbar=False, values_format="d")
    ax.set_title(f"{best} - test set (n=10,000)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix_best.png", dpi=200)
    plt.close(fig)

    # Top-10 informative words from the Logistic Regression weights
    lr = fitted["Logistic Regression (TF-IDF)"]
    vocab = joblib.load(FEATURES / "tfidf_vectorizer.joblib").get_feature_names_out()
    coef = lr.coef_[0]
    order = np.argsort(coef)
    top = pd.DataFrame({
        "Rank": range(1, 11),
        "Positive": vocab[order[::-1][:10]],
        "Positive weight": coef[order[::-1][:10]].round(3),
        "Negative/Neutral": vocab[order[:10]],
        "Negative/Neutral weight": coef[order[:10]].round(3),
    })
    top.to_csv(OUT / "logreg_top10_features.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].barh(top["Positive"][::-1], top["Positive weight"][::-1], color="#2e7d32")
    axes[0].set_title("Top 10 Positive words")
    axes[1].barh(top["Negative/Neutral"][::-1], -top["Negative/Neutral weight"][::-1],
                 color="#c62828")
    axes[1].set_title("Top 10 Negative/Neutral words")
    for ax in axes:
        ax.set_xlabel("|Logistic Regression weight|")
    fig.suptitle("Most informative words - Logistic Regression (TF-IDF)")
    fig.tight_layout()
    fig.savefig(OUT / "logreg_top10_features.png", dpi=200)
    plt.close(fig)

    # Misclassified test reviews for error analysis (best model)
    test_idx = split["test_idx"]
    wrong = np.flatnonzero(preds[best] != y_test)
    proba = fitted[best].predict_proba(
        (Xb_te if results[best]["features"] == "bow" else Xt_te)[wrong])[:, 1]
    mis = pd.DataFrame({
        "Id": raw["Id"].values[test_idx[wrong]],
        "Score": raw["Score"].values[test_idx[wrong]],
        "true_label": [LABELS[v] for v in y_test[wrong]],
        "predicted": [LABELS[v] for v in preds[best][wrong]],
        "prob_positive": proba.round(3),
        "word_count": [len(t.split()) for t in cleaned["clean_text"].values[test_idx[wrong]]],
        "clean_text": cleaned["clean_text"].values[test_idx[wrong]],
    })
    mis.to_csv(OUT / "misclassified_test_reviews.csv", index=False)
    print(f"\nBest model: {best}; {len(mis)} misclassified test reviews saved.")


if __name__ == "__main__":
    main()
