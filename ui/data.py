"""Cached loaders for the dataset and saved results."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from paths import BASELINE_RESULTS, FE_RESULTS, PROCESSED, RAW_DATA  # noqa: E402

LABELS = ["Negative/Neutral", "Positive"]  # 0, 1


@st.cache_data(show_spinner="Loading reviews...")
def load_reviews() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA, usecols=["Id", "Score", "Summary", "Text", "Time"])
    split = pd.read_csv(PROCESSED / "shared_split_seed44.csv", usecols=["row_index", "split"])
    df["split"] = split.sort_values("row_index")["split"].to_numpy()
    df["Text"] = df["Text"].str.replace(r"<br\s*/?>", " ", regex=True)  # raw HTML line breaks
    df["label"] = np.where(df["Score"] >= 4, "Positive", "Negative/Neutral")
    df["words"] = df["Text"].fillna("").str.split().str.len()
    df["year"] = pd.to_datetime(df["Time"], unit="s").dt.year
    return df.drop(columns="Time")


def _scores_from_confusion(cm):
    """All metrics from a 2x2 confusion matrix [[neg->neg, neg->pos], [pos->neg, pos->pos]]."""
    (nn, np_), (pn, pp) = cm
    div = lambda a, b: a / b if b else 0.0  # noqa: E731
    neg_p, neg_r = div(nn, nn + pn), div(nn, nn + np_)
    pos_p, pos_r = div(pp, pp + np_), div(pp, pp + pn)
    neg_f1 = div(2 * neg_p * neg_r, neg_p + neg_r)
    pos_f1 = div(2 * pos_p * pos_r, pos_p + pos_r)
    return {
        "Accuracy": div(nn + pp, nn + np_ + pn + pp),
        "Neg Precision": neg_p, "Neg Recall": neg_r, "Neg F1": neg_f1,
        "Pos Precision": pos_p, "Pos Recall": pos_r, "Pos F1": pos_f1,
        "Macro F1": (neg_f1 + pos_f1) / 2,
    }


@st.cache_data
def load_deep_runs() -> list[dict]:
    """Every saved deep model run, oldest first (deep_model.py may write to either folder)."""
    files = sorted({*FE_RESULTS.glob("deep_model_results*.json"),
                    *BASELINE_RESULTS.glob("deep_model_results*.json")},
                   key=lambda p: (p.stem.count("_"), p.stem))
    runs = []
    for i, path in enumerate(files, 1):
        run = json.loads(path.read_text())
        run["run"] = f"Run {i} ({len(run.get('train_loss', []))} epochs)"
        run["file"] = path.name
        runs.append(run)
    return runs


@st.cache_data
def load_model_results() -> pd.DataFrame:
    """One row per model: test metrics (computed from its confusion matrix) and details."""
    baseline = json.loads((BASELINE_RESULTS / "baseline_results.json").read_text())
    rows = []
    for name, r in baseline.items():
        kind = "Baseline" if name.startswith("Majority") else "Classical ML"
        rows.append({"Model": name, "Type": kind, "Features": r["features"].upper().replace("TFIDF", "TF-IDF"),
                     "Settings": ", ".join(f"{k}={v}" for k, v in r["best_params"].items()) or "-",
                     "confusion_matrix": r["confusion_matrix"],
                     **_scores_from_confusion(r["confusion_matrix"])})
    runs = load_deep_runs()
    if runs:
        deep = runs[-1]  # latest run
        rows.append({"Model": deep["name"], "Type": "Deep learning", "Features": "TF-IDF",
                     "Settings": f"256-64 hidden units, {deep['run'].split('(')[1].rstrip(')')}",
                     "confusion_matrix": deep["confusion_matrix"],
                     **_scores_from_confusion(deep["confusion_matrix"])})
    return pd.DataFrame(rows)


@st.cache_data
def load_top_words() -> pd.DataFrame:
    t = pd.read_csv(FE_RESULTS / "tfidf_top10.csv")
    pos = pd.DataFrame({"word": t["positive_word"], "chi2": t["positive_chi2"], "label": "Positive"})
    neg = pd.DataFrame({"word": t["negative_neutral_word"], "chi2": t["negative_neutral_chi2"],
                        "label": "Negative/Neutral"})
    return pd.concat([pos, neg], ignore_index=True)


@st.cache_data
def load_misclassified() -> pd.DataFrame:
    return pd.read_csv(BASELINE_RESULTS / "misclassified_test_reviews.csv")
