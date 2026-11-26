"""Loads the models once per server process and runs predictions for the pages.

Every model in the app is a SentimentModel. To add a new one (for example the
deep learning model once its weights are saved), subclass SentimentModel,
implement prob_positive(), and add it to load_models(). The pages pick it up
automatically.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from app_models import SPECS, load_or_train  # noqa: E402
from feature_engineering import prepare_review  # noqa: E402
from paths import FEATURES  # noqa: E402

LABELS = {0: "Negative/Neutral", 1: "Positive"}


@dataclass
class SentimentModel:
    key: str
    name: str
    description: str

    def prob_positive(self, cleaned_text: str) -> float:
        """Probability that the (already cleaned) review is Positive."""
        raise NotImplementedError

    def word_contributions(self, cleaned_text: str) -> pd.DataFrame | None:
        """Per-word push toward Positive (+) or Negative/Neutral (-), if the model has one."""
        return None


@dataclass
class SklearnModel(SentimentModel):
    vectorizer: object = None
    model: object = None

    def prob_positive(self, cleaned_text):
        X = self.vectorizer.transform([cleaned_text])
        return float(self.model.predict_proba(X)[0, 1])

    def word_contributions(self, cleaned_text):
        if hasattr(self.model, "coef_"):  # Logistic Regression
            weights = self.model.coef_[0]
        elif hasattr(self.model, "feature_log_prob_"):  # Naive Bayes: log P(w|pos) - log P(w|neg)
            weights = self.model.feature_log_prob_[1] - self.model.feature_log_prob_[0]
        else:  # KNN has no per-word weights
            return None
        X = self.vectorizer.transform([cleaned_text]).tocsr()
        if X.nnz == 0:
            return None
        words = self.vectorizer.get_feature_names_out()[X.indices]
        return (pd.DataFrame({"word": words, "contribution": X.data * weights[X.indices]})
                .sort_values("contribution", key=np.abs, ascending=False)
                .reset_index(drop=True))


DESCRIPTIONS = {
    "logreg": "Weighs every word on TF-IDF features. Best classical model.",
    "nb": "Probabilistic word-count model on Bag-of-Words features.",
    "knn": "Votes among the 5 most similar training reviews (cosine similarity on TF-IDF).",
}


@st.cache_resource(show_spinner="Loading models (the first run trains them, about a minute)...")
def load_models() -> dict[str, SentimentModel]:
    vectorizers = {rep: joblib.load(FEATURES / f"{rep}_vectorizer.joblib")
                   for rep in ("bow", "tfidf")}
    return {key: SklearnModel(key=key, name=name, description=DESCRIPTIONS[key],
                              vectorizer=vectorizers[rep], model=load_or_train(key))
            for key, (name, rep, _) in SPECS.items()}


def analyze(summary: str, review: str, models: list[SentimentModel]) -> tuple[str, pd.DataFrame]:
    """Clean the input the same way as the training data and score it with each model."""
    cleaned = prepare_review(summary, review)
    rows = []
    for m in models:
        p = m.prob_positive(cleaned)
        rows.append({"key": m.key, "Model": m.name, "p_positive": p,
                     "Prediction": LABELS[int(p > 0.5)]})  # same tie rule as model.predict()
    return cleaned, pd.DataFrame(rows)
