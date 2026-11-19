"""Model layer for the app (placeholder).

TODO: load the trained models here and return real predictions. Each model
should expose prob_positive(text) -> probability the review is Positive.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LABELS = {0: "Negative/Neutral", 1: "Positive"}

# Models the app will offer once they are wired up.
MODEL_NAMES = [
    "Logistic Regression (TF-IDF)",
    "Multinomial Naive Bayes (BoW)",
    "KNN (TF-IDF)",
    "Deep Learning (word embeddings)",
]


def analyze(summary: str, review: str, model_names: list[str]):
    """TODO: clean the text, run each model, and return the predictions."""
    raise NotImplementedError("Models are not connected yet.")
