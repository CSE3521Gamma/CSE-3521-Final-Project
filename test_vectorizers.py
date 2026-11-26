
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from feature_engineering import prepare_review

text = prepare_review("Great product", "I really liked the taste.")

for name in ["bow", "tfidf"]:
    vectorizer = joblib.load(f"features/{name}_vectorizer.joblib")
    features = vectorizer.transform([text])
    print(f"{name.upper()}: {features.shape}")
