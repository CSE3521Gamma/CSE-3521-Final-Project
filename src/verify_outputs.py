from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from feature_engineering import clean_text, prepare_review

from paths import FEATURES, PROCESSED, RAW_DATA

df = pd.read_csv(RAW_DATA)
idx = np.load(PROCESSED / "split_indices_seed44.npz")
train, test = idx["train_idx"], idx["test_idx"]
assert len(train) == 40000 and len(test) == 10000
assert np.array_equal(np.sort(np.r_[train, test]), np.arange(50000))

labels = (df["Score"] >= 4).astype(int).to_numpy()
assert np.array_equal(np.load(FEATURES / "y_train.npy"), labels[train])
assert np.array_equal(np.load(FEATURES / "y_test.npy"), labels[test])

bodies = df["Text"].map(clean_text)
assert not (set(bodies.iloc[train]) & set(bodies.iloc[test]))

for name in ("bow", "tfidf"):
    vec = joblib.load(FEATURES / f"{name}_vectorizer.joblib")
    X_train = load_npz(FEATURES / f"X_train_{name}.npz")
    X_test = load_npz(FEATURES / f"X_test_{name}.npz")
    assert X_train.shape == (40000, len(vec.vocabulary_))
    assert X_test.shape == (10000, len(vec.vocabulary_))

    for X, rows in [(X_train, train[:10]), (X_test, test[:10])]:
        texts = [prepare_review(df["Summary"].iloc[i], df["Text"].iloc[i]) for i in rows]
        difference = X[:10] - vec.transform(texts)
        assert np.allclose(difference.data, 0, atol=1e-12)

    print(f"{name.upper()} checked: {X_train.shape[1]} features")

print("Split, labels, and saved features checked successfully.")
