from pathlib import Path
import json
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.feature_selection import chi2

from paths import FE_RESULTS, FEATURES, PROCESSED, RAW_DATA

SEED = 44
DATA_FILE = RAW_DATA


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Keep negation words instead of splitting contractions into fragments.
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"\bain't\b", "is not", text)
    text = re.sub(r"n't\b", " not", text)
    for short, full in [("'re", " are"), ("'ve", " have"),
                        ("'ll", " will"), ("'m", " am"), ("'d", " would")]:
        text = re.sub(re.escape(short) + r"\b", full, text)
    return re.sub(r"\s+", " ", text).strip()


def prepare_review(summary, review):
    return (clean_text(summary) + " " + clean_text(review)).strip()


def split_data(df, bodies):
    # Reviews with the same text must stay on the same side of the split.
    group_id, _ = pd.factorize(bodies, sort=False)
    groups = pd.DataFrame({"group": group_id, "label": df["label"]})
    groups = groups.groupby("group").agg(
        size=("label", "size"),
        label=("label", "first"),
        label_count=("label", "nunique"),
    )
    if (groups["label_count"] != 1).any():
        raise ValueError("Identical reviews with different labels need to be checked.")

    rng = np.random.default_rng(SEED)
    sizes = groups["size"].to_numpy()
    labels = groups["label"].to_numpy()
    test_groups = np.zeros(len(groups), dtype=bool)

    # Choose 20% of each class while keeping duplicate groups together.
    for label in (0, 1):
        choices = np.flatnonzero(labels == label)
        rng.shuffle(choices)
        remaining = round((df["label"] == label).sum() * 0.20)
        for group in choices:
            if sizes[group] <= remaining:
                test_groups[group] = True
                remaining -= sizes[group]
            if remaining == 0:
                break
        if remaining:
            raise ValueError("Could not make the requested split.")

    test_mask = test_groups[group_id]
    train_idx = rng.permutation(np.flatnonzero(~test_mask))
    test_idx = rng.permutation(np.flatnonzero(test_mask))
    assert (len(train_idx), len(test_idx)) == (40000, 10000)
    assert not (set(bodies.iloc[train_idx]) & set(bodies.iloc[test_idx]))
    return train_idx, test_idx


def top_features(X, y, words, bow=False):
    values = X.copy() if bow else X
    if bow:
        values.data[:] = 1  # Use whether a word appears, not its raw count.

    scores, _ = chi2(values, y)
    avg_pos = np.asarray(values[y == 1].mean(axis=0)).ravel()
    avg_neg = np.asarray(values[y == 0].mean(axis=0)).ravel()

    # Chi-square gives association strength; class averages give its direction.
    result = {}
    for name, allowed in [("positive", avg_pos > avg_neg),
                          ("negative_neutral", avg_neg > avg_pos)]:
        candidates = np.flatnonzero(allowed & np.isfinite(scores))
        chosen = candidates[np.argsort(-scores[candidates], kind="stable")[:10]]
        result[name] = [
            {"term": str(words[i]), "chi2": round(float(scores[i]), 3)}
            for i in chosen
        ]
    return result


def plot_frequent_words(X, words):
    counts = np.asarray(X.sum(axis=0)).ravel()
    ignored = set(ENGLISH_STOP_WORDS) - {"no", "nor", "not", "never"}
    kept = np.array([i for i, word in enumerate(words) if word not in ignored])
    top = kept[np.argsort(counts[kept])[-20:][::-1]]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(words[top][::-1], counts[top][::-1])
    ax.set_title("Most common words in the training reviews")
    ax.set_xlabel("Number of occurrences")
    fig.tight_layout()
    fig.savefig(FE_RESULTS / "word_frequency_train.png", dpi=180)
    plt.close(fig)


def plot_tfidf_terms(features):
    positive = features["positive"]
    negative = features["negative_neutral"]
    names = [f"{x['term']} (neg./neutral)" for x in negative[::-1]]
    names += [f"{x['term']} (positive)" for x in positive]
    values = [-x["chi2"] for x in negative[::-1]]
    values += [x["chi2"] for x in positive]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(names, values)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Words most associated with each sentiment (TF-IDF)")
    ax.set_xlabel("Chi-square score (left = negative/neutral, right = positive)")
    fig.tight_layout()
    fig.savefig(FE_RESULTS / "tfidf_informative_features.png", dpi=180)
    plt.close(fig)


def save_feature_table(name, features):
    positive = features["positive"]
    negative = features["negative_neutral"]
    pd.DataFrame({
        "positive_word": [row["term"] for row in positive],
        "positive_chi2": [row["chi2"] for row in positive],
        "negative_neutral_word": [row["term"] for row in negative],
        "negative_neutral_chi2": [row["chi2"] for row in negative],
    }).to_csv(FE_RESULTS / f"{name}_top10.csv", index=False)


def main():
    df = pd.read_csv(DATA_FILE)
    if len(df) != 50000 or not {"Id", "Score", "Summary", "Text"}.issubset(df.columns):
        raise ValueError("Expected the original 50,000-row Seed 44 review CSV.")

    # M1 labels: 4-5 stars = positive, 1-3 = negative/neutral.
    df["label"] = (df["Score"] >= 4).astype(int)
    bodies = df["Text"].map(clean_text)
    review_text = (df["Summary"].map(clean_text) + " " + bodies).str.strip()
    if (bodies.str.len() == 0).any():
        raise ValueError("One or more reviews have empty text.")

    train_idx, test_idx = split_data(df, bodies)
    y = df["label"].to_numpy()
    y_train, y_test = y[train_idx], y[test_idx]
    train_text = review_text.iloc[train_idx]
    test_text = review_text.iloc[test_idx]

    np.save(FEATURES / "y_train.npy", y_train)
    np.save(FEATURES / "y_test.npy", y_test)
    np.savez_compressed(PROCESSED / "split_indices_seed44.npz",
                        train_idx=train_idx, test_idx=test_idx)

    split_info = df[["Id", "Score", "label"]].copy()
    split_info["row_index"] = np.arange(len(df))
    split_info["split"] = "train"
    split_info.loc[test_idx, "split"] = "test"
    split_info.to_csv(PROCESSED / "shared_split_seed44.csv", index=False)
    pd.DataFrame({"Id": df["Id"], "clean_text": review_text,
                  "label": y}).to_csv(PROCESSED / "cleaned_reviews_seed44.csv", index=False)

    # Fit each vectorizer on training reviews only.
    settings = dict(lowercase=False, strip_accents="unicode", ngram_range=(1, 1),
                    min_df=3, max_df=0.95, max_features=20000,
                    token_pattern=r"(?u)\b\w\w+\b")
    vectorizers = {
        "bow": CountVectorizer(**settings),
        "tfidf": TfidfVectorizer(**settings, sublinear_tf=True, norm="l2"),
    }

    results = {}
    for name, vectorizer in vectorizers.items():
        X_train = vectorizer.fit_transform(train_text)
        X_test = vectorizer.transform(test_text)
        save_npz(FEATURES / f"X_train_{name}.npz", X_train)
        save_npz(FEATURES / f"X_test_{name}.npz", X_test)
        joblib.dump(vectorizer, FEATURES / f"{name}_vectorizer.joblib")

        words = vectorizer.get_feature_names_out()
        features = top_features(X_train, y_train, words, bow=(name == "bow"))
        save_feature_table(name, features)
        results[name] = {"vocab_size": len(words), "top_features": features}
        print(f"{name.upper()}: train {X_train.shape}, test {X_test.shape}")

        if name == "bow":
            plot_frequent_words(X_train, words)
        else:
            plot_tfidf_terms(features)

    summary = {
        "seed": SEED,
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "train_labels": dict(zip(["negative_neutral", "positive"], np.bincount(y_train).tolist())),
        "test_labels": dict(zip(["negative_neutral", "positive"], np.bincount(y_test).tolist())),
        "repeated_review_bodies": int(len(bodies) - bodies.nunique()),
        "representations": results,
    }
    (FE_RESULTS / "feature_engineering_results.json").write_text(json.dumps(summary, indent=2))
    print("Saved features to", FEATURES, "and results to", FE_RESULTS)


if __name__ == "__main__":
    main()
