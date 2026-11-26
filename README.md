# AI-Powered Text Intelligence Platform (Sentiment Analysis of Amazon Food Reviews: From Data to Deployment)

**CSE 3521: Survey of Artificial Intelligence I (Autumn 2026) — Group Gamma**

## Overview
This project is an end-to-end text intelligence and sentiment classification system built on a 50,000-review sample (Seed `44`) of the Amazon Fine Food Reviews dataset. It compares classical machine learning baselines against a deep learning architecture trained on word embeddings. The final pipeline is deployed as an interactive web application via Streamlit Community Cloud, allowing users to input raw review text and compare live sentiment predictions across models.

**Live Application URL:** `[Streamlit Community Cloud URL put later]`

**Project Document URL:** [Link](https://buckeyemailosu-my.sharepoint.com/:w:/g/personal/talukder_13_osu_edu/IQBJ2naVpQrmQZyUDoeiQ0ENAfX4zLpeez-zbessNB1FSG0?e=oVOb4H)

## Team Members & Roles
* **Sid Guddanti** – ML Engineer 
* **Theo Ruppe** – App Engineer
* **Donte Beck-King** – Analyst / Writer & ML Engineer
* **Debapratim Talukder** – Data Engineer
* **Matthew Moloney** – Project Lead

## Repository Structure
```text
├── README.md                     # Project overview, team info, and how to run everything
├── requirements.txt              # Python libraries needed to run the code
├── app.py                        # Streamlit entry point (page navigation)
├── .streamlit/config.toml        # Streamlit theme and server settings
├── ui/
│   ├── predictor.py              # Loads the models and runs live predictions
│   ├── data.py                   # Cached loaders for the dataset and saved results
│   ├── theme.py                  # Shared chart colors (one per model, one per class)
│   └── pages/                    # Overview, Predict, Data, Model Comparison
├── data/
│   ├── raw/
│   │   └── group_sample_seed44.csv          # Original 50,000-review Seed 44 sample
│   └── processed/
│       ├── cleaned_reviews_seed44.csv       # Cleaned review text (Summary + Text) and labels
│       ├── shared_split_seed44.csv          # Train/test assignment for every review
│       └── split_indices_seed44.npz         # Same split as row indices (train_idx, test_idx)
├── features/
│   ├── X_train_bow.npz / X_test_bow.npz     # Bag of Words matrices (sparse)
│   ├── X_train_tfidf.npz / X_test_tfidf.npz # TF-IDF matrices (sparse)
│   ├── y_train.npy / y_test.npy             # Labels (1 = Positive, 0 = Negative/Neutral)
│   └── bow_vectorizer.joblib / tfidf_vectorizer.joblib  # Fitted vectorizers for new reviews
├── src/
│   ├── paths.py                  # Shared folder locations used by every script
│   ├── feature_engineering.py    # Cleans text, makes the 80/20 split, builds BoW and TF-IDF
│   ├── verify_outputs.py         # Checks that the split, labels, and features are correct
│   ├── baseline_models.py        # Trains and evaluates the classical baseline models
│   ├── deep_model.py             # Trains and evaluates the feedforward neural network
│   └── app_models.py             # Refits the tuned classical models for the app (models/)
└── results/
    ├── feature_engineering/      # Word frequency chart, chi-square top-10 features, summary JSON
    └── baseline/                 # Metrics table, confusion matrices, top-10 LR features,
                                  # and misclassified test reviews for error analysis
```

## Getting Started
Requires Python 3.10 or newer. Run all commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The processed data, feature matrices, and results are already included, so you can load them directly without rerunning anything:

```python
import numpy as np
from scipy.sparse import load_npz

X_train = load_npz("features/X_train_tfidf.npz")   # use "bow" instead of "tfidf" for Bag of Words
X_test = load_npz("features/X_test_tfidf.npz")
y_train = np.load("features/y_train.npy")
y_test = np.load("features/y_test.npy")
```

To reproduce everything from the raw data:

```bash
python src/feature_engineering.py   # writes data/processed/, features/, results/feature_engineering/
python src/verify_outputs.py        # sanity checks on the split, labels, and features
python src/baseline_models.py       # writes results/baseline/ (takes a few minutes because of KNN tuning)
```

## Running the App

```bash
streamlit run app.py
```

The app opens at http://localhost:8501 and has four pages:

* **Overview** – what the project does, how a prediction works, the model leaderboard, and the team.
* **Predict** – type a review and see Logistic Regression, Naive Bayes, and KNN classify it side by side, with the words that pushed each prediction.
* **Data** – class balance, star ratings, review lengths, the most telling words, and a filterable browser for all 50,000 reviews.
* **Model Comparison** – test-set metrics for every model (including the neural network), confusion matrices, the network's training curves, and the reviews Logistic Regression got wrong.

On the first run the app trains the tuned classical models from the saved features (under a minute) and caches them in `models/`, which is not committed. To build them ahead of time and check that their test scores match the M2 results, run `python src/app_models.py`.

The neural network appears on Model Comparison from its saved results. It is not offered for live predictions because its weights (`deep_model.pt`) are not committed and PyTorch is not in `requirements.txt`. To add it, save the weights, subclass `SentimentModel` in `ui/predictor.py`, and register it in `load_models()`.

**Deploying to Streamlit Community Cloud:** sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub, click **Create app**, pick this repository and the `main` branch, and set the main file path to `app.py`.

## Data and Features

The original dataset is available from [Amazon Fine Food Reviews on Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews).
The assigned 50,000-review sample used was generated with seed 44, saved as `data/raw/group_sample_seed44.csv`. This exact sample is required to reproduce our results.

* **Labels:** 4–5 stars = Positive (1), 1–3 stars = Negative/Neutral (0).
* **Class balance:** 38,786 Positive (77.6%) and 11,214 Negative/Neutral (22.4%).
* **Split:** 40,000 training and 10,000 testing reviews (80/20, seed 44, same class ratio). Reviews with identical text are always kept on the same side of the split to prevent data leakage.
* **Features:** Bag of Words and TF-IDF (unigrams, min_df = 3, max_df = 0.95, max 20,000 features), fitted on the training set only. Both have a vocabulary of 16,897 words.

## Baseline Results (Milestone 2)
Evaluated on the 10,000-review held-out test set. Precision, recall, and F1 are for the Negative/Neutral (minority) class. Hyperparameters were tuned with cross-validation on the training set only.

| Model | Accuracy | Precision | Recall | F1 | Macro F1 |
|---|---|---|---|---|---|
| Majority class (always Positive) | 0.776 | 0.000 | 0.000 | 0.000 | 0.437 |
| KNN (TF-IDF, K = 5) | 0.816 | 0.676 | 0.344 | 0.456 | 0.673 |
| Multinomial Naive Bayes (BoW) | 0.871 | 0.708 | 0.727 | 0.717 | 0.817 |
| **Logistic Regression (TF-IDF)** | **0.881** | **0.691** | **0.851** | **0.763** | **0.842** |

Full results, confusion matrices, and the top-10 most informative features are in [`results/`](results/).
