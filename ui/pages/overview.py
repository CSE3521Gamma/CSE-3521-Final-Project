import streamlit as st

st.title("About this project")

st.markdown("""
**AI-Powered Text Intelligence Platform** - CSE 3521: Survey of Artificial Intelligence I
(Autumn 2026), Group Gamma.

This app classifies Amazon Fine Food reviews as **Positive** (4-5 stars) or
**Negative/Neutral** (1-3 stars). The models were trained on 40,000 reviews from a
50,000-review sample (seed 44) and tested on the other 10,000.

#### How a prediction works
1. **Cleaning**: the title and review text are lowercased, HTML and links are removed, and
   contractions are expanded so negations survive (*didn't* becomes *did not*).
2. **Features**: the cleaned text becomes a Bag-of-Words count vector or a TF-IDF vector
   over a 16,897-word vocabulary learned from the training reviews.
3. **Classification**: each model turns that vector into a probability that the review is
   Positive; above 50% means Positive.

#### Models
| Model | Features | Test macro F1 |
|---|---|---|
| Logistic Regression | TF-IDF | 0.842 |
| Multinomial Naive Bayes | Bag of Words | 0.817 |
| K-Nearest Neighbors (K = 5) | TF-IDF | 0.673 |

A deep learning model trained on word embeddings will be added for the final comparison.

#### Limits
Words the models never saw in training are ignored, and the classical models read words
one at a time, so sarcasm and long-range negation ("not what I'd call good") can fool them.

#### Team
- Sid Guddanti - ML Engineer
- Theo Ruppe - App Engineer
- Donte Beck-King - Analyst / Writer & ML Engineer
- Debapratim Talukder - Data Engineer
- Matthew Moloney - Project Lead
""")
