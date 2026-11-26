"""Shared chart colors so every page uses the same color for the same thing."""

# One fixed color per model (never reassigned by rank or by filters).
MODEL_COLORS = {
    "Logistic Regression (TF-IDF)": "#2a78d6",   # blue
    "Multinomial Naive Bayes (BoW)": "#eb6834",  # orange
    "KNN (TF-IDF)": "#1baf7a",                   # aqua
    "Feedforward NN (TF-IDF)": "#eda100",        # yellow
    "Majority class (always Positive)": "#8a8985",  # neutral gray: reference baseline
}

# Sentiment polarity: two opposite poles.
POSITIVE = "#2a78d6"
NEGATIVE = "#e34948"
LABEL_COLORS = {"Positive": POSITIVE, "Negative/Neutral": NEGATIVE}


def model_scale(names):
    import altair as alt
    names = [n for n in MODEL_COLORS if n in set(names)]
    return alt.Scale(domain=names, range=[MODEL_COLORS[n] for n in names])


def label_scale():
    import altair as alt
    return alt.Scale(domain=list(LABEL_COLORS), range=list(LABEL_COLORS.values()))
