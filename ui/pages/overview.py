import streamlit as st

from ui.data import load_model_results

results = load_model_results()
ranked = results[results["Type"] != "Baseline"].sort_values("Macro F1", ascending=False)
best = ranked.iloc[0]

st.title("Review Sentiment Analyzer")
st.markdown("An end-to-end text classification system built on **50,000 Amazon Fine Food "
            "Reviews**. It predicts whether a review is **Positive** (4-5 stars) or "
            "**Negative/Neutral** (1-3 stars), and compares classical machine learning "
            "models against a neural network.")

c1, c2, c3, c4 = st.columns([1, 1, 1.7, 1])
c1.metric("Reviews", "50,000", border=True)
c2.metric("Models compared", len(ranked), border=True)
c3.metric("Best model", best["Model"].split(" (")[0], border=True)
c4.metric("Best macro F1", f"{best['Macro F1']:.3f}", border=True)

st.subheader("Explore")
cols = st.columns(3)
with cols[0].container(border=True, height="stretch"):
    st.markdown("**:material/rate_review: Predict**")
    st.caption("Type a review and see at least two models classify it side by side, with the "
               "words that drove each prediction.")
    st.page_link("ui/pages/predict.py", label="Try it", icon=":material/arrow_forward:")
with cols[1].container(border=True, height="stretch"):
    st.markdown("**:material/table_chart: Data**")
    st.caption("Class balance, star ratings, review lengths, the most telling words, and a "
               "browser for every review in the sample.")
    st.page_link("ui/pages/data.py", label="Explore the data", icon=":material/arrow_forward:")
with cols[2].container(border=True, height="stretch"):
    st.markdown("**:material/leaderboard: Model Comparison**")
    st.caption("Test-set metrics, confusion matrices, deep model training curves, and the "
               "reviews the best model got wrong.")
    st.page_link("ui/pages/model_comparison.py", label="Compare models",
                 icon=":material/arrow_forward:")

left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("How it works")
    st.markdown("""
1. **Cleaning** - the title and review text are lowercased, HTML and links are removed, and
   contractions are expanded so negations survive (*didn't* becomes *did not*).
2. **Features** - the cleaned text becomes a Bag-of-Words count vector or a TF-IDF vector over
   a 16,897-word vocabulary learned from the training reviews only.
3. **Classification** - each model turns that vector into a probability that the review is
   Positive; above 50% means Positive.
4. **Evaluation** - 40,000 reviews train the models and 10,000 held-out reviews test them.
   Identical reviews always stay on the same side of the split so nothing leaks.
""")
    st.subheader("Limitations")
    st.markdown("Words the models never saw in training are ignored, and the models read words "
                "without their order, so sarcasm and long-range negation (\"not what I'd call "
                "good\") can fool them. Only 22% of reviews are Negative/Neutral, so that class "
                "is harder to get right; we report its F1 and macro F1, not just accuracy.")
with right:
    st.subheader("Model leaderboard")
    st.dataframe(ranked[["Model", "Accuracy", "Macro F1"]], hide_index=True,
                 column_config={"Accuracy": st.column_config.NumberColumn(format="%.3f"),
                                "Macro F1": st.column_config.NumberColumn(format="%.3f")})
    st.subheader("Team")
    st.markdown("""
- **Matthew Moloney** - Project Lead
- **Debapratim Talukder** - Data Engineer
- **Sid Guddanti** - ML Engineer
- **Donte Beck-King** - Analyst / Writer & ML Engineer
- **Theo Ruppe** - App Engineer
""")
    st.caption("CSE 3521: Survey of Artificial Intelligence I, Autumn 2026 - Group Gamma")
