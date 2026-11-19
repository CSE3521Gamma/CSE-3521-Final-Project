import streamlit as st

from app.predictor import MODEL_NAMES

st.title("Analyze a Review")
st.write("Paste a food review and compare how each model classifies it. Reviews rated "
         "4-5 stars count as **Positive**; 1-3 stars count as **Negative/Neutral**.")

with st.form("review_form"):
    summary = st.text_input("Review title (optional)", placeholder="e.g. Great coffee")
    review = st.text_area("Review text", height=150, placeholder="Write or paste a review here...")
    chosen = st.multiselect("Models", MODEL_NAMES, default=MODEL_NAMES)
    submitted = st.form_submit_button("Analyze", type="primary")

st.subheader("Predictions")
if submitted:
    # TODO: call app.predictor.analyze(summary, review, chosen) and show the results.
    for col, name in zip(st.columns(max(len(chosen), 1)), chosen):
        with col.container(border=True):
            st.markdown(f"**{name}**")
            st.caption("Prediction coming soon")
else:
    st.info("Predictions will appear here.")

st.subheader("Which words drove the prediction?")
st.info("Placeholder: word-level explanations for each model.")
