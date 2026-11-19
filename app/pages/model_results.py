import streamlit as st

st.title("Model Results")
st.write("How each model scored on the 10,000-review held-out test set.")

# TODO: load results/baseline/baseline_results.csv (and the deep learning results).
st.info("Placeholder: metrics table comparing the models.")

tab_cm, tab_words, tab_errors = st.tabs(["Confusion matrices", "Informative words",
                                         "Misclassified reviews"])
with tab_cm:
    st.info("Placeholder: confusion matrices (results/baseline/).")
with tab_words:
    st.info("Placeholder: top informative words per model.")
with tab_errors:
    st.info("Placeholder: explorer for misclassified test reviews.")
