import altair as alt
import pandas as pd
import streamlit as st

from ui.predictor import analyze, load_models
from ui.theme import MODEL_COLORS, NEGATIVE, POSITIVE, model_scale

EXAMPLES = {
    "Positive": ("Best snack ever",
                 "These chips are delicious and perfectly crunchy. My kids love them and "
                 "I will definitely buy them again."),
    "Negative": ("Very disappointed",
                 "The coffee arrived stale and tasted awful. Unfortunately I would not "
                 "recommend this to anyone."),
    "Mixed": ("It's okay",
              "The flavor is good but the bag was half empty and it costs way too much "
              "for what you get."),
    "Tricky negation": ("",
                        "I was not disappointed at all, this tea is not bad. Not the best "
                        "I've had, but not something I regret buying."),
}


def fill_example(summary, review):
    st.session_state.summary = summary
    st.session_state.review = review


st.title("Predict")
st.write("Write or paste a food review and compare how each model classifies it, side by side. "
         "Reviews rated 4-5 stars count as **Positive**; 1-3 stars count as **Negative/Neutral**.")

models = load_models()

st.caption("Try an example:")
for col, (label, (summary, review)) in zip(st.columns(len(EXAMPLES)), EXAMPLES.items()):
    col.button(label, on_click=fill_example, args=(summary, review), key=f"ex_{label}",
               width="stretch")

with st.form("review_form"):
    summary = st.text_input("Review title (optional)", key="summary",
                            placeholder="e.g. Great coffee")
    review = st.text_area("Review text", key="review", height=140,
                          placeholder="Write or paste a review here...")
    chosen = st.multiselect("Models to compare (pick at least 2)", options=list(models),
                            default=list(models), format_func=lambda k: models[k].name)
    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    if not (summary.strip() or review.strip()):
        st.warning("Enter a review first.")
        st.stop()
    if len(chosen) < 2:
        st.warning("Pick at least two models to compare.")
        st.stop()
    cleaned, results = analyze(summary, review, [models[k] for k in chosen])
    st.session_state.last = {"cleaned": cleaned, "results": results}
    st.session_state.setdefault("history", []).append({
        "Review": (summary + " - " if summary.strip() else "") + review,
        **{row.Model: f"{row.Prediction} ({row.p_positive:.0%})" for row in results.itertuples()},
    })

last = st.session_state.get("last")
if not last:
    st.info("Predictions from each model will appear here side by side.")
    st.stop()

results, cleaned = last["results"], last["cleaned"]

st.subheader("Predictions")
votes = int((results["Prediction"] == "Positive").sum())
if votes in (0, len(results)):
    st.success(f"All {len(results)} models agree: **{results['Prediction'].iloc[0]}**")
else:
    st.warning(f"The models disagree: {votes} say Positive, {len(results) - votes} say "
               "Negative/Neutral.")

for col, row in zip(st.columns(len(results)), results.itertuples()):
    with col.container(border=True):
        st.markdown(f"**{row.Model}**")
        icon = ":material/thumb_up:" if row.Prediction == "Positive" else ":material/thumb_down:"
        color = "blue" if row.Prediction == "Positive" else "red"
        st.markdown(f"### :{color}[{icon} {row.Prediction}]")
        confidence = max(row.p_positive, 1 - row.p_positive)
        st.metric("Confidence", f"{confidence:.0%}")
        st.progress(row.p_positive, text=f"P(positive) = {row.p_positive:.1%}")
        st.caption(models[row.key].description)

# Probability chart: one bar per model, 50% line marks the decision threshold.
order = [n for n in MODEL_COLORS if n in set(results["Model"])]
bars = (alt.Chart(results)
        .mark_bar(cornerRadiusEnd=4, height=22)
        .encode(x=alt.X("p_positive:Q", title="Probability the review is Positive",
                        scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                y=alt.Y("Model:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
                color=alt.Color("Model:N", scale=model_scale(results["Model"]), legend=None),
                tooltip=["Model", alt.Tooltip("p_positive:Q", title="P(positive)", format=".1%"),
                         "Prediction"]))
labels = bars.mark_text(align="left", dx=4).encode(
    text=alt.Text("p_positive:Q", format=".0%"), color=alt.value("gray"))
threshold = (alt.Chart(pd.DataFrame({"x": [0.5]}))
             .mark_rule(strokeDash=[4, 4], color="gray")
             .encode(x="x:Q"))
st.altair_chart((bars + labels + threshold).properties(height=50 * len(results) + 30))
st.caption("Bars right of the dashed line (50%) are classified Positive.")

explainable = {k: models[k].word_contributions(cleaned) for k in results["key"]}
explainable = {k: df for k, df in explainable.items() if df is not None}
if explainable:
    st.subheader("Which words drove each prediction?")
    st.caption("Blue words push toward Positive, red words toward Negative/Neutral. Only words "
               "in the training vocabulary count. KNN has no per-word weights.")
    for col, (key, df) in zip(st.columns(len(explainable)), explainable.items()):
        top = df.head(10).assign(
            direction=lambda d: d["contribution"].gt(0).map(
                {True: "Positive", False: "Negative/Neutral"}))
        chart = (alt.Chart(top)
                 .mark_bar(cornerRadiusEnd=4)
                 .encode(x=alt.X("contribution:Q", title="Push toward Positive"),
                         y=alt.Y("word:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False)),
                         color=alt.Color("direction:N", legend=None,
                                         scale=alt.Scale(domain=["Positive", "Negative/Neutral"],
                                                         range=[POSITIVE, NEGATIVE])),
                         tooltip=["word", alt.Tooltip("contribution:Q", format=".3f")])
                 .properties(title=models[key].name, height=max(120, 26 * len(top))))
        col.altair_chart(chart)

with st.expander("Text the models actually saw (after cleaning)"):
    st.code(cleaned or "(empty)", language=None, wrap_lines=True)

history = st.session_state.get("history")
if history:
    st.subheader("This session")
    st.dataframe(pd.DataFrame(history[::-1]), hide_index=True)
    if st.button("Clear history"):
        st.session_state.history = []
        st.session_state.pop("last", None)
        st.rerun()
