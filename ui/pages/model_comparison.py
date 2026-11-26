import altair as alt
import pandas as pd
import streamlit as st

from ui.data import LABELS, load_deep_runs, load_misclassified, load_model_results
from ui.theme import MODEL_COLORS, model_scale

results = load_model_results()

st.title("Model Comparison")
st.write("Every model was scored on the same 10,000 held-out test reviews. Because only 22% of "
         "reviews are Negative/Neutral, a model that always says Positive already gets 77.6% "
         "accuracy, so **macro F1** (the average F1 of both classes) is the fairer score. "
         "*Neg* columns are for the Negative/Neutral class.")

metric_cols = ["Accuracy", "Neg Precision", "Neg Recall", "Neg F1", "Pos F1", "Macro F1"]
st.dataframe(
    results.sort_values("Macro F1", ascending=False)[["Model", "Type", "Features", "Settings", *metric_cols]],
    hide_index=True,
    column_config={c: st.column_config.NumberColumn(format="%.3f") for c in metric_cols},
)

tab_scores, tab_cm, tab_deep, tab_errors = st.tabs(
    ["Scores", "Confusion matrices", "Neural network training", "Error analysis"])

with tab_scores:
    metric = st.segmented_control("Metric", metric_cols, default="Macro F1")
    metric = metric or "Macro F1"
    order = results.sort_values(metric, ascending=False)["Model"].tolist()
    bars = (alt.Chart(results)
            .mark_bar(cornerRadiusEnd=4, height=24)
            .encode(x=alt.X(f"{metric}:Q", scale=alt.Scale(domain=[0, 1]), title=metric),
                    y=alt.Y("Model:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
                    color=alt.Color("Model:N", scale=model_scale(results["Model"]), legend=None),
                    tooltip=["Model", "Type", alt.Tooltip(f"{metric}:Q", format=".3f")]))
    labels = bars.mark_text(align="left", dx=4).encode(
        text=alt.Text(f"{metric}:Q", format=".3f"), color=alt.value("gray"))
    st.altair_chart((bars + labels).properties(height=50 * len(results) + 30))

    st.markdown("**Precision vs. recall on the Negative/Neutral class**")
    st.caption("Up and to the right is better. Recall: how many negative reviews were caught. "
               "Precision: how many flagged reviews were really negative.")
    pr = results[results["Type"] != "Baseline"]
    points = (alt.Chart(pr)
              .mark_point(filled=True, size=160, stroke="white", strokeWidth=2, opacity=1)
              .encode(x=alt.X("Neg Recall:Q", scale=alt.Scale(domain=[0.2, 1]), title="Recall"),
                      y=alt.Y("Neg Precision:Q", scale=alt.Scale(domain=[0.5, 0.8]), title="Precision"),
                      color=alt.Color("Model:N", scale=model_scale(pr["Model"]),
                                      legend=alt.Legend(title=None, orient="bottom", columns=2,
                                                        labelLimit=300)),
                      tooltip=["Model", alt.Tooltip("Neg Precision:Q", format=".3f"),
                               alt.Tooltip("Neg Recall:Q", format=".3f"),
                               alt.Tooltip("Neg F1:Q", format=".3f")]))
    # Points near the right edge are labeled on their left so names stay inside the plot;
    # the highest-precision point among them is labeled above so it doesn't collide.
    top = pr.loc[pr["Neg Recall"] > 0.6, "Neg Precision"].max()
    near_right = alt.datum["Neg Recall"] > 0.6
    is_top = alt.datum["Neg Precision"] == top
    text = dict(fontSize=12)
    names = [
        points.transform_filter(near_right & is_top).mark_text(align="center", dy=-14, **text),
        points.transform_filter(near_right & ~is_top).mark_text(align="right", dx=-12, **text),
        points.transform_filter(~near_right).mark_text(align="left", dx=12, **text),
    ]
    names = [n.encode(text="Model:N", color=alt.value("gray")) for n in names]
    st.altair_chart(alt.layer(points, *names).properties(height=340))

with tab_cm:
    st.caption("Rows are the true label, columns the prediction. Color shows the share of each "
               "true class; the diagonal is correct.")
    models = [m for m in MODEL_COLORS if m in set(results["Model"])]
    for row_start in range(0, len(models), 3):
        for col, name in zip(st.columns(3), models[row_start:row_start + 3]):
            cm = results.loc[results["Model"] == name, "confusion_matrix"].iloc[0]
            cells = pd.DataFrame([{"True": LABELS[t], "Predicted": LABELS[p], "Reviews": cm[t][p],
                                   "Share": cm[t][p] / sum(cm[t])}
                                  for t in (0, 1) for p in (0, 1)])
            base = alt.Chart(cells).encode(
                x=alt.X("Predicted:N", sort=LABELS, axis=alt.Axis(labelAngle=0, orient="top")),
                y=alt.Y("True:N", sort=LABELS))
            heat = base.mark_rect(stroke="white", strokeWidth=2).encode(
                color=alt.Color("Share:Q", legend=None,
                                scale=alt.Scale(domain=[0, 1], range=["#cde2fb", "#184f95"])),
                tooltip=["True", "Predicted", alt.Tooltip("Reviews:Q", format=","),
                         alt.Tooltip("Share:Q", format=".1%")])
            text = base.mark_text(fontSize=14).encode(
                text=alt.Text("Reviews:Q", format=","),
                color=alt.condition("datum.Share > 0.5", alt.value("white"), alt.value("black")))
            col.altair_chart((heat + text).properties(title=name, height=200))

with tab_deep:
    runs = load_deep_runs()
    if not runs:
        st.info("No neural network results found yet.")
    else:
        st.write("A feedforward network (TF-IDF input, 256 and 64 hidden units with dropout) "
                 "from M3. The table above uses the latest run. Validation loss that "
                 "climbs while training loss keeps falling means the network is overfitting, which "
                 "is why the weights from the lowest-validation-loss epoch are kept.")
        losses = pd.DataFrame([{"Run": r["run"], "Epoch": i + 1, "Loss": loss, "Set": s}
                               for r in runs
                               for s, key in (("Train", "train_loss"), ("Validation", "val_loss"))
                               for i, loss in enumerate(r.get(key, []))])
        for col, run in zip(st.columns(len(runs)), runs):
            sub = losses[losses["Run"] == run["run"]]
            base = alt.Chart(sub).encode(
                x=alt.X("Epoch:O", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Loss:Q"),
                color=alt.Color("Set:N", scale=alt.Scale(domain=["Train", "Validation"],
                                                         range=["#2a78d6", "#eb6834"]),
                                legend=alt.Legend(title=None, orient="top")),
                tooltip=["Set", "Epoch", alt.Tooltip("Loss:Q", format=".4f")])
            with col:
                st.altair_chart((base.mark_line(strokeWidth=2) + base.mark_point(filled=True, size=60))
                                .properties(title=run["run"], height=260))
                st.caption(f"Test macro F1 {run['macro_f1']:.3f} - accuracy {run['accuracy']:.3f}"
                           + (f" - trained in {run['train_time_sec']:.0f}s" if "train_time_sec" in run else ""))

with tab_errors:
    errors = load_misclassified()
    st.write(f"Logistic Regression got **{len(errors):,}** of 10,000 test reviews wrong. Filter "
             "them to look for patterns such as negation, sarcasm, mixed opinions, or 3-star reviews.")
    f1, f2, f3 = st.columns([1, 1, 2])
    true_label = f1.selectbox("True label", ["Any", *LABELS[::-1]])
    stars = f2.multiselect("Stars", sorted(errors["Score"].unique()))
    search = f3.text_input("Contains text", key="error_search")
    shown = errors
    if true_label != "Any":
        shown = shown[shown["true_label"] == true_label]
    if stars:
        shown = shown[shown["Score"].isin(stars)]
    if search:
        shown = shown[shown["clean_text"].str.contains(search, case=False, regex=False)]
    st.caption(f"{len(shown):,} reviews")
    st.dataframe(
        shown.drop(columns=["Id"]),
        hide_index=True,
        column_config={
            "Score": st.column_config.NumberColumn("Stars", format="%d ⭐"),
            "true_label": "True label",
            "predicted": "Predicted",
            "prob_positive": st.column_config.ProgressColumn(
                "P(positive)", min_value=0.0, max_value=1.0, format="%.2f"),
            "word_count": "Words",
            "clean_text": st.column_config.TextColumn("Review", width="large"),
        },
    )
