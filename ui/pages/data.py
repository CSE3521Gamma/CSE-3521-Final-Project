import altair as alt
import pandas as pd
import streamlit as st

from ui.data import load_reviews, load_top_words
from ui.theme import label_scale

df = load_reviews()

st.title("Data")
st.write("A 50,000-review sample (seed 44) of the "
         "[Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) "
         "dataset. The model reads each review's title and text together.")

pos_share = (df["label"] == "Positive").mean()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews", f"{len(df):,}", border=True)
c2.metric("Positive", f"{pos_share:.1%}", border=True)
c3.metric("Train / test", f"{(df['split'] == 'train').sum() / 1000:.0f}k / "
          f"{(df['split'] == 'test').sum() / 1000:.0f}k", border=True)
c4.metric("Median length", f"{int(df['words'].median())} words", border=True)

legend = alt.Legend(title=None, orient="top")
left, right = st.columns(2, gap="large")

with left:
    st.subheader("Star ratings")
    stars = df.groupby(["Score", "label"]).size().reset_index(name="reviews")
    chart = (alt.Chart(stars)
             .mark_bar(cornerRadiusEnd=4)
             .encode(x=alt.X("Score:O", title="Stars", axis=alt.Axis(labelAngle=0)),
                     y=alt.Y("reviews:Q", title="Reviews"),
                     color=alt.Color("label:N", scale=label_scale(), legend=legend),
                     tooltip=[alt.Tooltip("Score:O", title="Stars"), alt.Tooltip("label:N", title="Label"),
                              alt.Tooltip("reviews:Q", title="Reviews", format=",")]))
    text = chart.mark_text(dy=-6).encode(text=alt.Text("reviews:Q", format=","),
                                         color=alt.value("gray"))
    st.altair_chart((chart + text).properties(height=300))
    st.caption("Most reviews are 5 stars, so the classes are imbalanced: about 78% Positive.")

with right:
    st.subheader("Review length")
    bins = [0, 25, 50, 75, 100, 150, 200, 300, 500, float("inf")]
    names = ["<25", "25-49", "50-74", "75-99", "100-149", "150-199", "200-299", "300-499", "500+"]
    lengths = (df.assign(bucket=pd.cut(df["words"], bins=bins, labels=names, right=False))
                 .groupby(["label", "bucket"], observed=False).size()
                 .groupby(level=0).transform(lambda s: s / s.sum())
                 .reset_index(name="share"))
    base = alt.Chart(lengths).encode(
        x=alt.X("bucket:O", sort=names, title="Words in review text",
                axis=alt.Axis(labelAngle=-35, labelOverlap=False)),
        y=alt.Y("share:Q", title="Share of reviews in class", axis=alt.Axis(format="%")),
        color=alt.Color("label:N", scale=label_scale(), legend=legend),
        tooltip=[alt.Tooltip("label:N", title="Label"), alt.Tooltip("bucket:O", title="Words"),
                 alt.Tooltip("share:Q", title="Share", format=".1%")])
    st.altair_chart((base.mark_line(strokeWidth=2) + base.mark_point(filled=True, size=60))
                    .properties(height=300))
    med = df.groupby("label")["words"].median()
    st.caption(f"Negative/Neutral reviews run a little longer (median {med['Negative/Neutral']:.0f} "
               f"words vs. {med['Positive']:.0f} for Positive): people explain what went wrong.")

st.subheader("Words most associated with each class")
st.caption("Chi-square association on TF-IDF features from the training set.")
words = load_top_words()
for col, label in zip(st.columns(2, gap="large"), ["Positive", "Negative/Neutral"]):
    sub = words[words["label"] == label]
    col.altair_chart(
        alt.Chart(sub)
        .mark_bar(cornerRadiusEnd=4)
        .encode(x=alt.X("chi2:Q", title="Chi-square score"),
                y=alt.Y("word:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False)),
                color=alt.Color("label:N", scale=label_scale(), legend=None),
                tooltip=["word", alt.Tooltip("chi2:Q", title="Chi-square", format=",.0f")])
        .properties(title=label, height=320))

st.subheader("Reviews per year")
per_year = df.groupby(["year", "label"]).size().reset_index(name="reviews")
year_base = alt.Chart(per_year).encode(
    x=alt.X("year:O", title=None, axis=alt.Axis(labelAngle=0)),
    y=alt.Y("reviews:Q", title="Reviews"),
    color=alt.Color("label:N", scale=label_scale(), legend=legend),
    tooltip=[alt.Tooltip("year:O", title="Year"), alt.Tooltip("label:N", title="Label"),
             alt.Tooltip("reviews:Q", title="Reviews", format=",")])
st.altair_chart((year_base.mark_line(strokeWidth=2) + year_base.mark_point(filled=True, size=60))
                .properties(height=260))

st.subheader("Browse reviews")
f1, f2, f3, f4 = st.columns([1, 1, 1, 2])
label = f1.selectbox("Label", ["Any", "Positive", "Negative/Neutral"])
split = f2.selectbox("Split", ["Any", "train", "test"])
star_filter = f3.multiselect("Stars", [1, 2, 3, 4, 5])
search = f4.text_input("Contains text", placeholder="e.g. coffee")
shown = df
if label != "Any":
    shown = shown[shown["label"] == label]
if split != "Any":
    shown = shown[shown["split"] == split]
if star_filter:
    shown = shown[shown["Score"].isin(star_filter)]
if search:
    mask = (shown["Text"].str.contains(search, case=False, regex=False, na=False)
            | shown["Summary"].str.contains(search, case=False, regex=False, na=False))
    shown = shown[mask]
st.caption(f"{len(shown):,} matching reviews" + (" (showing the first 1,000)" if len(shown) > 1000 else ""))
st.dataframe(
    shown.head(1000)[["Score", "label", "split", "Summary", "Text", "words", "year"]],
    hide_index=True,
    column_config={
        "Score": st.column_config.NumberColumn("Stars", format="%d ⭐"),
        "label": "Label", "split": "Split", "words": "Words",
        "year": st.column_config.NumberColumn("Year", format="%d"),
        "Text": st.column_config.TextColumn("Review", width="large"),
    },
)
