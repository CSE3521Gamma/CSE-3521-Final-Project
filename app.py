"""Entry point for the Streamlit app.

Run locally from the repository root:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
for folder in (ROOT, ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

st.set_page_config(page_title="Review Sentiment Analyzer", page_icon=":material/reviews:",
                   layout="wide")

pages = [
    st.Page("ui/pages/overview.py", title="Overview", icon=":material/home:", default=True),
    st.Page("ui/pages/predict.py", title="Predict", icon=":material/rate_review:"),
    st.Page("ui/pages/data.py", title="Data", icon=":material/table_chart:"),
    st.Page("ui/pages/model_comparison.py", title="Model Comparison",
            icon=":material/leaderboard:"),
]

with st.sidebar:
    st.caption("CSE 3521 - Group Gamma  \nSentiment analysis of Amazon Fine Food Reviews")

st.navigation(pages).run()
