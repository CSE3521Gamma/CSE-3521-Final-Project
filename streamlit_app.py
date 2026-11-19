"""Entry point for the Streamlit app.

Run locally from the repository root:
    streamlit run streamlit_app.py

Streamlit Community Cloud looks for this file name by default.
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
    st.Page("app/pages/analyze.py", title="Analyze a Review", icon=":material/rate_review:",
            default=True),
    st.Page("app/pages/model_results.py", title="Model Results", icon=":material/leaderboard:"),
    st.Page("app/pages/about.py", title="About", icon=":material/info:"),
]

with st.sidebar:
    st.markdown("**CSE 3521 - Group Gamma**")
    st.caption("Sentiment analysis of Amazon Fine Food Reviews")

st.navigation(pages).run()
