"""Telco churn intelligence dashboard.

Run from the project root:  streamlit run app/streamlit_app.py
"""

import streamlit as st

st.set_page_config(page_title="Telco churn intelligence", page_icon=":material/insights:", layout="wide")

page = st.navigation(
    {
        "": [
            st.Page("app_pages/overview.py", title="Overview", icon=":material/dashboard:", default=True),
        ],
        "Understand": [
            st.Page("app_pages/explore.py", title="Explore the data", icon=":material/query_stats:"),
            st.Page("app_pages/models.py", title="Model performance", icon=":material/model_training:"),
            st.Page("app_pages/explain.py", title="Explainability", icon=":material/psychology:"),
        ],
        "Act": [
            st.Page("app_pages/predictor.py", title="Customer risk scorer", icon=":material/person_search:"),
            st.Page("app_pages/campaign.py", title="Campaign planner", icon=":material/savings:"),
            st.Page("app_pages/segments.py", title="Segments & target list", icon=":material/groups:"),
            st.Page("app_pages/batch.py", title="Batch scoring", icon=":material/upload_file:"),
        ],
        "Docs": [
            st.Page("app_pages/report.py", title="Full report", icon=":material/description:"),
        ],
    },
    position="top",
)

st.title(page.title, icon=page.icon)
page.run()
