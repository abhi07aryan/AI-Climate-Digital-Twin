import streamlit as st

# ---------------- Page Config ---------------- #

st.set_page_config(
    page_title="AGNI Climate Digital Twin",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- Hide Sidebar ---------------- #

st.markdown("""
<style>

/* Hide sidebar completely */
[data-testid="stSidebar"]{
    display:none;
}

/* Hide hamburger */
[data-testid="collapsedControl"]{
    display:none;
}

/* Reduce top padding */
.block-container{
    padding-top:2rem;
    max-width:95rem;
}

.main-title{
    font-size:56px;
    font-weight:800;
    color:white;
    margin-bottom:5px;
}

.subtitle{
    font-size:20px;
    color:#c9d1d9;
    margin-bottom:25px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- Header ---------------- #

st.markdown("""
<div class="main-title">
AGNI Climate Digital Twin
</div>

<div class="subtitle">
AI-Powered Rainfall Forecasting | Climate Simulation | Uncertainty Estimation
</div>
""", unsafe_allow_html=True)

# ---------------- Navigation ---------------- #

pg = st.navigation(
    [
        st.Page("pages/1_Data_Explorer.py", title="Data Explorer", icon="📊"),
        st.Page("pages/2_Model_Evaluation.py", title="Model Evaluation", icon="📈"),
        st.Page("pages/3_Forecasting.py", title="Forecasting", icon="🌧️"),
        st.Page("pages/4_Digital_Twin.py", title="Digital Twin", icon="🌍"),
        st.Page("pages/5_About.py", title="About", icon="ℹ️"),
    ],
    position="top",
)

st.markdown("---")

pg.run()