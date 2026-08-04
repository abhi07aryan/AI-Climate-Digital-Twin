import base64
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

css_path = Path(__file__).parent / "assets" / "theme.css"

with open(css_path) as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# ---------------- Page Config ---------------- #
def set_background(image_path):

    img = Path(image_path).read_bytes()
    encoded = base64.b64encode(img).decode()

    st.markdown(f"""
    <style>

    .stApp {{
        background-image:
            linear-gradient(
                rgba(8,12,24,0.72),
                rgba(8,12,24,0.72)
            ),
            url("data:image/jpeg;base64,{encoded}");

        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    </style>
    """, unsafe_allow_html=True)

set_background("assets/background2.jpg")
st.set_page_config(
    page_title="AGNI Climate Digital Twin",
    page_icon="",
    layout="wide"
)

# ---------------- Hide Sidebar ---------------- #

st.markdown("""
<style>

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

# ---------------- Navigation ---------------- #

pg = st.navigation(
    [
        st.Page("pages/1_Home.py", title="Home", icon=":material/home:"),
        st.Page("pages/1_Data_Explorer.py", title="Data Explorer", icon=":material/bar_chart:"),
        st.Page("pages/2_Model_Evaluation.py", title="Model Evaluation", icon=":material/analytics:"),
        st.Page("pages/3_Forecasting.py", title="Forecasting", icon=":material/cloud:"),
        st.Page("pages/4_Whatif.py", title="What-if Simulation", icon=":material/science:"),
        st.Page("pages/5_About.py", title="About", icon=":material/info:"),
    ],
    position="top",
)

# ---------------- Header ---------------- #

if pg.title != "Home":
    st.markdown("""
    <div class="main-title">
    AGNI Climate Digital Twin
    </div>

    <div class="subtitle">
    AI-Powered Rainfall Forecasting | Climate Simulation | Uncertainty Estimation
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

pg.run()