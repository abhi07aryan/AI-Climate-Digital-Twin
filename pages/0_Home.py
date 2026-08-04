import streamlit as st
import base64

st.set_page_config(
    page_title="AGNI",
    page_icon=" ",
    layout="wide"
)

# ==========================================================
# HERO
# ==========================================================

st.markdown("""
<div class="hero">

<h1>AGNI</h1>

<h2>AI-Powered Climate Digital Twin</h2>

<p>
Predict future climate, simulate <b>what-if scenarios</b>,
assess <b>flood & drought risks</b>, and generate
<b>AI-powered insights</b> for climate resilience,
disaster preparedness, and informed decision-making.
</p>

</div>
""", unsafe_allow_html=True)

st.divider()

# ==========================================================
# QUICK STATS
# ==========================================================

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.metric(
        "Forecast Horizon",
        "7 Days"
    )

with c2:
    st.metric(
        "Model",
        "ConvLSTM"
    )

with c3:
    st.metric(
        "Resolution",
        "0.25° x 0.25°"
    )

with c4:
    st.metric(
        "Climate Variables",
        "3"
    )

st.divider()

# ==========================================================
# FEATURES
# ==========================================================


st.markdown("## Platform Features")

st.markdown("""
<div class="feature-grid">

<div class="card">
<h2>Forecasting</h2>
<p>
Generate rainfall forecasts up to seven days ahead
using a ConvLSTM deep learning model trained on
historical IMD climate observations.
</p>
</div>

<div class="card">
<h2>What-if Simulation</h2>
<p>
Modify rainfall and temperature to analyse future
climate scenarios and compare impacts.
</p>
</div>

<div class="card">
<h2>Flood & Drought Assessment</h2>
<p>
Automatically convert rainfall forecasts into flood
and drought risk maps for rapid disaster
preparedness.
</p>
</div>

<div class="card">
<h2>ASTRAA AI</h2>
<p>
AI assistant capable of explaining predictions,
uncertainty and climate impacts in natural language.
</p>
</div>

</div>
""", unsafe_allow_html=True)

st.divider()

# ==========================================================
# HOW IT WORKS
# ==========================================================

st.markdown("## ⚙ How AGNI Works")

st.markdown("""
<div class="architecture">

<div class="step">IMD & INSAT Climate Data</div>
<div class="arrow">↓</div>

<div class="step">Data Preprocessing</div>
<div class="arrow">↓</div>

<div class="step">ConvLSTM Deep Learning</div>
<div class="arrow">↓</div>

<div class="step">Recursive Forecasting</div>
<div class="arrow">↓</div>

<div class="step">Monte Carlo Uncertainty</div>
<div class="arrow">↓</div>

<div class="step">Flood & Drought Analysis</div>
<div class="arrow">↓</div>

<div class="step">Interactive Dashboard</div>

</div>
""", unsafe_allow_html=True)

st.divider()

# ==========================================================
# CAPABILITIES
# ==========================================================

st.markdown("## Key Capabilities")

left,right = st.columns(2)

with left:

    st.success("✔ Rainfall Forecasting")

    st.success("✔ Flood Prediction")

    st.success("✔ Drought Assessment")

    st.success("✔ Recursive Multi-Day Forecasting")

with right:

    st.success("✔ Monte Carlo Uncertainty")

    st.success("✔ Interactive Climate Simulation")

    st.success("✔ Explainable AI")

    st.success("✔ Decision Support Dashboard")

    st.divider()

# ==========================================================
# ABOUT
# ==========================================================

st.markdown("## About AGNI")
st.markdown("""
<div class="card" style="height:150px;">
<p>
AGNI is an AI-powered Climate Digital Twin designed to
support climate resilience through predictive analytics,
scenario simulation and explainable AI.

Using historical climate observations from the Indian
Meteorological Department (IMD) and Indian National Satellite System (INSAT), AGNI predicts rainfall,
assesses flood and drought risk and enables users to
perform interactive climate what-if analyses.
</p>
</div>
""", unsafe_allow_html=True)

st.divider()

st.link_button(
    "Github Repository",
    "https://github.com/abhi07aryan/AI-Climate-Digital-Twin",
    use_container_width=True
)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
"""

<div style="text-align:center;color:gray;padding:20px;">

Made with ❤️ by Team 'A' Game

<b>Apurva Mishra</b> •
<b>Aarti Priyadarshini</b> •
<b>Abhi Aryan</b> •
<b>Aditya Sinha</b>

</div> """, unsafe_allow_html=True )