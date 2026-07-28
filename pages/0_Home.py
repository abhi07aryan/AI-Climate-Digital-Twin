import streamlit as st

st.set_page_config(
    page_title="AGNI - AI Climate Digital Twin",
    layout="wide"
)

# ---------------------------------------------------------
# CSS
# ---------------------------------------------------------

st.markdown("""
<style>

.stApp{
    background: linear-gradient(to bottom,#071426,#0d223b);
    color:white;
}

h1,h2,h3{
    color:white;
}

.hero{
    background: linear-gradient(135deg,#102b4e,#0a5c8f);
    padding:50px;
    border-radius:20px;
    text-align:center;
    box-shadow:0px 5px 20px rgba(0,0,0,0.3);
}

.hero h1{
    font-size:60px;
    font-weight:700;
}

.hero h3{
    color:#9ad8ff;
}

.metric-box{
    background:#132c49;
    padding:25px;
    border-radius:15px;
    text-align:center;
    box-shadow:0px 0px 12px rgba(0,0,0,0.3);
}

.card{
    background:#132c49;
    padding:25px;
    border-radius:15px;
    height:250px;
    box-shadow:0px 0px 12px rgba(0,0,0,0.3);
    transition:0.3s;
}

.card:hover{
    transform:scale(1.03);
}

.pipeline{
    background:#132c49;
    padding:30px;
    border-radius:20px;
}

.footer{
    text-align:center;
    color:gray;
    padding:20px;
}

</style>
""",unsafe_allow_html=True)

# ---------------------------------------------------------
# HERO
# ---------------------------------------------------------

st.markdown("""
<div class='hero'>

<h1>🌍 AGNI</h1>

<h3>AI Climate Digital Twin of India</h3>

<h4>Predict • Simulate • Analyze • Adapt</h4>

</div>
""",unsafe_allow_html=True)

st.write("")

c1,c2,c3=st.columns([2,1,1])

with c2:
    st.button("Launch Dashboard",use_container_width=True)

with c3:
    st.link_button(
        "GitHub",
        "https://github.com/abhi07aryan/AI-Climate-Digital-Twin",
        use_container_width=True
    )

st.divider()

# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

st.subheader("📊 Current Climate Snapshot")

m1,m2,m3,m4=st.columns(4)

m1.metric("Temperature","34.2 °C","+1.2°C")

m2.metric("Rainfall","112 mm","+18 mm")

m3.metric("Humidity","74 %","+4%")

m4.metric("Wind","15 km/h","-2 km/h")

st.divider()

# ---------------------------------------------------------
# APPLICATIONS
# ---------------------------------------------------------

st.subheader("Applications")

c1,c2=st.columns(2)

with c1:

    st.markdown("""
<div class='card'>

<h2>Flood Preparedness</h2>

Uses AI-based rainfall forecasting to estimate flood risk
and identify vulnerable regions.

✔ Heavy Rain Alerts

✔ Flood Risk Mapping

✔ Early Warning

</div>
""",unsafe_allow_html=True)

with c2:

    st.markdown("""
<div class='card'>

<h2>🌾 Drought Monitoring</h2>

Predicts rainfall deficit and water stress using
temperature and precipitation forecasts.

✔ Rainfall Deficit

✔ Water Stress

✔ Climate Anomaly

</div>
""",unsafe_allow_html=True)

st.write("")

c3,c4=st.columns(2)

with c3:

    st.markdown("""
<div class='card'>

<h2>🛰 Climate Forecasting</h2>

Forecast future climate variables using
ConvLSTM deep learning.

✔ Rainfall

✔ Temperature

✔ Multi-day Forecast

</div>
""",unsafe_allow_html=True)

with c4:

    st.markdown("""
<div class='card'>

<h2>🌍 Digital Twin</h2>

Simulate climate scenarios and evaluate
"What-if" conditions.

✔ Scenario Analysis

✔ AI Simulation

✔ Decision Support

</div>
""",unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------

st.subheader("⚡ Model Pipeline")

st.markdown("""
<div class='pipeline'>

<h3 align='center'>

IMD Climate Dataset

⬇

Feature Engineering

⬇

ConvLSTM AI Model

⬇

Climate Forecast

⬇

Digital Twin

⬇

Flood • Drought • Climate Analytics

</h3>

</div>

""",unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------
# ABOUT
# ---------------------------------------------------------

st.subheader("About AGNI")

st.write("""
AGNI (AI-powered Climate Digital Twin of India) is an intelligent
decision-support platform that combines climate observations,
deep learning and digital twin technology to forecast future
weather conditions and simulate environmental scenarios.

The platform uses **IMD climate datasets**, **ConvLSTM neural
networks**, and interactive visualizations to support:

- Climate Forecasting
- Flood Preparedness
- Drought Monitoring
- Scenario Simulation
- Disaster Management
""")

st.divider()

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown("""
<div class='footer'>

Made with ❤️ by Team 'A' Game

</div>
""",unsafe_allow_html=True)