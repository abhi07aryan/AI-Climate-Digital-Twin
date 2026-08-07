import streamlit as st

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
<h2>Data Explorer</h2>
<p>
Explore rainfall and temperature data for the
past 75 years along with other features such
as rain anomaly, past 7 days average, etc.
</p>
</div>

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



# from pathlib import Path

# import gdown
# import geopandas as gpd
# import folium
# import streamlit as st
# import xarray as xr
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize
# import matplotlib.colors as mcolors
# from streamlit_folium import st_folium

# DATASET = Path("data/processed/climate_up_clip.nc")

# def download_dataset():
#     DATASET.parent.mkdir(parents=True, exist_ok=True)

#     gdown.download(
#         id="1sZeQ45vGjq-7xx1RfLeoBDg_ZQqHmSWT",
#         output=str(DATASET),
#         quiet=False,
#     )

# @st.cache_data
# def load_dataset():
#     if not DATASET.exists():
#         download_dataset()
#     return xr.open_dataset(DATASET)

# ds = load_dataset()

# # UP boundary
# up = gpd.read_file("data/raw/shapefiles/UP_Boundary/UP_Boundary.shp")
# up = up.to_crs(epsg=4326)

# # --------------------------
# # Rainfall for selected day
# # --------------------------
# rain = ds["rainfall"].sel(time="2024-07-08")

# # Create map
# m = folium.Map(tiles="CartoDB positron")

# # Fit to UP boundary
# m.fit_bounds([
#     [up.total_bounds[1], up.total_bounds[0]],
#     [up.total_bounds[3], up.total_bounds[2]],
# ])

# # Colour map
# norm = Normalize(
#     vmin=0,
#     vmax=np.nanpercentile(rain.values, 98)
# )

# cmap = plt.cm.turbo

# res = 0.25  # IMD grid resolution

# # --------------------------
# # Draw rainfall cells
# # --------------------------
# for i, lat in enumerate(rain.lat.values):
#     for j, lon in enumerate(rain.lon.values):

#         value = rain.values[i, j]

#         if np.isnan(value):
#             continue

#         color = mcolors.to_hex(cmap(norm(value)))

#         folium.Rectangle(
#             bounds=[
#                 [lat - res/2, lon - res/2],
#                 [lat + res/2, lon + res/2],
#             ],
#             stroke=False,
#             fill=True,
#             fill_color=color,
#             fill_opacity=0.75,
#             popup=f"{value:.1f} mm",
#         ).add_to(m)

# # --------------------------
# # UP Boundary
# # --------------------------
# folium.GeoJson(
#     up,
#     style_function=lambda x: {
#         "fillOpacity": 0,
#         "color": "black",
#         "weight": 2,
#     },
# ).add_to(m)

# st_folium(m, width=900, height=650)