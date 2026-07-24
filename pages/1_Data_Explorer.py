from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import xarray as xr
import pandas as pd

# Colormap for each variable
CMAPS = {
    # Rainfall variables
    "rainfall": "Blues",
    "rain_lag1": "Blues",
    "rain_lag3": "Blues",
    "rain_lag7": "Blues",
    "rain_7day": "Blues",
    "rain_anomaly": "RdBu_r",

    # Temperature variables
    "tmax": "coolwarm",
    "tmin": "coolwarm",
    "temp_mean": "coolwarm",
    "temp_range": "plasma",
    "season": "tab10"
}

COLOR_LIMITS = {
    "rainfall": (0, 60),
    "rain_lag1": (0, 60),
    "rain_lag3": (0, 60),
    "rain_lag7": (0, 60),
    "rain_7day": (0, 150),
    "rain_anomaly": (-20, 20),

    "tmax": (20, 45),
    "tmin": (5, 30),
    "temp_mean": (15, 40),
    "temp_range": (5, 25),
}

LABELS = {
    "rainfall": "Rainfall (mm/day)",
    "rain_lag1": "Rainfall (1-day lag) (mm/day)",
    "rain_lag3": "Rainfall (3-day lag) (mm/day)",
    "rain_lag7": "Rainfall (7-day lag) (mm/day)",
    "rain_7day": "7-Day Rainfall (mm)",
    "rain_anomaly": "Rainfall Anomaly (mm/day)",

    "tmax": "Maximum Temperature (°C)",
    "tmin": "Minimum Temperature (°C)",
    "temp_mean": "Mean Temperature (°C)",
    "temp_range": "Temperature Range (°C)",
    "season": "Season",
}

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET = Path("data/processed/climate_up.nc")

st.set_page_config(
    page_title="Data Explorer",
    layout="wide"
)

st.title("Climate Data Explorer")
st.caption(
    "Explore climate variables across the region through interactive spatial maps, temporal trends, and statistical summaries."
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

@st.cache_data
def load_dataset():
    return xr.open_dataset(DATASET)

ds = load_dataset()

variables = sorted(list(ds.data_vars))

# --------------------------------------------------
# Variables
# --------------------------------------------------

exclude = {
    "month",
    "dayofyear",
}

preferred_order = [
    "rainfall",
    "tmax",
    "tmin",
    "temp_mean",
    "temp_range",
    "rain_7day",
    "rain_anomaly",
    "rain_lag1",
    "rain_lag3",
    "rain_lag7",
    "season",
]

VARIABLE_VALIDITY = {
    "rainfall": 0,
    "tmax": 0,
    "tmin": 0,
    "temp_mean": 0,
    "temp_range": 0,
    "rain_lag1": 1,
    "rain_lag3": 3,
    "rain_lag7": 7,
    "rain_7day": 7,
    "rain_anomaly": 30,   # Adjust if your anomaly uses a different window
    "season": 0,
}

VAR_DESCRIPTIONS = {
    "rainfall": "Daily observed rainfall (mm/day).",
    "tmax": "Daily maximum air temperature (°C).",
    "tmin": "Daily minimum air temperature (°C).",
    "temp_mean": "Average daily temperature computed from Tmax and Tmin (°C).",
    "temp_range": "Difference between Tmax and Tmin (°C).",
    "rain_lag1": "Rainfall recorded one day before the selected date (mm/day).",
    "rain_lag3": "Rainfall recorded three days before the selected date (mm/day).",
    "rain_lag7": "Rainfall recorded seven days before the selected date (mm/day).",
    "rain_7day": "Cumulative rainfall over the previous seven days (mm).",
    "rain_anomaly": "Deviation of daily rainfall from its climatological average (mm/day).",
    "season": "Meteorological season corresponding to the selected date."
}

available_dates = pd.to_datetime(ds.time.values).date

st.subheader("Climate Data Explorer")

col1, col2, col3, col4 = st.columns(
    [3, 3, 1.5, 1.5],
    vertical_alignment="bottom"
)

# -----------------------------
# Date Selector
# -----------------------------
with col2:
    selected_date = st.date_input(
        "Date",
        value=available_dates[0],
        min_value=available_dates[0],
        max_value=available_dates[-1],
        key="selected_date"
    )

days_since_start = (
    pd.Timestamp(selected_date)
    - pd.Timestamp(ds.time.values[0])
).days

# -----------------------------
# Valid Variables
# -----------------------------
variables = [
    var
    for var in preferred_order
    if (
        var in ds.data_vars
        and var not in exclude
        and days_since_start >= VARIABLE_VALIDITY.get(var, 0)
    )
]

if not variables:
    st.warning("No variables are available for the selected date.")
    st.stop()

# -----------------------------
# Variable Selector
# -----------------------------
with col1:
    variable = st.selectbox(
        "Parameter",
        variables,
        key="parameter"
    )

# -----------------------------
# Display Options
# -----------------------------
with col3:
    show_stats = st.checkbox(
        "Statistics",
        value=True,
        key="show_stats"
    )

with col4:
    show_hist = st.checkbox(
        "Histogram",
        value=True,
        key="show_hist"
    )

# -----------------------------
# Time Index
# -----------------------------
try:
    time_index = available_dates.tolist().index(selected_date)
except ValueError:
    st.error("Selected date is not available.")
    st.stop()

data = ds[variable]

st.subheader(variable)
st.caption(
    VAR_DESCRIPTIONS.get(
        variable,
        "No description available for this variable."
    )
)
st.caption(
    f"Date : {str(ds.time.values[time_index])[:10]}"
)

# --------------------------------------------------
# Spatial Variables
# --------------------------------------------------
cmap_name = CMAPS.get(variable, "viridis")
vmin, vmax = COLOR_LIMITS.get(variable, (None, None))

STREAMLIT_BG = "#0E1117"

if data.ndim == 3:

    image = data.isel(time=time_index)

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(STREAMLIT_BG)

    mask = np.isfinite(image.values)

    lat = ds.lat.values
    lon = ds.lon.values

    valid_rows = np.where(mask.any(axis=1))[0]
    valid_cols = np.where(mask.any(axis=0))[0]

    extent = [
        lon[valid_cols[0]],
        lon[valid_cols[-1]],
        lat[valid_rows[0]],
        lat[valid_rows[-1]]
    ]

    fig, ax = plt.subplots(
        figsize=(6, 5),
        facecolor=STREAMLIT_BG
    )

    ax.set_facecolor(STREAMLIT_BG)

    lon2d, lat2d = np.meshgrid(
        lon[valid_cols[0]:valid_cols[-1]+1],
        lat[valid_rows[0]:valid_rows[-1]+1]
    )

    im = ax.pcolormesh(
        lon2d,
        lat2d,
        image.values[
            valid_rows[0]:valid_rows[-1]+1,
            valid_cols[0]:valid_cols[-1]+1
        ],
        cmap=cmap,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlim(75.8, 85.7)
    ax.set_ylim(23.2, 31.0)
    ax.set_xlabel("Longitude (°E)", color="white")
    ax.set_ylabel("Latitude (°N)", color="white")
    ax.tick_params(colors="white")

    for spine in ax.spines.values():
        spine.set_visible(False)

    label = LABELS.get(variable, variable)
    ax.set_title(label, color="white")
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(label, color="white")
    cbar.ax.tick_params(colors="white")
    plt.setp(cbar.ax.get_yticklabels(), color="white")
    cbar.outline.set_edgecolor("white")

    plt.tight_layout()

    st.pyplot(fig)
    
    if show_stats:
        st.subheader("Statistics")
        c1, c2, c3, c4 = st.columns(4)
        values = image.values
        c1.metric("Mean", f"{np.nanmean(values):.3f}")
        c2.metric("Maximum", f"{np.nanmax(values):.3f}")
        c3.metric("Minimum", f"{np.nanmin(values):.3f}")
        c4.metric("Std Dev", f"{np.nanstd(values):.3f}")

    if show_hist:
        st.subheader("Distribution")
        fig, ax = plt.subplots(
            figsize=(8,4),
            facecolor=STREAMLIT_BG
        )
        ax.set_facecolor(STREAMLIT_BG)
        values = image.values
        values = values[np.isfinite(values)]
        ax.hist(
            values,
            bins=50,
            color="steelblue"
        )
        ax.grid(linestyle="--",alpha=0.2,color="white")
        ax.tick_params(colors="white")
        ax.set_xlabel(variable, color="white")
        ax.set_ylabel("Frequency", color="white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        st.pyplot(fig)

# --------------------------------------------------
# Time Variables
# --------------------------------------------------

elif data.ndim == 1:

    fig, ax = plt.subplots(figsize=(10,4))

    ax.plot(
        ds.time.values,
        data.values,
        linewidth=1.5
    )

    ax.scatter(
        ds.time.values[time_index],
        data.values[time_index],
        color="red",
        s=40
    )

    ax.set_xlabel("Time")
    ax.set_ylabel(variable)
    ax.set_title(variable)

    plt.xticks(rotation=30)

    st.pyplot(fig)

    if show_stats:

        st.subheader("Statistics")

        c1, c2, c3, c4 = st.columns(4)

        values = data.values

        c1.metric(
            "Mean",
            f"{np.nanmean(values):.3f}"
        )

        c2.metric(
            "Maximum",
            f"{np.nanmax(values):.3f}"
        )

        c3.metric(
            "Minimum",
            f"{np.nanmin(values):.3f}"
        )

        c4.metric(
            "Current",
            f"{values[time_index]:.3f}"
        )

with st.expander("What do these statistics mean?"):

    st.markdown("""
    **Mean:** Average value across all grid cells for the selected date.

    **Maximum:** Largest value observed in the selected map.

    **Minimum:** Smallest value observed in the selected map.

    **Standard Deviation:** Indicates how much the values vary spatially. A larger standard deviation means greater variability across the region.
    """)

# --------------------------------------------------
# Dataset Information
# --------------------------------------------------

st.divider()
st.subheader("Dataset Information")
c1, c2, c3 = st.columns(3) 
c1.metric( "Variables", len(ds.data_vars) )
c2.metric( "Time Steps", len(ds.time) ) 
c3.metric( "Grid Size", f"{len(ds.lat)} × {len(ds.lon)}" )