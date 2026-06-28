from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import xarray as xr

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET = Path("data/processed/climate_up.nc")

st.set_page_config(
    page_title="Data Explorer",
    layout="wide"
)

st.title("📊 Climate Data Explorer")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

@st.cache_data
def load_dataset():
    return xr.open_dataset(DATASET)

ds = load_dataset()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Explorer")

variables = sorted(list(ds.data_vars))

# Variables to hide
exclude = {
    "month",
    "dayofyear"
}

VARIABLE_INFO = {

    "rainfall":
        "Daily observed rainfall (mm). Primary target variable used for training and forecasting.",

    "tmax":
        "Daily maximum surface air temperature (°C), representing the warmest part of the day.",

    "tmin":
        "Daily minimum surface air temperature (°C), representing the coolest part of the day.",

    "temp_mean":
        "Mean daily temperature computed as the average of Tmax and Tmin, representing overall daily thermal conditions.",

    "temp_range":
        "Difference between Tmax and Tmin, indicating the daily temperature variability.",

    "rain_7day":
        "Rolling 7-day accumulated rainfall, capturing recent precipitation trends and antecedent wetness.",

    "rain_30day":
        "Rolling 30-day accumulated rainfall, capturing recent precipitation trends and antecedent wetness.",

    "rain_lag1":
        "Rainfall observed one day before the current date. Used to capture short-term temporal dependencies.",

    "rain_lag3":
        "Rainfall observed three days before the current date. Helps model medium-term rainfall persistence.",

    "rain_lag7":
        "Rainfall observed seven days before the current date, representing weekly rainfall memory.",

    "rain_anomaly":
        "Difference between observed rainfall and its long-term climatological average, highlighting unusually wet or dry conditions.",

    "season":
        "Numerical encoding of the meteorological season (Winter, Pre-Monsoon, Monsoon, or Post-Monsoon)."
}

variables = sorted([
    var
    for var in ds.data_vars
    if var not in exclude
])

variable = st.sidebar.selectbox(
    "Variable",
    variables
)

st.markdown("### Variable Description")

st.success(
    VARIABLE_INFO.get(
        variable,
        "No description available."
    )
)

import pandas as pd

# Available dates in dataset
available_dates = pd.to_datetime(ds.time.values).date

selected_date = st.sidebar.date_input(
    "Select Date",
    value=available_dates[0],
    min_value=available_dates[0],
    max_value=available_dates[-1]
)

# Convert selected date to dataset index
try:
    time_index = np.where(available_dates == selected_date)[0][0]
except IndexError:
    st.error("Selected date is not available in the dataset.")
    st.stop()

cmap = st.sidebar.selectbox(
    "Colormap",
    [
        "Blues",
        "viridis",
        "plasma",
        "coolwarm",
        "terrain"
    ]
)

show_hist = st.sidebar.checkbox(
    "Show Histogram",
    True
)

show_stats = st.sidebar.checkbox(
    "Show Statistics",
    True
)

# --------------------------------------------------
# Variable
# --------------------------------------------------

data = ds[variable]

st.subheader(variable)

st.caption(
    f"Date : {str(ds.time.values[time_index])[:10]}"
)

# --------------------------------------------------
# Spatial Variables
# --------------------------------------------------

if data.ndim == 3:

    image = data.isel(time=time_index)

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        image.values,
        cmap=cmap,
        origin="lower",
        extent=[
            float(ds.lon.min()),
            float(ds.lon.max()),
            float(ds.lat.min()),
            float(ds.lat.max())
        ]
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(variable)

    plt.colorbar(
        im,
        ax=ax,
        shrink=0.8,
        label=variable
    )

    st.pyplot(fig)

    if show_stats:

        st.subheader("Statistics")

        c1, c2, c3, c4 = st.columns(4)

        values = image.values

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
            "Std Dev",
            f"{np.nanstd(values):.3f}"
        )

    if show_hist:

        st.subheader("Distribution")

        fig, ax = plt.subplots(figsize=(8,4))

        ax.hist(
            image.values.flatten(),
            bins=50,
            color="steelblue"
        )

        ax.set_xlabel(variable)
        ax.set_ylabel("Frequency")

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

# --------------------------------------------------
# Dataset Information
# --------------------------------------------------

st.divider()

st.subheader("Dataset Information")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Variables",
    len(ds.data_vars)
)

c2.metric(
    "Time Steps",
    len(ds.time)
)

c3.metric(
    "Grid Size",
    f"{len(ds.lat)} × {len(ds.lon)}"
)

st.write("Available Variables")

st.dataframe(
    {
        "Variable": list(ds.data_vars.keys())
    },
    use_container_width=True
)