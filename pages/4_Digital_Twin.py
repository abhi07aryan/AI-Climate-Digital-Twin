from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import xarray as xr

from climate_twin.models.convlstm import ConvLSTM
from climate_twin.preprocessing.normalize import ClimateNormalizer
from climate_twin.preprocessing.split import TimeSeriesSplit
from climate_twin.ml.dataset import ClimateTorchDataset

from climate_twin.simulation.scenario import ClimateScenario
from climate_twin.forecasting.recursive_forecast import RecursiveForecaster


# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

DATASET = "data/processed/climate_up.nc"

MODEL = "models/convlstm_best.pth"

WINDOW = 7

FEATURES = [
    "rainfall",
    "tmax",
    "tmin"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ----------------------------------------------------
# Streamlit
# ----------------------------------------------------

st.set_page_config(
    page_title="Climate Digital Twin",
    layout="wide"
)

st.title("🌍 Climate Digital Twin")

# ----------------------------------------------------
# Load Model
# ----------------------------------------------------

@st.cache_resource
def load_model():

    model = ConvLSTM(
        input_channels=3,
        hidden_channels=8,
        output_channels=1
    )

    checkpoint = torch.load(
        MODEL,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    model.eval()

    return model


# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------

@st.cache_data
def load_dataset():

    ds = xr.open_dataset(DATASET)

    splitter = TimeSeriesSplit()

    train, valid, test = splitter.split(ds)

    normalizer = ClimateNormalizer()

    normalizer.fit(train, FEATURES)

    test = normalizer.transform(test)

    dataset = ClimateTorchDataset(
        test,
        input_features=FEATURES,
        target="rainfall",
        window_size=WINDOW
    )

    return dataset


model = load_model()

dataset = load_dataset()

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

st.sidebar.header("Simulation Settings")

sample = st.sidebar.slider(
    "Sample",
    0,
    len(dataset)-1,
    0
)

days = st.sidebar.slider(
    "Forecast Days",
    1,
    7,
    3
)

temperature = st.sidebar.slider(
    "Temperature Change",
    -2.0,
    2.0,
    0.0,
    0.5
)

rainfall = st.sidebar.slider(
    "Rainfall Multiplier",
    0.5,
    2.0,
    1.0,
    0.1
)

run = st.sidebar.button("Run Simulation")

# ----------------------------------------------------
# Simulation
# ----------------------------------------------------

if run:

    X, y = dataset[sample]

    sequence = X.numpy()

    # -------------------------
    # Baseline
    # -------------------------

    baseline_forecaster = RecursiveForecaster(
        model,
        DEVICE
    )

    baseline = baseline_forecaster.forecast(
        sequence,
        days
    )

    # -------------------------
    # Scenario
    # -------------------------

    scenario = ClimateScenario(sequence)

    scenario.increase_temperature(
        temperature
    )

    scenario.multiply_rainfall(
        rainfall
    )

    modified_sequence = scenario.get_sequence()

    scenario_forecaster = RecursiveForecaster(
        model,
        DEVICE
    )

    scenario_prediction = scenario_forecaster.forecast(
        modified_sequence,
        days
    )

    day = st.slider(
        "Display Forecast Day",
        1,
        days,
        1
    )

    base = baseline[day-1]

    scen = scenario_prediction[day-1]

    diff = scen - base

    # -------------------------
    # Maps
    # -------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        st.subheader("Baseline")

        fig, ax = plt.subplots(figsize=(5,5))

        im = ax.imshow(
            base,
            cmap="Blues"
        )

        plt.colorbar(im)

        ax.axis("off")

        st.pyplot(fig)

    with c2:

        st.subheader("Scenario")

        fig, ax = plt.subplots(figsize=(5,5))

        im = ax.imshow(
            scen,
            cmap="Blues"
        )

        plt.colorbar(im)

        ax.axis("off")

        st.pyplot(fig)

    with c3:

        st.subheader("Difference")

        fig, ax = plt.subplots(figsize=(5,5))

        im = ax.imshow(
            diff,
            cmap="RdBu_r"
        )

        plt.colorbar(im)

        ax.axis("off")

        st.pyplot(fig)

    # -------------------------
    # Statistics
    # -------------------------

    st.divider()

    st.subheader("Simulation Statistics")

    a, b, c = st.columns(3)

    a.metric(
        "Average Change",
        f"{np.mean(diff):.4f}"
    )

    b.metric(
        "Maximum Increase",
        f"{np.max(diff):.4f}"
    )

    c.metric(
        "Maximum Decrease",
        f"{np.min(diff):.4f}"
    )