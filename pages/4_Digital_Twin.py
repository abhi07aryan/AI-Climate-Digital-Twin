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

WINDOW_SIZE = 30

FEATURES = [
        "rainfall",
        "tmax",
        "tmin"]
    #     "temp_mean",
    #     "temp_range",
    #     "rain_7day",
    #     "rain_30day",
    #     "rain_lag1",
    #     "rain_lag3",
    #     "rain_lag7",
    #     "month",
    #     "season",
    #     "dayofyear",
    #     "rain_anomaly"
    # ]

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

st.title("Climate Digital Twin")

# ----------------------------------------------------
# Load Model
# ----------------------------------------------------

@st.cache_resource
def load_model():

    model = ConvLSTM(
        input_channels=len(FEATURES),
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
        window_size=WINDOW_SIZE
    )

    return dataset, test

MC_SAMPLES = 30

def mc_recursive_predict(
    model,
    sequence,
    days,
    samples=MC_SAMPLES
):
    """
    Monte Carlo Dropout recursive forecasting.

    Returns:
        mean_prediction,
        std_prediction,
        confidence
    """

    predictions = []

    for _ in range(samples):

        # Enable dropout during inference
        model.train()

        forecaster = RecursiveForecaster(
            model,
            DEVICE
        )

        pred = forecaster.forecast(
            sequence.copy(),
            days
        )

        predictions.append(pred)

    predictions = np.array(predictions)

    mean_prediction = predictions.mean(axis=0)

    std_prediction = predictions.std(axis=0)

    confidence = np.exp(
        -std_prediction.mean()
    ) * 100

    return (
        mean_prediction,
        std_prediction,
        confidence
    )

model = load_model()

dataset, test = load_dataset()

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------

st.sidebar.header("Simulation Settings")

import pandas as pd

# Load test dates

forecast_dates = pd.to_datetime(
    test.time.values[WINDOW_SIZE:]
).date

selected_date = st.sidebar.date_input(
    "Forecast Date",
    value=forecast_dates[0],
    min_value=forecast_dates[0],
    max_value=forecast_dates[-1]
)

sample = np.where(
    forecast_dates == selected_date
)[0][0]

forecast_options = {
    "1 Day": 1,
    "3 Days": 3,
    "5 Days": 5,
    "7 Days": 7
}

selected = st.sidebar.selectbox(
    "Forecast Horizon",
    list(forecast_options.keys())
)

forecast_days = forecast_options[selected]

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

mc_samples = st.sidebar.selectbox(
    "Monte Carlo Samples",
    [10, 20, 30, 50],
    index=2
)

run = st.sidebar.button("Run Simulation")

st.sidebar.markdown("---")

with st.sidebar.expander("Simulation Guide", expanded=False):

    st.markdown("""
### Forecast Date
Select the date for which rainfall will be forecast. The model uses the previous seven days of climate data as input.

### Forecast Horizon
Choose how many days into the future the AI model should predict. Longer horizons use recursive forecasting, where each predicted day is used to forecast the next.

### Temperature Change
Increase or decrease the historical maximum and minimum temperatures before prediction to simulate warming or cooling scenarios.

### Rainfall Multiplier
Scale the historical rainfall values used as model input.
- **1.0** → No change
- **>1.0** → Wetter conditions
- **<1.0** → Drier conditions

### Monte Carlo Samples
Monte Carlo Dropout estimates the uncertainty of the AI model by performing multiple stochastic forward passes with dropout enabled during inference.

Each forward pass produces a slightly different rainfall prediction. The model then computes:
- **Mean Prediction:** Average of all predicted rainfall maps.
- **Prediction Uncertainty:** Standard deviation of the predictions.
- **Confidence Score:** A relative measure derived from the prediction uncertainty.

Higher sample counts produce more stable and reliable uncertainty estimates but require additional computation time.

**Recommended Settings:**
- **10 Samples:** Fast preview
- **20 Samples:** Balanced performance
- **30 Samples:** Recommended for most simulations
- **50 Samples:** Highest stability and most reliable uncertainty estimates

### Run Simulation
Runs two forecasts:
- **Baseline:** Original climate conditions.
- **Scenario:** Modified climate conditions.

The dashboard then compares both forecasts and highlights the impact of the simulated climate scenario.
""")

# ----------------------------------------------------
# Simulation
# ----------------------------------------------------

if run:

    X, y = dataset[sample]

    sequence = X.numpy()

    # -------------------------
    # Baseline
    # -------------------------


    baseline_mean, baseline_std, baseline_conf = mc_recursive_predict(
        model,
        sequence,
        forecast_days,
        mc_samples
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


    scenario_mean, scenario_std, scenario_conf = mc_recursive_predict(
        model,
        modified_sequence,
        forecast_days,
        mc_samples
    )

    col1, col2 = st.columns([1, 4])

    with col1:
        day = st.selectbox(
            "Display Forecast Day",
            range(1, forecast_days + 1)
        )

    base = baseline_mean[day-1]

    scen = scenario_mean[day-1]

    base_uncertainty = baseline_std[day-1]

    scenario_uncertainty = scenario_std[day-1]

    vmin = min(
        np.nanmin(base),
        np.nanmin(scen)
    )

    vmax = max(
        np.nanmax(base),
        np.nanmax(scen)
    )

    st.divider()

    st.subheader("Prediction Confidence")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Baseline",
        f"{baseline_conf:.1f}%"
    )

    c2.metric(
        "Scenario",
        f"{scenario_conf:.1f}%"
    )

    c3.metric(
        "Confidence Change",
        f"{scenario_conf-baseline_conf:+.1f}%"
    )

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
            cmap="Blues",
            vmin=vmin,
            vmax=vmax
        )

        plt.colorbar(im)

        ax.axis("off")

        st.pyplot(fig)

    with c2:

        st.subheader("Scenario")

        fig, ax = plt.subplots(figsize=(5,5))

        im = ax.imshow(
            scen,
            cmap="Blues",
            vmin=vmin,
            vmax=vmax
        )

        plt.colorbar(im)

        ax.axis("off")

        st.pyplot(fig)

    with c3:

        st.subheader("Difference")

        fig, ax = plt.subplots(figsize=(5,5))

        limit = np.max(np.abs(diff))

        im = ax.imshow(
            diff,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit
        )

        plt.colorbar(im)

        ax.axis("off")

        st.pyplot(fig)

    st.divider()

    st.subheader("Prediction Uncertainty")

    u1, u2 = st.columns(2)

    with u1:

        st.markdown("### Baseline Uncertainty")

        fig, ax = plt.subplots(figsize=(5,5))

        im = ax.imshow(
            base_uncertainty,
            cmap="inferno"
        )

        plt.colorbar(im)

        ax.axis("off")

        st.pyplot(fig)

    with u2:

        st.markdown("### Scenario Uncertainty")

        fig, ax = plt.subplots(figsize=(5,5))

        im = ax.imshow(
            scenario_uncertainty,
            cmap="inferno"
        )

        plt.colorbar(im)

        ax.axis("off")

        st.pyplot(fig)
    with st.expander("What does Prediction Uncertainty mean?"):

        st.markdown("""
    Prediction uncertainty is estimated using **Monte Carlo Dropout**.

    The AI model performs multiple stochastic forward passes.

    - **Dark colours** → High confidence (low uncertainty)

    - **Bright colours** → Low confidence (high uncertainty)

    Higher uncertainty indicates regions where the model is less certain about the rainfall prediction.
    """)
    # -------------------------
    # Statistics
    # -------------------------

    st.divider()

    st.subheader("Simulation Summary")

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Average Baseline Rainfall",
        f"{base.mean():.3f}"
    )

    m2.metric(
        "Average Scenario Rainfall",
        f"{scen.mean():.3f}"
    )

    m3.metric(
        "Average Change",
        f"{diff.mean():+.3f}"
    )

    m4, m5, m6 = st.columns(3)

    m4.metric(
        "Maximum Increase",
        f"{diff.max():.3f}"
    )

    m5.metric(
        "Maximum Decrease",
        f"{diff.min():.3f}"
    )

    m6.metric(
        "Maximum Uncertainty",
        f"{scenario_uncertainty.max():.3f}"
    )

    st.divider()

    st.subheader("Scenario Interpretation")

    change = diff.mean()

    if change > 0:

        message = (
            f"The selected climate scenario increased the average "
            f"predicted rainfall by **{change:.3f}** compared with "
            f"the baseline forecast."
        )

    elif change < 0:

        message = (
            f"The selected climate scenario decreased the average "
            f"predicted rainfall by **{abs(change):.3f}** compared "
            f"with the baseline forecast."
        )

    else:

        message = (
            "The selected scenario produced negligible changes "
            "in predicted rainfall."
        )

    st.info(f"""
    ### Simulation Summary

    **Temperature Change:** {temperature:+.1f} °C

    **Rainfall Scaling:** ×{rainfall:.1f}

    **Forecast Horizon:** {forecast_days} day(s)

    {message}

    Baseline Confidence: **{baseline_conf:.1f}%**

    Scenario Confidence: **{scenario_conf:.1f}%**
    """)