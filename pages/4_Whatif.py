from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import streamlit as st
import torch
import xarray as xr
import pandas as pd
import gdown

from climate_twin.models.convlstm import ConvLSTM
from climate_twin.preprocessing.normalize import ClimateNormalizer
from climate_twin.preprocessing.split import TimeSeriesSplit
from climate_twin.ml.dataset import ClimateTorchDataset
from climate_twin.simulation.scenario import ClimateScenario
from climate_twin.forecasting.recursive_forecast import RecursiveForecaster


# Major cities in Uttar Pradesh
UP_CITIES = {
    "Lucknow":    (26.8467, 80.9462),
    "Kanpur":     (26.4499, 80.3319),
    "Prayagraj":  (25.4358, 81.8463),
    "Varanasi":   (25.3176, 82.9739),
    "Gorakhpur":  (26.7606, 83.3732),
    "Bareilly":   (28.3670, 79.4304),
    "Meerut":     (28.9845, 77.7064),
    "Jhansi":     (25.4484, 78.5685),
}
# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

DATASET = Path("data/processed/climate_up.nc")
def download_dataset():
    DATASET.parent.mkdir(parents=True, exist_ok=True)

    file_id = "1Ld7oVZJ5XCFi6o8ZPZ0iM9vErvmQTmZu"

    gdown.download(
        id=file_id,
        output=str(DATASET),
        quiet=False
    )

MODEL = "models/convlstm_up_best.pth"

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

# Major cities in Uttar Pradesh
UP_CITIES = {
    "Lucknow":    (26.8467, 80.9462),
    "Kanpur":     (26.4499, 80.3319),
    "Prayagraj":  (25.4358, 81.8463),
    "Varanasi":   (25.3176, 82.9739),
    "Gorakhpur":  (26.7606, 83.3732),
    "Bareilly":   (28.3670, 79.4304),
    "Meerut":     (28.9845, 77.7064),
    "Jhansi":     (25.4484, 78.5685),
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def add_up_cities(ax):
    """
    Add major Uttar Pradesh cities to any Cartopy axis.
    """

    for city, (lat, lon) in UP_CITIES.items():

        ax.scatter(
            lon,
            lat,
            s=18,
            c="red",
            edgecolors="white",
            linewidth=0.6,
            zorder=20,
        )

        ax.text(
            lon + 0.08,
            lat + 0.08,
            city,
            fontsize=7,
            color="white",
            weight="bold",
            bbox=dict(
                facecolor="black",
                alpha=0.45,
                edgecolor="none",
                pad=1,
            ),
            zorder=21,
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

    model.load_state_dict(checkpoint)

    model.to(DEVICE)

    model.eval()

    return model


# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------

@st.cache_data
def load_dataset():
    if not DATASET.exists():
        with st.spinner("Downloading dataset..."):
            download_dataset()

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

    return dataset, test, normalizer

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

dataset, test, normalizer = load_dataset()

lat = test.lat.values
lon = test.lon.values

STREAMLIT_BG = "#0E1117"

def plot_map(data, title, cmap, vmin=None, vmax=None, center=None, label=""):

    data = np.ma.masked_invalid(data)

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap).copy()

    cmap.set_bad(STREAMLIT_BG)
    fig, ax = plt.subplots(
        figsize=(6, 5),
        facecolor=STREAMLIT_BG
    )

    ax.set_facecolor(STREAMLIT_BG)

    if center is None:
        im = ax.pcolormesh(
            lon,
            lat,
            data,
            cmap=cmap,
            shading="auto",
            vmin=vmin,
            vmax=vmax
        )
    else:
        limit = max(abs(vmin), abs(vmax))

        im = ax.pcolormesh(
            lon,
            lat,
            data,
            cmap=cmap,
            shading="auto",
            vmin=-limit,
            vmax=limit,
        )
    add_up_cities(ax)
    ax.set_title(title, color="white", fontsize=14)
    ax.set_xlim(75.8, 85.7)
    ax.set_ylim(23.2, 31.0)
    ax.set_xlabel("Longitude", color="white", fontsize=11)
    ax.set_ylabel("Latitude", color="white", fontsize=11)

    ax.tick_params(
        colors="white",
        labelsize=10
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = plt.colorbar(
        im,
        ax=ax,
        fraction=0.045,
        pad=0.04
    )

    formatter = ScalarFormatter(useMathText=False)
    formatter.set_scientific(False)
    formatter.set_useOffset(False)

    cbar.ax.yaxis.set_major_formatter(formatter)
    cbar.ax.yaxis.get_offset_text().set_visible(False)

    cbar.update_ticks()

    cbar.ax.tick_params(colors="white")
    cbar.set_label(label, color="white")

    fig.tight_layout()

    return fig

# ----------------------------------------------------
# Simulation Settings
# ----------------------------------------------------

st.subheader("Simulation Settings")

forecast_dates = pd.to_datetime(
    test.time.values[WINDOW_SIZE:]
).date

forecast_options = {
    "1 Day": 1,
    "3 Days": 3,
    "5 Days": 5,
    "7 Days": 7
}

c1, c2, c3 = st.columns([2, 1, 1])

with c1:
    selected_date = st.date_input(
        "Forecast Date",
        value=forecast_dates[0],
        min_value=forecast_dates[0],
        max_value=forecast_dates[-1]
    )

with c2:
    selected = st.selectbox(
        "Forecast Horizon",
        list(forecast_options.keys())
    )

with c3:
    mc_samples = st.selectbox(
        "MC Samples",
        [10, 20, 30, 50],
        index=2
    )

# -------------------------------
# Row 2
# -------------------------------

c4, c5 = st.columns(2)

with c4:
    temperature = st.slider(
        "Temperature Change (°C)",
        min_value=-2.0,
        max_value=2.0,
        value=0.0,
        step=0.5,
        format="%.1f °C"
    )

with c5:
    rainfall = st.slider(
        "Rainfall Multiplier",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
        format="%.1f×"
    )

forecast_days = forecast_options[selected]

st.write("")

run = st.button(
    "Run Simulation",
    use_container_width=True
)

st.divider()

with st.expander("Simulation Guide"):

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

    forecast_days = forecast_options[selected]
    # Find the selected date in the dataset
    sample = np.where(
        forecast_dates == selected_date
    )[0][0]
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
    valid_mask = ~np.isnan(
        test["rainfall"].isel(time=0).values
    )

    base                 = np.ma.masked_where(~valid_mask, base)
    scen                 = np.ma.masked_where(~valid_mask, scen)
    base_uncertainty     = np.ma.masked_where(~valid_mask, base_uncertainty)
    scenario_uncertainty = np.ma.masked_where(~valid_mask, scenario_uncertainty)
    diff                 = np.ma.masked_where(~valid_mask, diff)

    # -------------------------
    # Maps
    # -------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        st.subheader("Baseline")

        fig = plot_map(
            base,
            "Baseline",
            cmap="Blues",
            vmin=vmin,
            vmax=vmax,
            label="mm/day"
        )

        st.pyplot(fig)
        plt.close()

    with c2:

        st.subheader("Scenario")

        fig = plot_map(
            scen,
            "Scenerio",
            cmap="Blues",
            vmin=vmin,
            vmax=vmax,
            label="mm/day"
        )

        st.pyplot(fig)

    with c3:

        st.subheader("Difference")
        limit = np.nanmax(np.abs(diff))
        fig = plot_map(
            diff,
            "Difference",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            label="mm/day"
        )

        st.pyplot(fig)

    st.divider()

    st.subheader("Prediction Uncertainty")

    u1, u2 = st.columns(2)

    with u1:

        st.subheader("Baseline Uncertainty")

        fig = plot_map(
            base_uncertainty,
            "Baseline Uncertainty",
            cmap="inferno",
            vmin=0,
            vmax=np.nanmax(base_uncertainty),
            label="Std. Dev."
        )

        st.pyplot(fig)

    with u2:

        st.subheader("Scenario Uncertainty")

        fig = plot_map(
            scenario_uncertainty,
            "Scenario Uncertainty",
            cmap="inferno",
            vmin=0,
            vmax=np.nanmax(scenario_uncertainty),
            label="Std. Dev."
        )

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
    
    baseline = normalizer.inverse_transform_array(
        base,
        "rainfall"
    )

    scenario = normalizer.inverse_transform_array(
        scen,
        "rainfall"
    )
    difference = scenario - base
    m1.metric(
        "Average Baseline Rainfall",
        f"{baseline.mean():.3f}"
    )

    m2.metric(
        "Average Scenario Rainfall",
        f"{scenario.mean():.3f}"
    )

    m3.metric(
        "Average Change",
        f"{(difference).mean():+.3f}"
    )

    m4, m5, m6 = st.columns(3)

    m4.metric(
        "Maximum Increase",
        f"{difference.max():.3f}"
    )

    m5.metric(
        "Maximum Decrease",
        f"{difference.min():.3f}"
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