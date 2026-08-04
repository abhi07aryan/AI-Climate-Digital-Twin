from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
import torch
import xarray as xr
import gdown

from climate_twin.forecasting.recursive_forecast import RecursiveForecaster
from climate_twin.models.convlstm import ConvLSTM
from climate_twin.preprocessing.normalize import ClimateNormalizer
from climate_twin.preprocessing.split import TimeSeriesSplit
from climate_twin.ml.dataset import ClimateTorchDataset
from matplotlib.ticker import ScalarFormatter
from climate_twin.applications.flood import compute_flood_risk
from climate_twin.applications.drought import compute_drought



DATASET = Path("data/processed/climate_up_compressed.nc")
def download_dataset():
    DATASET.parent.mkdir(parents=True, exist_ok=True)

    file_id = "1Ld7oVZJ5XCFi6o8ZPZ0iM9vErvmQTmZu"

    gdown.download(
        id=file_id,
        output=str(DATASET),
        quiet=False
    )

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

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

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

MODEL = Path("models/convlstm_up_best.pth")

WINDOW_SIZE = 30

MC_SAMPLES = 30

FEATURES = [
    "rainfall",
    "tmax",
    "tmin"]
#     "temp_mean",
#     "temp_range",
#     "rain_7day",
#     "rain_lag1",
#     "rain_lag3",
#     "rain_lag7",
#     "month",
#     "season",
#     "dayofyear",
#     "rain_anomaly",
# ]

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

st.set_page_config(
    page_title="Rainfall Forecast",
    page_icon=" ",
    layout="wide"
)

st.title("AI Rainfall Forecasting")

with open(Path("assets/theme.css")) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
Predict daily rainfall using a trained ConvLSTM model.
Forecasts include uncertainty estimation using
Monte Carlo Dropout.
""")

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

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

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.to(DEVICE)

    return model

# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

@st.cache_data
def load_dataset():
    if not DATASET.exists():
        with st.spinner("Downloading dataset..."):
            download_dataset()
    ds = xr.open_dataset(DATASET)
    splitter = TimeSeriesSplit()

    train, valid, test = splitter.split(ds)

    normalizer = ClimateNormalizer()

    normalizer.fit(
        train,
        FEATURES
    )

    test = normalizer.transform(test)

    dataset = ClimateTorchDataset(
        test,
        input_features=FEATURES,
        target="rainfall",
        window_size=WINDOW_SIZE
    )

    forecast_dates = pd.to_datetime(
        test.time.values[WINDOW_SIZE:]
    ).date

    return dataset, forecast_dates, test, normalizer

model = load_model()

dataset, forecast_dates, test_ds, normalizer = load_dataset()
lat = test_ds.lat.values
lon = test_ds.lon.values
# ---------------------------------------------------------
# Monte Carlo Dropout
# ---------------------------------------------------------

def mc_predict(model, sequence, days, samples=MC_SAMPLES):

    model.eval()

    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

    predictions = []

    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

    with torch.no_grad():

        for _ in range(samples):

            forecaster = RecursiveForecaster(model, DEVICE)

            pred = forecaster.forecast(
                sequence.copy(),
                days
            )

            # Keep only the selected forecast day
            predictions.append(pred[-1])

    predictions = np.stack(predictions)

    mean_prediction = predictions.mean(axis=0)

    uncertainty = predictions.std(axis=0)

    confidence = 100 * np.exp(
        -np.nanmean(uncertainty) /
        (np.nanstd(mean_prediction) + 1e-6)
    )
    confidence = np.clip(confidence, 0, 100)
    return mean_prediction, uncertainty, confidence

STREAMLIT_BG = (17/255, 24/255, 39/255, 0.35)

def plot_map(data, title, cmap, vmin=None, vmax=None, center=None, label="", export=False, filename="map.png"):

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
    if export:

        import io

        buf = io.BytesIO()

        fig.savefig(
            buf,
            format="png",
            dpi=300,
            bbox_inches="tight"
        )

        st.download_button(
            "Download Map",
            data=buf.getvalue(),
            file_name=filename,
            mime="image/png"
        )
    return fig

# ---------------------------------------------------------
# Forecast Settings
# ---------------------------------------------------------

st.subheader("Forecast Settings")

c1, c2, c3, c4 = st.columns([2, 1, 1, 1])

with c1:
    selected_date = st.date_input(
        "Forecast Date",
        value=forecast_dates[0],
        min_value=forecast_dates[0],
        max_value=forecast_dates[-1]
    )

with c2:
    forecast_day = st.selectbox(
        "Forecast Day",
        range(1, 8)
    )

with c3:
    mc_samples = st.selectbox(
        "MC Samples",
        [10, 20, 30, 50],
        index=2
    )

with c4:
    st.write("")
    st.write("")
    run = st.button(
        "Run Forecast",
        use_container_width=True
    )

st.divider()

with st.expander("Forecast Guide"):

    st.markdown("""
### Forecast Date
Select the day for which rainfall will be predicted.

The model uses the previous **30 days**
of rainfall and temperature observations
as input.

### Forecast Horizon
Supports recursive forecasting up to **7 days**.
prediction.

### Monte Carlo Samples
The model performs multiple forward
passes with dropout enabled.

More samples produce smoother
uncertainty estimates but increase
computation time.

### Run Forecast
Generates:

- Ground Truth
- AI Prediction
- Prediction Error
- Uncertainty Map
- Confidence Score
""")

# ---------------------------------------------------------
# Prepare Prediction
# ---------------------------------------------------------

matches = np.where(forecast_dates == selected_date)[0]

if len(matches) == 0:
    st.error("Selected forecast date is unavailable.")
    st.stop()

sample = matches[0]

X, y = dataset[sample]

X = X.unsqueeze(0).to(DEVICE)

truth = np.nan_to_num(
    test_ds["rainfall"].isel(
        time=sample + WINDOW_SIZE + forecast_day - 1
    ).values,
    nan=0.0
)

# ---------------------------------------------------------
# Monte Carlo Prediction
# ---------------------------------------------------------

sequence = X.squeeze(0).cpu().numpy()

prediction, uncertainty, confidence = mc_predict(
    model,
    sequence,
    days=forecast_day,
    samples=mc_samples
)

difference = prediction - truth

valid_mask = ~np.isnan(truth)

difference_metric = (
    prediction[valid_mask]
    - truth[valid_mask]
)

rmse = np.sqrt(np.mean(difference_metric**2))
mae = np.mean(np.abs(difference_metric))

difference = prediction - truth

if np.std(truth) == 0 or np.std(prediction) == 0:
    correlation = np.nan
else:
    correlation = np.corrcoef(
        truth.flatten(),
        prediction.flatten()
    )[0, 1]

confidence_interval = (
    prediction.mean()
    - 1.96 * uncertainty.mean(),
    prediction.mean()
    + 1.96 * uncertainty.mean()
)

# ---------------------------------------------------------
# Forecast Summary
# ---------------------------------------------------------
from datetime import timedelta

predicted_date = selected_date + timedelta(days=forecast_day - 1)
st.subheader("Forecast Summary")

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric(
    "Forecast Date",
    str(selected_date)
)

c2.metric(
    "Prediction Date",
    str(predicted_date)
)

c3.metric(
    "Forecast Horizon",
    f"{forecast_day} Day{'s' if forecast_day > 1 else ''}"
)

c4.metric(
    "Average Rainfall",
    f"{prediction.mean():.2f}"
)

c5.metric(
    "Confidence",
    f"{confidence:.1f}%"
)

c6.metric(
    "RMSE",
    f"{rmse:.4f}"
)

forecast_tab, flood_tab, drought_tab = st.tabs(
    [
        "Rainfall Forecast",
        "Flood Preparedness",
        "Drought Monitoring"
    ]
)
cmap = plt.cm.Blues.copy()
cmap.set_bad(STREAMLIT_BG)

cmap.set_bad(color=STREAMLIT_BG)

rainfall = prediction
flood = compute_flood_risk(truth, normalizer)
tmax = X[0, -1, FEATURES.index("tmax")].cpu().numpy()
tmin = X[0, -1, FEATURES.index("tmin")].cpu().numpy()
drought = compute_drought(truth, tmax, tmin, normalizer)

valid_mask = ~np.isnan(
    test_ds["rainfall"].isel(time=0).values
)

prediction = np.ma.masked_where(~valid_mask, prediction)
truth      = np.ma.masked_where(~valid_mask, truth)
flood      = np.ma.masked_where(~valid_mask, flood)
drought    = np.ma.masked_where(~valid_mask, drought)

# ---------------------------------------------------------
# Prediction Maps
# ---------------------------------------------------------

st.divider()

vmin = min(
    np.nanmin(truth),
    np.nanmin(prediction)
)

vmax = max(
    np.nanmax(truth),
    np.nanmax(prediction)
)


with forecast_tab:

    # ---------------------------------------------------------
    # Ground Truth
    # ---------------------------------------------------------

    st.title("Climate Forecast")
    col1, col2 = st.columns(2)

    with col1:

        st.subheader(f"Ground Truth (Day {forecast_day})")

        fig = plot_map(
            truth,
            f"Observed Rainfall (Day {forecast_day})",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            label="mm/day",
            export=True,
            filename=f"ground_truth_day_{predicted_date}.png"
        )

        st.pyplot(fig)

        plt.close(fig)

    # ---------------------------------------------------------
    # Mean Prediction
    # ---------------------------------------------------------

    with col2:

        st.subheader(f"Prediction (Day {forecast_day})")

        fig = plot_map(
            prediction,
            f"Predicted Rainfall (Day {forecast_day})",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            label="mm/day",
            export=True,
            filename=f"prediction_day_{predicted_date}.png"
        )

        st.pyplot(fig)

        plt.close(fig)

    # ---------------------------------------------------------
    # Error & Uncertainty
    # ---------------------------------------------------------

    st.divider()

    col3, col4 = st.columns(2)

    # ---------------------------------------------------------
    # Error Map
    # ---------------------------------------------------------

    with col3:

        st.subheader("Prediction Error")

        limit = np.nanmax(np.abs(difference))

        fig = plot_map(
            difference,
            "Prediction Error",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            center=0,
            label="mm/day",
            export=True,
            filename=f"difference_day_{predicted_date}.png"
        )

        st.pyplot(fig)

        plt.close(fig)

    # ---------------------------------------------------------
    # Uncertainty
    # ---------------------------------------------------------

    with col4:

        st.subheader("Prediction Uncertainty")

        fig = plot_map(
            uncertainty,
            "Model Uncertainty",
            cmap="inferno",
            vmin=0,
            vmax=np.nanmax(uncertainty),
            label="Std. Dev.",
            export=True,
            filename=f"uncertainty_day_{predicted_date}.png"
        )

        st.pyplot(fig)
        plt.close(fig)

with flood_tab:

    st.subheader("Flood Preparedness")
    flood_min = np.nanmin(flood)
    flood_max = np.nanmax(flood)

    fig = plot_map(
        flood,
        "Flood Risk",
        cmap="Reds",
        vmin=flood_min,
        vmax=flood_max,
        label="Risk",
        export=True,
        filename=f"flood_risk_day_{predicted_date}.png"
    )

    st.pyplot(fig)
    plt.close(fig)

with drought_tab:

    st.subheader("Drought Monitoring")
    drought_min = np.nanmin(drought)
    drought_max = np.nanmax(drought)
    fig = plot_map(
        drought,
        "Drought Risk",
        cmap="YlOrBr",
        vmin=drought_min,
        vmax=drought_max,
        label="Risk",
        export=True,
        filename=f"drought_day_{predicted_date}.png"
    )

    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------

st.divider()

st.subheader("Evaluation Metrics")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "RMSE",
    f"{rmse:.4f}"
)

c2.metric(
    "MAE",
    f"{mae:.4f}"
)

c3.metric(
    "Correlation",
    f"{correlation:.3f}"
)

c4.metric(
    "Confidence",
    f"{confidence:.1f}%"
)

# ---------------------------------------------------------
# Confidence Interval
# ---------------------------------------------------------

st.subheader("Forecast Confidence")

st.info(
    f"""
Average Predicted Rainfall : **{prediction.mean():.2f}**

95% Confidence Interval :

**{confidence_interval[0]:.2f}**
to
**{confidence_interval[1]:.2f}**
"""
)

# ---------------------------------------------------------
# Prediction Distribution
# ---------------------------------------------------------

st.divider()

st.subheader("Prediction Distribution")

fig, ax = plt.subplots(figsize=(8,4), facecolor=STREAMLIT_BG)

fig.patch.set_facecolor(STREAMLIT_BG)
ax.set_facecolor(STREAMLIT_BG)

ax.hist(
    prediction.flatten(),
    bins=40,
    edgecolor="white",
    color="#4C9BE8"
)

# White title and labels
ax.set_title(
    "Distribution of Predicted Rainfall",
    color="white",
    fontsize=16
)

ax.set_xlabel(
    "Predicted Rainfall (mm/day)",
    color="white",
    fontsize=12
)

ax.set_ylabel(
    "Grid Cells",
    color="white",
    fontsize=12
)

# White tick labels
ax.tick_params(
    axis="both",
    colors="white",
    labelsize=11
)

# White spines
for spine in ax.spines.values():
    spine.set_color("white")

# Light grid
ax.grid(
    alpha=0.25,
    color="white"
)

st.pyplot(fig)
plt.close(fig)

# ---------------------------------------------------------
# Forecast Interpretation
# ---------------------------------------------------------

st.divider()

st.subheader("Forecast Interpretation")

avg = prediction.mean()

if avg < -0.5:
    rainfall_level = "Very Low Rainfall"
elif avg < 0:
    rainfall_level = "Low Rainfall"
elif avg < 0.5:
    rainfall_level = "Moderate Rainfall"
elif avg < 1:
    rainfall_level = "High Rainfall"
else:
    rainfall_level = "Very High Rainfall"

if confidence > 95:
    confidence_level = "Very High"
elif confidence > 90:
    confidence_level = "High"
elif confidence > 80:
    confidence_level = "Moderate"
else:
    confidence_level = "Low"

st.success(
    f"""
### Forecast Summary

• Expected rainfall: **{rainfall_level}**

• Average predicted rainfall: **{prediction.mean():.2f}**

• Model confidence: **{confidence_level} ({confidence:.1f}%)**

• RMSE: **{rmse:.4f}**

• MAE: **{mae:.4f}**
"""
)


# ---------------------------------------------------------
# What do the maps mean?
# ---------------------------------------------------------

st.divider()

with st.expander("Understanding the Maps"):

    st.markdown("""
### Ground Truth
Observed rainfall from the IMD dataset for the selected forecast date.

### AI Prediction
Rainfall predicted by the ConvLSTM model using the previous 30 days of climate observations.

### Prediction Error
Difference between prediction and observation.

- Blue → Underprediction
- White → Close agreement
- Red → Overprediction

### Prediction Uncertainty
Estimated using Monte Carlo Dropout.

Brighter colours indicate regions where the model is less certain about its prediction.
""")

# ---------------------------------------------------------
# Understanding the Metrics
# ---------------------------------------------------------

with st.expander("Understanding the Metrics"):

    st.markdown("""
### RMSE
Root Mean Squared Error.

Measures the average prediction error while giving more weight to larger errors.

Lower values indicate better performance.

---

### MAE
Mean Absolute Error.

Average absolute difference between predicted and observed rainfall.

Lower is better.

---

### Correlation
Measures how closely the spatial rainfall pattern matches observations.

Closer to **1** indicates stronger agreement.

---

### Confidence
Estimated using Monte Carlo Dropout.

Higher confidence indicates more consistent predictions across multiple stochastic forward passes.

---

### 95% Confidence Interval
The expected range within which the predicted rainfall is likely to lie based on the model's uncertainty estimate.
""")

# ---------------------------------------------------------
# Model Information
# ---------------------------------------------------------

st.divider()

with st.expander("About this Forecast"):

    st.markdown(f"""
**Model**

ConvLSTM (Convolutional Long Short-Term Memory)

**Input Variables**

- Rainfall
- Maximum Temperature
- Minimum Temperature

**Input Window**

{WINDOW_SIZE} days

**Forecast Horizon**

1-7 days (recursive)

**Monte Carlo Samples**

{mc_samples}

**Training Period**

2010-2020

**Validation Period**

2021-2022

**Testing Period**

2023-2024

**Spatial Resolution**

129 × 135 grid
""")