from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
import torch
import xarray as xr
import gdown

from climate_twin.models.convlstm import ConvLSTM
from climate_twin.preprocessing.normalize import ClimateNormalizer
from climate_twin.preprocessing.split import TimeSeriesSplit
from climate_twin.ml.dataset import ClimateTorchDataset

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATASET = Path("data/processed/climate_up.nc")
def download_dataset():
    DATASET.parent.mkdir(parents=True, exist_ok=True)

    file_id = "13lBEsLoVTmgFnEIOJXYiXMc6QAKch71k"

    gdown.download(
        id=file_id,
        output=str(DATASET),


        quiet=False
    )

if not DATASET.exists():
    with st.spinner("Downloading dataset..."):
        download_dataset()
    
MODEL = Path("models/convlstm_up_best.pth")

WINDOW_SIZE = 30

MC_SAMPLES = 30

FEATURES = [
    "rainfall",
    "tmax",
    "tmin",
    "temp_mean",
    "temp_range",
    "rain_7day",
    "rain_lag1",
    "rain_lag3",
    "rain_lag7",
    "month",
    "season",
    "dayofyear",
    "rain_anomaly",
]

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
        hidden_channels=32,
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

    return dataset, forecast_dates, test

model = load_model()

dataset, forecast_dates, test_ds = load_dataset()
lat = test_ds.lat.values
lon = test_ds.lon.values

# ---------------------------------------------------------
# Monte Carlo Dropout
# ---------------------------------------------------------

def mc_predict(
    model,
    x,
    samples=MC_SAMPLES
):

    model.eval()

    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

    predictions = []

    with torch.no_grad():
        for _ in range(samples):
            pred = model(x)
            predictions.append(pred.squeeze().cpu().numpy())

    predictions = np.stack(predictions)

    mean_prediction = predictions.mean(axis=0)
    std_prediction = predictions.std(axis=0)
    confidence = np.exp(-std_prediction.mean()) * 100

    return mean_prediction, std_prediction, confidence

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

    from matplotlib.ticker import ScalarFormatter

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
    forecast_horizon = st.selectbox(
        "Forecast Horizon",
        [1]
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
Currently supports **1-day ahead**
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

truth = y.squeeze().numpy()

# ---------------------------------------------------------
# Monte Carlo Prediction
# ---------------------------------------------------------

if run:

    prediction, uncertainty, confidence = mc_predict(
        model,
        X,
        samples=mc_samples
    )

else:

    prediction, uncertainty, confidence = mc_predict(
        model,
        X,
        samples=MC_SAMPLES
    )

difference = prediction - truth

rmse = np.sqrt(
    np.mean(
        difference ** 2
    )
)

mae = np.mean(
    np.abs(
        difference
    )
)

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

st.subheader("Forecast Summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Forecast Date",
    str(selected_date)
)

c2.metric(
    "Average Rainfall",
    f"{prediction.mean():.2f}"
)

c3.metric(
    "Confidence",
    f"{confidence:.1f}%"
)

c4.metric(
    "RMSE",
    f"{rmse:.4f}"
)

# ---------------------------------------------------------
# Prediction Maps
# ---------------------------------------------------------

st.divider()

st.subheader("Forecast Visualisation")

vmin = min(
    np.nanmin(truth),
    np.nanmin(prediction)
)

vmax = max(
    np.nanmax(truth),
    np.nanmax(prediction)
)

col1, col2 = st.columns(2)

# ---------------------------------------------------------
# Ground Truth
# ---------------------------------------------------------

cmap = plt.cm.Blues.copy()
cmap.set_bad(STREAMLIT_BG)

cmap.set_bad(color=STREAMLIT_BG)


with col1:

    st.subheader("Ground Truth")

    fig = plot_map(
        truth,
        "Observed Rainfall",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        label="mm/day"
    )

    st.pyplot(fig)

    plt.close(fig)

plt.close(fig)

# ---------------------------------------------------------
# Mean Prediction
# ---------------------------------------------------------

with col2:

    st.subheader("AI Prediction")

    fig = plot_map(
        prediction,
        "Predicted Rainfall",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        label="mm/day"
    )

    st.pyplot(fig)

    plt.close(fig)

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
        label="mm/day"
    )

    st.pyplot(fig)

    plt.close(fig)

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
        label="Std. Dev."
    )

    st.pyplot(fig)

    plt.close(fig)

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

with st.expander("🗺️ Understanding the Maps"):

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

with st.expander("🤖 About this Forecast"):

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

1 day

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