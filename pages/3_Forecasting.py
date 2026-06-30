from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
import torch
import xarray as xr

from climate_twin.models.convlstm import ConvLSTM
from climate_twin.preprocessing.normalize import ClimateNormalizer
from climate_twin.preprocessing.split import TimeSeriesSplit
from climate_twin.ml.dataset import ClimateTorchDataset

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATASET = Path("data/processed/climate_up.nc")
MODEL = Path("models/convlstm_best.pth")

WINDOW_SIZE = 30

MC_SAMPLES = 30

FEATURES = [
    "rainfall",
    "tmax",
    "tmin"
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

    return dataset, forecast_dates

model = load_model()

dataset, forecast_dates = load_dataset()

# ---------------------------------------------------------
# Monte Carlo Dropout
# ---------------------------------------------------------

def mc_predict(
    model,
    x,
    samples=MC_SAMPLES
):

    model.train()

    predictions = []

    with torch.no_grad():

        for _ in range(samples):

            pred = model(x)

            predictions.append(
                pred.squeeze().cpu().numpy()
            )

    predictions = np.stack(predictions)

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

# ---------------------------------------------------------
# Forecast Settings
# ---------------------------------------------------------

st.sidebar.header("Forecast Settings")

selected_date = st.sidebar.date_input(
    "Forecast Date",
    value=forecast_dates[0],
    min_value=forecast_dates[0],
    max_value=forecast_dates[-1]
)

forecast_horizon = st.sidebar.selectbox(
    "Forecast Horizon",
    [1],
    index=0,
    help="Current model predicts one day ahead."
)

mc_samples = st.sidebar.selectbox(
    "Monte Carlo Samples",
    [10, 20, 30, 50],
    index=2,
    help="Number of stochastic forward passes used to estimate prediction uncertainty."
)

run = st.sidebar.button(
    "Run Forecast",
    use_container_width=True
)

st.sidebar.markdown("---")

with st.sidebar.expander("Forecast Guide"):

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

sample = np.where(
    forecast_dates == selected_date
)[0][0]

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

st.markdown("---")

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

with col1:

    st.markdown("### Ground Truth")

    fig, ax = plt.subplots(
        figsize=(6,6),
        facecolor="#1E1E1E"
    )

    im = ax.imshow(
        truth,
        cmap="Blues",
        vmin=vmin,
        vmax=vmax
    )

    ax.axis("off")

    plt.colorbar(
        im,
        shrink=0.8
    )

    st.pyplot(fig)

# ---------------------------------------------------------
# Mean Prediction
# ---------------------------------------------------------
import matplotlib.pyplot as plt

cmap = plt.cm.Blues.copy()

cmap.set_bad(color="#3F94EF")   # Light gray

with col2:

    st.markdown("### AI Prediction")

    fig, ax = plt.subplots(figsize=(6,6))

    fig.patch.set_facecolor("#0E1117")    
    ax.set_facecolor("#20232A")       

    cmap = plt.cm.Blues.copy()
    cmap.set_bad("#20232A")

    im = ax.imshow(
        prediction,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    ax.axis("off")

    plt.colorbar(
        im,
        shrink=0.8
    )

    st.pyplot(fig)

# ---------------------------------------------------------
# Error & Uncertainty
# ---------------------------------------------------------

st.markdown("---")

col3, col4 = st.columns(2)

# ---------------------------------------------------------
# Error Map
# ---------------------------------------------------------

with col3:

    st.markdown("### Prediction Error")

    limit = np.max(
        np.abs(difference)
    )

    fig, ax = plt.subplots(figsize=(6,6))

    im = ax.imshow(
        difference,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit
    )

    ax.axis("off")

    plt.colorbar(
        im,
        shrink=0.8
    )

    st.pyplot(fig)

# ---------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------

with col4:

    st.markdown("### Prediction Uncertainty")

    fig, ax = plt.subplots(figsize=(6,6))

    im = ax.imshow(
        uncertainty,
        cmap="inferno"
    )

    ax.axis("off")

    plt.colorbar(
        im,
        shrink=0.8
    )

    st.pyplot(fig)
# ---------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------

st.markdown("---")

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

st.markdown("---")

st.subheader("Prediction Distribution")

fig, ax = plt.subplots(figsize=(8,4))

ax.hist(
    prediction.flatten(),
    bins=40,
    edgecolor="black"
)

ax.set_xlabel("Predicted Rainfall")

ax.set_ylabel("Grid Cells")

ax.set_title("Distribution of Predicted Rainfall")

st.pyplot(fig)

# ---------------------------------------------------------
# Forecast Interpretation
# ---------------------------------------------------------

st.markdown("---")

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

st.markdown("---")

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

st.markdown("---")

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