from pathlib import Path

import streamlit as st
import torch
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from climate_twin.models.convlstm import ConvLSTM
from climate_twin.preprocessing.normalize import ClimateNormalizer
from climate_twin.preprocessing.split import TimeSeriesSplit
from climate_twin.ml.dataset import ClimateTorchDataset

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

DATASET = Path("data/processed/climate_up.nc")
MODEL = Path("models/convlstm_best.pth")

WINDOW_SIZE = 7

FEATURES = [
    "rainfall",
    "tmax",
    "tmin"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

st.set_page_config(
    page_title="Forecasting",
    layout="wide"
)

st.title("🌧 Rainfall Forecast")

# -------------------------------------------------------
# Load Model
# -------------------------------------------------------

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

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

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

    return dataset

model = load_model()

dataset = load_dataset()

# -------------------------------------------------------
# Sample Selector
# -------------------------------------------------------

sample = st.slider(
    "Forecast Sample",
    0,
    len(dataset)-1,
    0
)

# -------------------------------------------------------
# Predict
# -------------------------------------------------------

X, y = dataset[sample]

with torch.no_grad():

    pred = model(
        X.unsqueeze(0).to(DEVICE)
    )

prediction = pred.squeeze().cpu().numpy()

truth = y.squeeze().numpy()

# -------------------------------------------------------
# Display
# -------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("Ground Truth")

    fig, ax = plt.subplots(figsize=(5,5))

    im = ax.imshow(
        truth,
        cmap="Blues"
    )

    ax.axis("off")

    plt.colorbar(im)

    st.pyplot(fig)

with col2:

    st.subheader("Prediction")

    fig, ax = plt.subplots(figsize=(5,5))

    im = ax.imshow(
        prediction,
        cmap="Blues"
    )

    ax.axis("off")

    plt.colorbar(im)

    st.pyplot(fig)

# -------------------------------------------------------
# Difference
# -------------------------------------------------------

st.subheader("Difference")

difference = prediction - truth

fig, ax = plt.subplots(figsize=(6,6))

im = ax.imshow(
    difference,
    cmap="RdBu_r"
)

ax.axis("off")

plt.colorbar(im)

st.pyplot(fig)

# -------------------------------------------------------
# Statistics
# -------------------------------------------------------

st.subheader("Prediction Statistics")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Prediction Mean",
    f"{prediction.mean():.4f}"
)

c2.metric(
    "Ground Truth Mean",
    f"{truth.mean():.4f}"
)

c3.metric(
    "Mean Error",
    f"{difference.mean():.4f}"
)