from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Model Evaluation",
    layout="wide"
)

st.title("Model Evaluation")

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

RESULTS = Path("results")
EVAL = RESULTS / "evaluation"

metrics_file = EVAL / "metrics.csv"
history_file = RESULTS / "history.csv"

prediction_file = EVAL / "prediction.png"
error_file = EVAL / "error.png"
histogram_file = EVAL / "histogram.png"

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

st.header("Evaluation Metrics")

if metrics_file.exists():

    metrics = pd.read_csv(metrics_file)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "RMSE",
        f"{metrics['RMSE'][0]:.4f}"
    )

    c2.metric(
        "MAE",
        f"{metrics['MAE'][0]:.4f}"
    )

    c3.metric(
        "R²",
        f"{metrics['R2'][0]:.4f}"
    )

    c4.metric(
        "Correlation",
        f"{metrics['Correlation'][0]:.4f}"
    )

else:

    st.warning(
        "Run evaluate.py first to generate metrics."
    )

st.divider()

# ------------------------------------------------------------
# Training Curve
# ------------------------------------------------------------

st.header("Training History")

if history_file.exists():

    history = pd.read_csv(history_file)

    fig, ax = plt.subplots(figsize=(8,4))

    ax.plot(
        history["epoch"],
        history["train_loss"],
        label="Train"
    )

    ax.plot(
        history["epoch"],
        history["valid_loss"],
        label="Validation"
    )

    ax.set_xlabel("Epoch")

    ax.set_ylabel("Loss")

    ax.grid(True)

    ax.legend()

    st.pyplot(fig)

else:

    st.info(
        "history.csv not found."
    )

st.divider()

# ------------------------------------------------------------
# Prediction Images
# ------------------------------------------------------------

st.header("Prediction Results")

col1, col2 = st.columns(2)

with col1:

    st.subheader("Prediction")

    if prediction_file.exists():

        st.image(
            Image.open(prediction_file),
            use_container_width=True
        )

    else:

        st.info("prediction.png not found.")

with col2:

    st.subheader("Error Map")

    if error_file.exists():

        st.image(
            Image.open(error_file),
            use_container_width=True
        )

    else:

        st.info("error.png not found.")

st.divider()

# ------------------------------------------------------------
# Error Histogram
# ------------------------------------------------------------

st.header("Prediction Error Distribution")

if histogram_file.exists():

    st.image(
        Image.open(histogram_file),
        use_container_width=True
    )

else:

    st.info("histogram.png not found.")