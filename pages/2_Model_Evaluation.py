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
st.caption(
    "Evaluate the ConvLSTM model using quantitative metrics and visual comparisons."
)

col1, col2, col3 = st.columns([2,2,2])

with col1:
    show_history = st.checkbox(
        "Training History",
        value=True,
        key="history"
    )

with col2:
    show_maps = st.checkbox(
        "Prediction Maps",
        value=True,
        key="maps"
    )

with col3:
    show_hist = st.checkbox(
        "Error Histogram",
        value=True,
        key="hist"
    )


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

    with st.expander("What do these metrics mean?"):
        st.markdown("""
    **RMSE**
    - Penalizes large prediction errors.
    - Lower is better.

    **MAE**
    - Average absolute prediction error.
    - Easier to interpret than RMSE.

    **R² Score**
    - Measures how well predictions explain observed variability.
    - Values closer to 1 indicate better performance.

    **Correlation**
    - Indicates agreement between predicted and observed spatial patterns.
    - Higher values are better.
    """)

else:

    st.warning(
        "Run evaluate.py first to generate metrics."
    )

st.divider()

# ------------------------------------------------------------
# Training Curve
# ------------------------------------------------------------
if show_hist:
    st.header("Training History")

    if history_file.exists():

        history = pd.read_csv(history_file)

        STREAMLIT_BG = "#0E1117"

        fig, ax = plt.subplots(
            figsize=(8,4),
            facecolor=STREAMLIT_BG
        )

        ax.set_facecolor(STREAMLIT_BG)

        ax.tick_params(colors="white")

        ax.set_xlabel("Epochs",color="white")
        ax.set_ylabel("Loss",color="white")

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.plot(
            history["epoch"],
            history["train_loss"],
            linewidth=2,
            label="Training"
        )

        ax.plot(
            history["epoch"],
            history["valid_loss"],
            linewidth=2,
            label="Validation"
        )

        ax.set_title(
            "Training & Validation Loss",
            color="white"
        )

        ax.grid(alpha=0.3)

        legend = ax.legend(frameon=False)

        for text in legend.get_texts():
            text.set_color("white")

        left, center, right = st.columns([1, 4, 1])

        with center:
            st.pyplot(fig)

    else:

        st.info(
            "history.csv not found."
        )

st.divider()

# ------------------------------------------------------------
# Prediction Images
# ------------------------------------------------------------

if show_maps:

    st.header("Prediction Results")

    col1, col2 = st.columns([2,1])

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

with st.expander("About these visualizations"):

    st.markdown("""
- **Prediction:** Rainfall predicted by the ConvLSTM model.
- **Prediction Error:** Difference between predicted and observed rainfall.
- Blue shades indicate underprediction, while red shades indicate overprediction.
""")

    st.divider()

# ------------------------------------------------------------
# Error Histogram
# ------------------------------------------------------------

if show_hist:
    st.header("Prediction Error Distribution")

    if histogram_file.exists():

        left, center, right = st.columns([1,4,1])

        with center:
            st.image(
                Image.open(histogram_file),
                use_container_width=True
            )
        st.caption(
        "Distribution of prediction errors across the evaluation dataset."
        )
    else:
        st.info("histogram.png not found.")

    st.divider()

    st.header("Evaluation Information")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Model",
        "ConvLSTM"
    )

    c2.metric(
        "Forecast Horizon",
        "1 Day"
    )

    c3.metric(
        "Grid Size",
        "129 × 135"
    )

    c4.metric(
        "Metrics",
        "RMSE / MAE / R²"
    )

    st.caption(
        "Performance metrics are computed using the held-out test dataset."
    )
