import streamlit as st

st.set_page_config(
    page_title="Model Information",
    layout="wide"
)

st.title("Model Information")

st.header("Overview")

st.write("""
This project develops an AI-powered Climate Digital Twin for
rainfall forecasting over Uttar Pradesh.

The forecasting engine uses a ConvLSTM neural network that
captures both spatial and temporal climate patterns.
""")

st.divider()

st.header("Dataset")

st.write("""
• Rainfall (IMD)

• Maximum Temperature

• Minimum Temperature

• Uttar Pradesh Boundary

• Daily Resolution
""")

st.divider()

st.header("Model")

st.code("""
Input

7 × 3 × 129 × 135

↓

ConvLSTM

↓

1 × 129 × 135

Rainfall Prediction
""")

st.divider()

st.header("Features")

st.write("""
• Rainfall

• Tmax

• Tmin
""")

st.divider()

st.header("Training")

st.write("""
Loss Function : MSE

Optimizer : Adam

Scheduler : ReduceLROnPlateau

Framework : PyTorch
""")

st.divider()

st.header("Evaluation")

st.write("""
• RMSE

• MAE

• R²

• Correlation
""")