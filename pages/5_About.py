import streamlit as st

st.set_page_config(
    page_title="About AGNI",
    layout="wide"
)

st.title("About AGNI")
st.caption("AI-Powered Climate Digital Twin for India")

st.markdown("""
AGNI (**Atmospheric and Geographic Neural Intelligence**) is an AI-powered
Climate Digital Twin that enables short-term rainfall forecasting,
scenario simulation, and uncertainty estimation using deep learning.

The system combines historical climate observations with ConvLSTM neural
networks to capture both spatial and temporal climate patterns. Users can
modify environmental conditions such as temperature and rainfall and
observe how the predicted rainfall distribution changes across the region.
""")

st.divider()

col1, col2 = st.columns([2,1])

with col1:

    st.header("Project Objectives")

    st.markdown("""
- Develop an AI-powered Digital Twin for regional climate analysis.
- Forecast rainfall using deep spatio-temporal deep learning.
- Simulate climate change scenarios interactively.
- Quantify prediction uncertainty using Monte Carlo Dropout.
- Provide an intuitive decision-support dashboard.
""")

with col2:

    st.info("""
**Project Name**

AGNI

Atmospheric and Geographic Neural Intelligence

**Pilot Region**

Uttar Pradesh, India

**Prediction Target**

Daily Rainfall
""")

st.divider()

st.header("System Architecture")

st.markdown("""
IMD Climate Data
│
▼
Data Cleaning & Preprocessing
│
▼
Feature Engineering
│
▼
ConvLSTM Deep Learning Model
│
▼
Monte Carlo Dropout
│
▼
Forecast Engine
│
▼
Digital Twin Simulator
│
▼
Interactive Streamlit Dashboard""")

st.divider()

st.header("Dataset")

st.markdown("""
The Digital Twin is trained using historical meteorological observations
from the **India Meteorological Department (IMD)**.

Variables include:

- Daily Rainfall
- Maximum Temperature
- Minimum Temperature

Additional engineered features include:

- Rainfall lags
- Rolling rainfall averages
- Temperature statistics
- Calendar features
- Rainfall anomalies
""")

st.divider()

st.header("Artificial Intelligence Model")

st.markdown("""
### ConvLSTM

The forecasting engine uses a Convolutional Long Short-Term Memory
(ConvLSTM) network that simultaneously models:

- Spatial relationships between neighbouring grid cells.
- Temporal evolution of weather patterns.
- Multi-day climate dynamics.

Unlike traditional LSTM models, ConvLSTM preserves the geographical
structure of climate fields by replacing matrix multiplications with
convolutional operations.
""")

st.divider()

st.header("Digital Twin Simulation")

st.markdown("""
The Digital Twin allows users to explore "what-if" scenarios by modifying:

- Temperature
- Rainfall intensity
- Forecast horizon

The modified climate state is propagated through the AI model to generate
new rainfall forecasts, enabling rapid scenario analysis.
""")

st.divider()

st.header("Prediction Uncertainty")

st.markdown("""
AGNI estimates prediction uncertainty using **Monte Carlo Dropout**.

Multiple stochastic forward passes are performed through the ConvLSTM
network to estimate the variability of predictions.

The dashboard displays:

- Baseline forecast
- Scenario forecast
- Difference map
- Uncertainty map
- Confidence score
""")

st.divider()

st.header("Technology Stack")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Data")
    st.markdown("""
- IMD Climate Data
- NetCDF
- Xarray
- NumPy
- Pandas
""")

with col2:
    st.subheader("Machine Learning")
    st.markdown("""
- PyTorch
- ConvLSTM
- Monte Carlo Dropout
- Recursive Forecasting
""")

with col3:
    st.subheader("Visualization")
    st.markdown("""
- Streamlit
- Matplotlib
- Climate Maps
- Interactive Dashboard
""")

st.divider()

st.header("Applications")

st.markdown("""
Potential applications include:

- Flood preparedness
- Drought monitoring
- Agricultural planning
- Water resource management
- Climate risk assessment
- Disaster management
""")

st.divider()

st.success("""
AGNI demonstrates how Artificial Intelligence and Digital Twin technology
can be integrated to create interactive climate forecasting systems that
support informed environmental decision-making.
""")