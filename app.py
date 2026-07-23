import streamlit as st

st.set_page_config(
    page_title="AI Climate Digital Twin",
    page_icon=" ",
    layout="wide"
)

st.title("AI Climate Digital Twin")

st.markdown(
"""
Welcome!

This dashboard demonstrates an AI-powered Climate Digital Twin
for rainfall forecasting over Uttar Pradesh.

Use the navigation menu on the left.
"""
)

# st.image(
#     "assets/banner.png",
#     use_container_width=True
# )

st.markdown("---")

st.subheader("Modules")

st.write("Data Explorer")

st.write("Model Evaluation")

st.write("Rainfall Forecast")

st.write("Digital Twin Simulation")

st.markdown("---")

st.write("Built using")

st.write("- PyTorch")

st.write("- xarray")

st.write("- ConvLSTM")

st.write("- Streamlit")