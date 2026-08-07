from pathlib import Path

import gdown
import io
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import xarray as xr
import pandas as pd

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

# Colormap for each variable
CMAPS = {
    # Rainfall variables
    "rainfall": "Blues",
    "rain_7day": "Blues",
    "rain_anomaly": "RdBu_r",

    # Temperature variables
    "tmax": "coolwarm",
    "tmin": "coolwarm",
    "temp_mean": "coolwarm",
    "temp_range": "plasma",
}

COLOR_LIMITS = {
    "rainfall": (0, 60),
    "rain_7day": (0, 150),
    "rain_anomaly": (-20, 20),

    "tmax": (20, 45),
    "tmin": (5, 30),
    "temp_mean": (15, 40),
    "temp_range": (5, 25),
}

LABELS = {
    "rainfall": "Rainfall (mm/day)",
    "rain_7day": "7-Day Rainfall (mm)",
    "rain_anomaly": "Rainfall Anomaly (mm)",

    "tmax": "Maximum Temperature (°C)",
    "tmin": "Minimum Temperature (°C)",
    "temp_mean": "Mean Temperature (°C)",
    "temp_range": "Temperature Range (°C)",
}

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET = Path("data/processed/climate_up_compressed.nc")
def download_dataset():
    DATASET.parent.mkdir(parents=True, exist_ok=True)

    file_id = "1jqMmwyXjB0gxQpMvNdOx283-YrKJo9dq"

    gdown.download(
        id=file_id,
        output=str(DATASET),
        quiet=False
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

st.set_page_config(
    page_title="Data Explorer",
    layout="wide"
)

st.title("Climate Data Explorer")

with open(Path("assets/theme.css")) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.caption(
    "Explore climate variables across the region through interactive spatial maps, temporal trends, and statistical summaries."
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

@st.cache_data
def load_dataset():
    if not DATASET.exists():
        with st.spinner("Downloading dataset..."):
            download_dataset()
    return xr.open_dataset(DATASET)

ds = load_dataset()

variables = sorted(list(ds.data_vars))

# --------------------------------------------------
# Variables
# --------------------------------------------------

exclude = {
    "month",
    "dayofyear",
}

preferred_order = [
    "rainfall",
    "tmax",
    "tmin",
    "temp_mean",
    "temp_range",
    "rain_7day",
    "rain_anomaly",
]

VARIABLE_VALIDITY = {
    "rainfall": 0,
    "tmax": 0,
    "tmin": 0,
    "temp_mean": 0,
    "temp_range": 0,
    "rain_7day": 7,
    "rain_anomaly": 30,
}

VAR_DESCRIPTIONS = {
    "rainfall": "Daily observed rainfall (mm/day).",
    "tmax": "Daily maximum air temperature (°C).",
    "tmin": "Daily minimum air temperature (°C).",
    "temp_mean": "Average daily temperature computed from Tmax and Tmin (°C).",
    "temp_range": "Difference between Tmax and Tmin (°C).",
    "rain_7day": "Cumulative rainfall over the previous seven days (mm).",
    "rain_anomaly": "Deviation of daily rainfall from its climatological average (mm/day).",
}

available_dates = pd.to_datetime(ds.time.values).date

st.subheader("Climate Data Explorer")

col1, col_from, col_to, col3, col4 = st.columns(
    [3, 1.5, 1.5, 1.5, 1.5],
    vertical_alignment="bottom"
)

DEFAULT_START = pd.Timestamp("2024-07-25").date()
DEFAULT_END = pd.Timestamp("2024-07-25").date()

# -----------------------------
# Date Range Selector (From / To)
# -----------------------------
with col_from:
    start_date = st.date_input(
        "From",
        value=DEFAULT_START,
        min_value=available_dates[0],
        max_value=available_dates[-1],
        key="date_from",
    )

with col_to:
    end_date = st.date_input(
        "To",
        value=DEFAULT_END,
        min_value=available_dates[0],
        max_value=available_dates[-1],
        key="date_to",
    )

days_since_start = (
    pd.Timestamp(start_date)
    - pd.Timestamp(ds.time.values[0])
).days

# -----------------------------
# Valid Variables
# -----------------------------
variables = [
    var
    for var in preferred_order
    if (
        var in ds.data_vars
        and var not in exclude
        and days_since_start >= VARIABLE_VALIDITY.get(var, 0)
    )
]

if not variables:
    st.warning("No variables are available for the selected date range.")
    st.stop()

# -----------------------------
# Variable Selector
# -----------------------------
with col1:
    variable = st.selectbox(
        "Parameter",
        variables[:3],
        key="parameter"
    )

c1,c2,c3,c4 = st.columns(4)

if start_date > end_date:
    st.error("The start date must be earlier than or equal to the end date.")
    st.stop()

if start_date == end_date:
    c1.metric("Selected Date: ", str(start_date))
else:
    c1.metric("Date Range: ", f"{start_date} → {end_date}")

map_tab, graph_tab = st.tabs(["Map visuals", "Graph visuals"])

# -----------------------------
# Display Options
# -----------------------------

with col3:
    show_stats = st.checkbox(
        "Statistics",
        value=True,
        key="show_stats"
    )
with col4:
    show_hist = st.checkbox(
        "Histogram",
        value=True,
        key="show_hist"
    )

# -----------------------------
# Time Index / Range Mask
# -----------------------------
try:
    start_index = available_dates.tolist().index(start_date)
    end_index = available_dates.tolist().index(end_date)
except ValueError:
    st.error("Selected date range is not available.")
    st.stop()

time_index = end_index  # used for single-point references (e.g. "latest" marker)

time_mask = (
    (ds.time.values >= np.datetime64(start_date))
    & (ds.time.values <= np.datetime64(end_date))
)

data = ds[variable]
data_range = data.sel(time=slice(str(start_date), str(end_date)))

# --------------------------------------------------
# Spatial Variables
# --------------------------------------------------
cmap_name = CMAPS.get(variable, "viridis")

STREAMLIT_BG = (17/255, 24/255, 39/255, 0.35)
with map_tab:

    if data.ndim == 3:

        # Average over the selected date range. If a single day is selected,
        # this reduces to that day's map exactly as before.
        image = data_range.mean(dim="time", skipna=True)

        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(STREAMLIT_BG)

        mask = np.isfinite(image.values)

        lat = ds.lat.values
        lon = ds.lon.values

        valid_rows = np.where(mask.any(axis=1))[0]
        valid_cols = np.where(mask.any(axis=0))[0]

        extent = [
            lon[valid_cols[0]],
            lon[valid_cols[-1]],
            lat[valid_rows[0]],
            lat[valid_rows[-1]]
        ]

        fig, ax = plt.subplots(
            figsize=(6, 5),
            facecolor=STREAMLIT_BG
        )

        ax.set_facecolor(STREAMLIT_BG)

        lon2d, lat2d = np.meshgrid(
            lon[valid_cols[0]:valid_cols[-1]+1],
            lat[valid_rows[0]:valid_rows[-1]+1]
        )
        if variable in ["rainfall", "rain_7day"]:

            values = image.values[np.isfinite(image.values)]

            vmin = 0
            vmax = np.percentile(values, 99)

            # Prevent a tiny range on dry days
            vmax = max(vmax, 5)

        else:
            vmin, vmax = COLOR_LIMITS.get(variable, (None, None))
        im = ax.pcolormesh(
            lon2d,
            lat2d,
            image.values[
                valid_rows[0]:valid_rows[-1]+1,
                valid_cols[0]:valid_cols[-1]+1
            ],
            cmap=cmap,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlim(75.8, 85.7)
        ax.set_ylim(23.2, 31.0)
        ax.set_xlabel("Longitude (°E)", color="white")
        ax.set_ylabel("Latitude (°N)", color="white")
        ax.tick_params(colors="white")

        for spine in ax.spines.values():
            spine.set_visible(False)

        label = LABELS.get(variable, variable)
                
        title_suffix = (
            str(start_date) if start_date == end_date
            else f"{start_date} to {end_date} (mean)"
        )
        st.caption(
            VAR_DESCRIPTIONS.get(
                variable,
                "No description available for this variable."
            )
        )
        ax.set_title(f"{label} - {title_suffix}", color="white")
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label(label, color="white")
        cbar.ax.tick_params(colors="white")
        plt.setp(cbar.ax.get_yticklabels(), color="white")
        cbar.outline.set_edgecolor("white")

        fig.tight_layout()

        add_up_cities(ax)

        st.pyplot(fig)

        buf = io.BytesIO()

        fig.savefig(
            buf,
            format="png",
            dpi=300,
            bbox_inches="tight"
        )

        file_suffix = (
            str(start_date) if start_date == end_date
            else f"{start_date}_to_{end_date}"
        )

        st.download_button(
            "Download Map",
            data=buf.getvalue(),
            file_name=f"baseline_{file_suffix}.png",
            mime="image/png"
        )

        plt.close(fig)

        if show_hist:
            st.subheader("Distribution")
            fig, ax = plt.subplots(
                figsize=(8,4),
                facecolor=STREAMLIT_BG
            )
            ax.set_facecolor(STREAMLIT_BG)
            values = data_range.values
            values = values[np.isfinite(values)]
            ax.hist(
                values,
                bins=50,
                color="steelblue"
            )
            ax.grid(linestyle="--",alpha=0.2,color="white")
            ax.tick_params(colors="white")
            ax.set_xlabel(LABELS.get(variable, variable), color="white")
            ax.set_ylabel("Frequency", color="white")
            for spine in ax.spines.values():
                spine.set_visible(False)
            st.pyplot(fig)
            buf = io.BytesIO()

            fig.savefig(
                buf,
                format="png",
                dpi=300,
                bbox_inches="tight"
            )

            file_suffix = (
                str(start_date) if start_date == end_date
                else f"{start_date}_to_{end_date}"
            )

            st.download_button(
                "Download Chart",
                data=buf.getvalue(),
                file_name=f"histogram_{file_suffix}.png",
                mime="image/png"
            )
            plt.close(fig)

    # --------------------------------------------------
    # Time Variables
    # --------------------------------------------------

    elif data.ndim == 1:

        fig, ax = plt.subplots(figsize=(10,4))

        ax.plot(
            ds.time.values,
            data.values,
            linewidth=1.5
        )

        # Highlight the selected date range instead of a single point
        ax.axvspan(
            np.datetime64(start_date),
            np.datetime64(end_date),
            color="red",
            alpha=0.15,
            zorder=5,
        )

        ax.scatter(
            ds.time.values[time_mask],
            data.values[time_mask],
            color="red",
            s=25,
            zorder=6,
        )

        ax.set_xlabel("Time")
        ax.set_ylabel(LABELS.get(variable, variable))
        ax.set_title(LABELS.get(variable, variable))

        plt.xticks(rotation=30)

        st.pyplot(fig)
        plt.close(fig)

with graph_tab:
    st.subheader("Temporal Trend")

    trend = data.sel(
        time=slice(str(start_date), str(end_date))
    )

    trend_series = trend.mean(
        dim=["lat", "lon"],
        skipna=True
    )

    fig, ax = plt.subplots(
        figsize=(9, 3.5),
        facecolor=STREAMLIT_BG
    )

    ax.set_facecolor(STREAMLIT_BG)

    ax.plot(
        trend_series.time.values,
        trend_series.values,
        linewidth=2,
        color="#00C2FF"
    )
    valid = np.isfinite(trend_series.values)

    time = trend_series.time.values

    # Convert to years since first observation
    x = (time - time[0]) / np.timedelta64(1, "D") / 365.25
    y = trend_series.values

    # Keep only finite values
    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]
    legend = ax.legend(frameon=False)

    for text in legend.get_texts():
        text.set_color("white")
    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        trendline = slope * x + intercept

        ax.plot(
            time[mask],
            trendline,
            "--",
            color="red",
            linewidth=2,
            label="Trend"
        )
        
    else:
        st.warning("Not enough valid data to compute trend.")

    ax.legend(frameon=False)
    ax.grid(
        alpha=0.25,
        linestyle="--"
    )

    ax.tick_params(colors="white")

    ax.set_xlabel(
        "Date",
        color="white"
    )

    ax.set_ylabel(
        LABELS.get(variable, variable),
        color="white"
    )

    ax.set_title(
        f"{LABELS.get(variable, variable)} Trend",
        color="white"
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.xticks(rotation=30)

    st.pyplot(fig)
    buf = io.BytesIO()

    fig.savefig(
        buf,
        format="png",
        dpi=300,
        bbox_inches="tight"
    )
    file_suffix = (
        str(start_date) if start_date == end_date
        else f"{start_date}_to_{end_date}"
    )
    st.download_button(
        "Download Temporal Trend",
        data=buf.getvalue(),
        file_name=f"temporal_trend_{file_suffix}.png",
        mime="image/png"
    )
    plt.close(fig)

# --------------------------------------------------
# Statistics Section (separate date selector + maps)
# --------------------------------------------------

STATS_VARS = ["temp_range", "rain_anomaly"]
STATS_LABELS = {v: LABELS.get(v, v) for v in STATS_VARS}

if show_stats:

    st.divider()
    st.subheader("Statistics")
    st.caption(
        "View Temperature Range or Rainfall Anomaly"
        "for an independently selected date range."
    )

    # --- Controls row: variable dropdown + date selectors ---
    stat_var_col, stat_col_from, stat_col_to, stat_spacer = st.columns(
        [3, 1.5, 1.5, 4],
        vertical_alignment="bottom"
    )

    with stat_var_col:
        stats_variable = st.selectbox(
            "Statistics Variable",
            STATS_VARS,
            format_func=lambda v: STATS_LABELS[v],
            key="stats_variable"
        )

    with stat_col_from:

        if stats_variable == "rain_anomaly":

            stats_date = st.date_input(
                "Date",
                value=available_dates[0],
                min_value=available_dates[0],
                max_value=available_dates[-1],
                key="stats_single_date"
            )

        else:

            stats_start = st.date_input(
                "Stats From",
                value=available_dates[0],
                min_value=available_dates[0],
                max_value=available_dates[-1],
                key="stats_date_from"
            )

    with stat_col_to:

        if stats_variable != "rain_anomaly":

            stats_end = st.date_input(
                "Stats To",
                value=available_dates[0],
                min_value=available_dates[0],
                max_value=available_dates[-1],
                key="stats_date_to"
            )
            
    if stats_variable != "rain_anomaly":
        c1,c2,c3,c4 = st.columns(4)
        if stats_start > stats_end:
            st.error("The start date must be earlier than or equal to the end date.")
            st.stop()
        if stats_start == stats_end:
            c1.metric("Selected Date: ", str(start_date))
        else:
            c1.metric("Date Range: ", f"{start_date} → {end_date}")

    if stats_variable == "rain_anomaly":
        c5,c6,c7,c8 = st.columns(4)
        c5.metric("Selected Date: ", str(start_date))
        stats_days_since_start = (
            pd.Timestamp(stats_date)
            - pd.Timestamp(ds.time.values[0])
        ).days

    else:

        if stats_start > stats_end:
            stats_start, stats_end = stats_end, stats_start

        stats_days_since_start = (
            pd.Timestamp(stats_start)
            - pd.Timestamp(ds.time.values[0])
        ).days

    # Check validity for the selected variable
    svar = stats_variable

    if (
        svar not in ds.data_vars
        or stats_days_since_start < VARIABLE_VALIDITY.get(svar, 0)
    ):
        st.warning(
            f"Not enough data before the selected statistics date for "
            f"**{STATS_LABELS.get(svar, svar)}**. Try a later start date."
        )
    else:
        st.markdown(f"#### {LABELS.get(svar, svar)}")
        st.caption(VAR_DESCRIPTIONS.get(svar, ""))

        sdata = ds[svar]
        if sdata.ndim == 3:
            if svar == "rain_anomaly":

                simage = sdata.sel(
                    time=str(stats_date)
                )

                sdata_range = simage

            else:

                sdata_range = sdata.sel(
                    time=slice(
                        str(stats_start),
                        str(stats_end)
                    )
                )

                simage = sdata_range.mean(
                    dim="time",
                    skipna=True
                )
            s_cmap_name = CMAPS.get(svar, "viridis")
            s_vmin, s_vmax = COLOR_LIMITS.get(svar, (None, None))
            s_cmap = plt.get_cmap(s_cmap_name).copy()
            s_cmap.set_bad(STREAMLIT_BG)

            s_mask = np.isfinite(simage.values)
            lat = ds.lat.values
            lon = ds.lon.values
            s_valid_rows = np.where(s_mask.any(axis=1))[0]
            s_valid_cols = np.where(s_mask.any(axis=0))[0]

            if len(s_valid_rows) == 0 or len(s_valid_cols) == 0:
                st.info(f"No valid data for {svar} in selected range.")
            else:
                s_lon2d, s_lat2d = np.meshgrid(
                    lon[s_valid_cols[0]:s_valid_cols[-1]+1],
                    lat[s_valid_rows[0]:s_valid_rows[-1]+1]
                )

                map_col, metrics_col = st.columns([3, 2])

                with map_col:
                    fig, ax = plt.subplots(
                        figsize=(5, 4),
                        facecolor=STREAMLIT_BG
                    )
                    ax.set_facecolor(STREAMLIT_BG)

                    im = ax.pcolormesh(
                        s_lon2d,
                        s_lat2d,
                        simage.values[
                            s_valid_rows[0]:s_valid_rows[-1]+1,
                            s_valid_cols[0]:s_valid_cols[-1]+1
                        ],
                        cmap=s_cmap,
                        shading="auto",
                        vmin=s_vmin,
                        vmax=s_vmax,
                    )

                    ax.set_xlim(75.8, 85.7)
                    ax.set_ylim(23.2, 31.0)
                    ax.set_xlabel("Longitude (°E)", color="white")
                    ax.set_ylabel("Latitude (°N)", color="white")
                    ax.tick_params(colors="white")
                    for spine in ax.spines.values():
                        spine.set_visible(False)

                    s_label = LABELS.get(svar, svar)
                    if svar == "rain_anomaly":
                        s_title_suffix = str(stats_date)
                    else:
                        s_title_suffix = (
                            str(stats_start)
                            if stats_start == stats_end
                            else f"{stats_start} to {stats_end} (Mean)"
                        )
                    ax.set_title(
                        f"{s_label} - {s_title_suffix}",
                        color="white"
                    )

                    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
                    cbar.set_label(s_label, color="white")
                    cbar.ax.tick_params(colors="white")
                    plt.setp(
                        cbar.ax.get_yticklabels(), color="white"
                    )
                    cbar.outline.set_edgecolor("white")

                    fig.tight_layout()
                    add_up_cities(ax)
                    st.pyplot(fig)
                    buf = io.BytesIO()

                    fig.savefig(
                        buf,
                        format="png",
                        dpi=300,
                        bbox_inches="tight"
                    )

                    file_suffix = (
                        str(start_date) if start_date == end_date
                        else f"{start_date}_to_{end_date}"
                    )

                    st.download_button(
                        "Download Stats Map",
                        data=buf.getvalue(),
                        file_name=f"stats_{file_suffix}.png",
                        mime="image/png"
                    )

                    plt.close(fig)

                with metrics_col:
                    if svar == "rain_anomaly":
                        vals = simage.values
                    else:
                        vals = sdata_range.values
                    m1, m2 = st.columns(2)
                    m1.metric("Mean", f"{np.nanmean(vals):.3f}")
                    m2.metric("Std Dev", f"{np.nanstd(vals):.3f}")
                    m3, m4 = st.columns(2)
                    m3.metric("Maximum", f"{np.nanmax(vals):.3f}")
                    m4.metric("Minimum", f"{np.nanmin(vals):.3f}")
                    with st.expander("What do these statistics mean?"):
                        st.markdown("""
                        **Mean:** Average value across all grid cells (and all days in
                        the selected range).

                        **Maximum / Minimum:** Largest / smallest value observed in the
                        selected map or range.

                        **Standard Deviation:** How much the values vary spatially (and
                        temporally for a multi-day range).
                        """)

        elif sdata.ndim == 1:
            sdata_range = sdata.sel(
                time=slice(str(stats_start), str(stats_end))
            )
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(
                sdata_range.time.values,
                sdata_range.values,
                linewidth=1.5
            )
            ax.set_xlabel("Time")
            ax.set_ylabel(LABELS.get(svar, svar))
            ax.set_title(LABELS.get(svar, svar))
            plt.xticks(rotation=30)
            st.pyplot(fig)
            plt.close(fig)

# --------------------------------------------------
# Dataset Information
# --------------------------------------------------

st.divider()
st.subheader("Dataset Information")
c1, c2, c3 = st.columns(3)
c1.metric("Variables", len(LABELS))
c2.metric("Time Steps", len(ds.time))
c3.metric("Grid Size", f"{len(ds.lat)} × {len(ds.lon)}")