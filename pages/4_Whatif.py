from pathlib import Path
import io

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import streamlit as st
import torch
import xarray as xr
import pandas as pd
import gdown

from climate_twin.models.convlstm import ConvLSTM
from climate_twin.preprocessing.normalize import ClimateNormalizer
from climate_twin.preprocessing.split import TimeSeriesSplit
from climate_twin.ml.dataset import ClimateTorchDataset
from climate_twin.forecasting.recursive_forecast import RecursiveForecaster

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

DATASET = Path("data/processed/climate_up_compressed.nc")

MODEL = "models/convlstm_up_best.pth"

WINDOW_SIZE = 30

FEATURES = [
    "rainfall",
    "tmax",
    "tmin",
]

TEMPERATURE_FEATURES = {"tmax", "tmin", "temp_mean"}
RAINFALL_FEATURES = {"rainfall"}

# --- Climatological baseline ---------------------------------------------

CLIMATOLOGY_YEARS = 10
CLIM_WINDOW_DAYS = 0

# --- Clausius-Clapeyron ---------------------------------------------------

MAGNUS_A = 17.67
MAGNUS_B = 243.5      # degrees C
MAGNUS_E0 = 6.112     # hPa at 0 C

PRECIP_SCALING_MODES = {
    "Mean rainfall (energy constrained, ~2 %/°C)": 0.35,
    "Extreme rainfall (~6 %/°C)": 0.90,
    "Full Clausius-Clapeyron (~7 %/°C)": 1.00,
}
DEFAULT_SCALING_MODE = "Full Clausius-Clapeyron (~7 %/°C)"

PATTERN_CLIP = (0.25, 2.50)

MC_SAMPLES = 30

FEEDBACK_ALPHA = 1.0

# --- Sensitivity finite-difference step ----------------------------------
#
# Reference mean warming used to compute dR/dT. Small enough to stay in
# the model's near-linear regime; large enough that the finite difference
# is not dominated by MC noise. Sensitivity is stored as mm/day per °C of
# spatial-mean warming and extrapolates linearly to the user's slider.
EPSILON_T = 1.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
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

def download_dataset():
    DATASET.parent.mkdir(parents=True, exist_ok=True)
    file_id = "11jqMmwyXjB0gxQpMvNdOx283-YrKJo9dq"
    gdown.download(id=file_id, output=str(DATASET), quiet=False)

def add_up_cities(ax):
    for city, (lat_c, lon_c) in UP_CITIES.items():
        ax.scatter(
            lon_c, lat_c,
            s=18, c="red",
            edgecolors="white", linewidth=0.6,
            zorder=20,
        )
        ax.text(
            lon_c + 0.08, lat_c + 0.08, city,
            fontsize=7, color="white", weight="bold",
            bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=1),
            zorder=21,
        )


# ----------------------------------------------------
# Streamlit
# ----------------------------------------------------

st.set_page_config(page_title="Climate Digital Twin", layout="wide")

with open(Path("assets/theme.css")) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Climate Digital Twin")


# ----------------------------------------------------
# Load Model
# ----------------------------------------------------

@st.cache_resource
def load_model():
    model = ConvLSTM(
        input_channels=len(FEATURES),
        hidden_channels=8,
        output_channels=1,
    )
    checkpoint = torch.load(MODEL, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model


# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------

@st.cache_resource
def load_dataset():
    if not DATASET.exists():
        with st.spinner("Downloading dataset..."):
            download_dataset()

    ds = xr.open_dataset(DATASET)
    splitter = TimeSeriesSplit()
    train, valid, test = splitter.split(ds)

    normalizer = ClimateNormalizer()
    normalizer.fit(train, FEATURES)
    test_norm = normalizer.transform(test)

    dataset = ClimateTorchDataset(
        test_norm,
        input_features=FEATURES,
        target="rainfall",
        window_size=WINDOW_SIZE,
    )
    return dataset, test_norm, normalizer, ds


model = load_model()
dataset, test, normalizer, raw_ds = load_dataset()

lat = test.lat.values
lon = test.lon.values


# ----------------------------------------------------
# Normalisation helpers
# ----------------------------------------------------

@st.cache_data(show_spinner=False)
def affine_params(feature):
    def probe(value):
        arr = np.full((1, 1), value, dtype=np.float32)
        out = normalizer.inverse_transform_array(arr, feature)
        return float(np.asarray(out).ravel()[0])

    offset = probe(0.0)
    scale = probe(1.0) - offset
    predicted = 2.0 * scale + offset
    actual = probe(2.0)
    tolerance = 1e-4 * max(1.0, abs(actual))
    is_linear = abs(actual - predicted) <= tolerance
    return scale, offset, is_linear


def to_physical(values, feature):
    return np.asarray(
        normalizer.inverse_transform_array(
            np.asarray(values, dtype=np.float32),
            feature,
        ),
        dtype=np.float64,
    )


def to_normalized(values, feature):
    for name in ("transform_array", "forward_array", "normalize_array"):
        fn = getattr(normalizer, name, None)
        if callable(fn):
            try:
                return np.asarray(
                    fn(np.asarray(values, dtype=np.float32), feature),
                    dtype=np.float64,
                )
            except TypeError:
                pass

    scale, offset, _ = affine_params(feature)
    if scale == 0:
        raise ValueError(
            f"Degenerate normalisation scale for feature '{feature}'."
        )
    return (np.asarray(values, dtype=np.float64) - offset) / scale


# ----------------------------------------------------
# Climatological baseline
# ----------------------------------------------------

def _target_month_days(timestamp, window_days):
    ts = pd.Timestamp(timestamp)
    if ts.month == 2 and ts.day == 29:
        anchor = pd.Timestamp(year=2001, month=2, day=28)
    else:
        anchor = pd.Timestamp(year=2001, month=ts.month, day=ts.day)

    pairs = set()
    for offset in range(-window_days, window_days + 1):
        shifted = anchor + pd.Timedelta(days=offset)
        pairs.add((shifted.month, shifted.day))
    if ts.month == 2 and ts.day == 29:
        pairs.add((2, 29))
    return pairs


@st.cache_data(show_spinner=False)
def climatology_for_dates(
    _ds, variable, date_keys,
    years=CLIMATOLOGY_YEARS,
    window_days=CLIM_WINDOW_DAYS,
):
    da = _ds[variable]
    times = pd.to_datetime(da["time"].values)
    last_year = int(times.year.max())
    first_year = last_year - int(years) + 1
    year_mask = (times.year >= first_year) & (times.year <= last_year)
    months = times.month.values
    days = times.day.values

    means, stds, counts = [], [], []
    for key in date_keys:
        wanted = _target_month_days(key, window_days)
        date_mask = np.array(
            [(m, d) in wanted for m, d in zip(months, days)],
            dtype=bool,
        )
        selector = np.where(year_mask & date_mask)[0]

        if selector.size == 0:
            shape = (da.sizes["lat"], da.sizes["lon"])
            means.append(np.full(shape, np.nan))
            stds.append(np.full(shape, np.nan))
            counts.append(0)
            continue

        subset = da.isel(time=selector)
        means.append(subset.mean("time", skipna=True).values)
        stds.append(subset.std("time", ddof=1, skipna=True).values)
        counts.append(int(selector.size))

    return (
        np.asarray(means, dtype=np.float64),
        np.asarray(stds, dtype=np.float64),
        np.asarray(counts, dtype=int),
    )


# ----------------------------------------------------
# Warming pattern (mean-preserving)
# ----------------------------------------------------

def area_weights(latitudes, shape):
    w = np.cos(np.deg2rad(np.asarray(latitudes, dtype=np.float64)))
    w = np.clip(w, 0.0, None)
    return np.broadcast_to(w[:, None], shape)


def weighted_mean(field, latitudes):
    field = np.asarray(field, dtype=np.float64)
    w = area_weights(latitudes, field.shape)
    finite = np.isfinite(field)
    denom = np.sum(w[finite])
    if denom == 0:
        return np.nan
    return float(np.sum(field[finite] * w[finite]) / denom)


@st.cache_data(show_spinner=False)
def warming_pattern(_ds, years=CLIMATOLOGY_YEARS):
    if "tmax" not in _ds or "tmin" not in _ds:
        return None

    tmean = (_ds["tmax"] + _ds["tmin"]) / 2.0
    annual = tmean.groupby("time.year").mean("time", skipna=True)

    year_values = annual["year"].values.astype(np.float64)
    values = np.asarray(annual.values, dtype=np.float64)

    if year_values.size < 5:
        return None

    centred_years = year_values - year_values.mean()
    denominator = float(np.sum(centred_years ** 2))
    if denominator == 0:
        return None

    anomalies = values - np.nanmean(values, axis=0, keepdims=True)
    slope = np.nansum(
        centred_years[:, None, None] * anomalies, axis=0
    ) / denominator

    all_nan = np.all(~np.isfinite(values), axis=0)
    slope = np.where(all_nan, np.nan, slope)

    mean_slope = weighted_mean(slope, _ds["lat"].values)
    if not np.isfinite(mean_slope) or mean_slope <= 0:
        return None

    pattern = slope / mean_slope
    pattern = np.clip(pattern, PATTERN_CLIP[0], PATTERN_CLIP[1])
    pattern = np.where(np.isfinite(pattern), pattern, 1.0)

    renorm = weighted_mean(pattern, _ds["lat"].values)
    if not np.isfinite(renorm) or renorm == 0:
        return None

    return pattern / renorm


def temperature_field(delta_mean, pattern, shape):
    if pattern is None:
        return np.full(shape, float(delta_mean), dtype=np.float64)
    return float(delta_mean) * np.asarray(pattern, dtype=np.float64)


# ----------------------------------------------------
# Clausius-Clapeyron
# ----------------------------------------------------

def saturation_vapour_pressure(temp_celsius):
    t = np.asarray(temp_celsius, dtype=np.float64)
    return MAGNUS_E0 * np.exp((MAGNUS_A * t) / (t + MAGNUS_B))


def cc_rainfall_multiplier(base_temp_c, delta_temp_field, exponent):
    base = np.asarray(base_temp_c, dtype=np.float64)
    delta = np.asarray(delta_temp_field, dtype=np.float64)
    ratio = (
        saturation_vapour_pressure(base + delta)
        / saturation_vapour_pressure(base)
    )
    ratio = np.where(np.isfinite(ratio), ratio, 1.0)
    return np.power(ratio, float(exponent))


# ----------------------------------------------------
# Scenario input construction
# ----------------------------------------------------

def channel_axis(sequence):
    n = len(FEATURES)
    if sequence.ndim != 4:
        raise ValueError(
            f"Expected a 4-D input sequence, received shape {sequence.shape}."
        )
    if sequence.shape[1] == n and sequence.shape[0] != n:
        return 1
    if sequence.shape[0] == n and sequence.shape[1] != n:
        return 0
    return 1


def apply_temperature_perturbation(sequence, delta_temp_field):
    modified = np.array(sequence, dtype=np.float32, copy=True)
    axis = channel_axis(modified)

    for index, feature in enumerate(FEATURES):
        if feature not in TEMPERATURE_FEATURES:
            continue

        selector = [slice(None)] * modified.ndim
        selector[axis] = index
        selector = tuple(selector)

        physical = to_physical(modified[selector], feature)
        physical = physical + delta_temp_field

        modified[selector] = np.asarray(
            to_normalized(physical, feature), dtype=np.float32,
        )
    return modified


# ----------------------------------------------------
# Clausius-Clapeyron validation
# ----------------------------------------------------

MIN_RAIN_FOR_RATIO = 0.10   # mm/day
CC_AGREEMENT_TOLERANCE = 0.25


def validate_against_cc(
    control,
    warmed,
    cc_multiplier,
    user_multiplier,
    delta_temp_mean,
    latitudes,
    min_rain=MIN_RAIN_FOR_RATIO,
):
    """
    Compare a rainfall response against Clausius-Clapeyron.

    In the sensitivity workflow, `warmed` is the pre-multiplier
    sensitivity-derived scenario (climatology + dR/dT · ΔT) and `control`
    is the climatology, so the diagnostic reports how the STORED
    sensitivity, extrapolated to the current slider ΔT, matches
    thermodynamic expectation.
    """

    control = np.asarray(control, dtype=np.float64)
    warmed = np.asarray(warmed, dtype=np.float64)

    reference_multiplier = np.asarray(cc_multiplier, dtype=np.float64) \
        * float(user_multiplier)

    expected = control * reference_multiplier
    residual = warmed - expected

    wet = np.isfinite(control) & (control >= min_rain)

    with np.errstate(divide="ignore", invalid="ignore"):
        model_multiplier = np.where(wet, warmed / control, np.nan)

    multiplier_error = model_multiplier - reference_multiplier

    weights = area_weights(latitudes, control.shape)

    def weighted_sum(field):
        finite = np.isfinite(field)
        return float(np.sum(field[finite] * weights[finite]))

    control_total = weighted_sum(control)
    warmed_total = weighted_sum(warmed)

    if control_total > 0:
        model_response = warmed_total / control_total
    else:
        model_response = np.nan

    cc_response = weighted_mean(
        np.where(np.isfinite(control) & (control > 0), reference_multiplier, np.nan),
        latitudes,
    )

    if abs(delta_temp_mean) > 1e-9:
        model_rate = (model_response - 1.0) / delta_temp_mean * 100.0
        cc_rate = (cc_response - 1.0) / delta_temp_mean * 100.0
    else:
        model_rate = np.nan
        cc_rate = np.nan

    valid = wet & np.isfinite(model_multiplier) & np.isfinite(reference_multiplier)
    if valid.sum() >= 10:
        a = model_multiplier[valid]
        b = reference_multiplier[valid]
        if a.std() > 0 and b.std() > 0:
            pattern_correlation = float(np.corrcoef(a, b)[0, 1])
        else:
            pattern_correlation = np.nan
        within_tolerance = float(
            np.mean(np.abs(a - b) <= CC_AGREEMENT_TOLERANCE * np.abs(b))
        ) * 100.0
    else:
        pattern_correlation = np.nan
        within_tolerance = np.nan

    return {
        "expected": expected,
        "residual": residual,
        "reference_multiplier": reference_multiplier,
        "model_multiplier": model_multiplier,
        "multiplier_error": multiplier_error,
        "wet_mask": wet,
        "model_response": model_response,
        "cc_response": cc_response,
        "model_rate": model_rate,
        "cc_rate": cc_rate,
        "pattern_correlation": pattern_correlation,
        "within_tolerance": within_tolerance,
    }


# ----------------------------------------------------
# Sensitivity computation
# ----------------------------------------------------

def has_dropout(module):
    return any(
        isinstance(sub, torch.nn.modules.dropout._DropoutNd)
        for sub in module.modules()
    )


def mc_sensitivity(model, sequence, warmed_sequence, days, samples):
    """
    Finite-difference sensitivity dR/dT with MC-Dropout uncertainty.

    Both forecasts start from the same initial condition. The warmed input
    differs only in its temperature channels, shifted by a spatial field
    whose area-weighted mean is EPSILON_T. Dividing (R_warmed - R_control)
    by EPSILON_T gives per-pixel rainfall change per °C of spatial-mean
    warming, with the observed warming pattern baked in.

    Returns physical-unit fields (mm/day, mm/day per °C):
        control_mean         (days, H, W)
        control_std          (days, H, W)
        sensitivity_mean     (days, H, W)
        sensitivity_std      (days, H, W)
        confidence           float
        collapsed            bool
    """

    control_forecaster = RecursiveForecaster(
        model, DEVICE,
        rainfall_index=FEATURES.index("rainfall"),
        feedback_alpha=FEEDBACK_ALPHA,
        mc_dropout=True,
    )
    warmed_forecaster = RecursiveForecaster(
        model, DEVICE,
        rainfall_index=FEATURES.index("rainfall"),
        feedback_alpha=FEEDBACK_ALPHA,
        mc_dropout=True,
    )

    control_preds = []
    warmed_preds = []

    for _ in range(samples):
        r0 = control_forecaster.forecast(sequence.copy(), days)
        r1 = warmed_forecaster.forecast(warmed_sequence.copy(), days)

        # Convert to physical units before differencing so dR/dT is in
        # mm/day per °C directly.
        r0_real = np.clip(to_physical(r0, "rainfall"), 0.0, None)
        r1_real = np.clip(to_physical(r1, "rainfall"), 0.0, None)

        control_preds.append(r0_real)
        warmed_preds.append(r1_real)

    model.eval()

    control_arr = np.asarray(control_preds, dtype=np.float64)   # (S, D, H, W)
    warmed_arr = np.asarray(warmed_preds, dtype=np.float64)

    sens_samples = (warmed_arr - control_arr) / EPSILON_T

    control_mean = control_arr.mean(axis=0)
    sensitivity_mean = sens_samples.mean(axis=0)

    if samples > 1:
        control_std = control_arr.std(axis=0, ddof=1)
        sensitivity_std = sens_samples.std(axis=0, ddof=1)
    else:
        control_std = np.zeros_like(control_mean)
        sensitivity_std = np.zeros_like(sensitivity_mean)

    confidence = float(np.exp(-np.nanmean(sensitivity_std)) * 100.0)

    collapsed = bool(
        samples > 1
        and np.nanmax(control_std) < 1e-9
        and np.nanmax(sensitivity_std) < 1e-9
    )

    return (
        control_mean,
        control_std,
        sensitivity_mean,
        sensitivity_std,
        confidence,
        collapsed,
    )


def scenario_from_sensitivity(
    climatology_baseline,
    sensitivity,
    delta_temp_mean,
    rain_multiplier,
):
    """
    scenario = (climatology + dR/dT · ΔT_mean) · multiplier

    Clamped to non-negative rainfall. Pure arithmetic; every slider move
    re-runs it for free.
    """
    scen = (
        np.asarray(climatology_baseline, dtype=np.float64)
        + np.asarray(sensitivity, dtype=np.float64) * float(delta_temp_mean)
    ) * float(rain_multiplier)
    return np.clip(scen, 0.0, None)


# ----------------------------------------------------
# Plotting
# ----------------------------------------------------

STREAMLIT_BG = (17 / 255, 24 / 255, 39 / 255, 0.35)


def mask_field(values, valid_mask):
    array = np.ma.masked_invalid(np.asarray(values, dtype=np.float64))
    return np.ma.masked_where(~valid_mask, array)


def plot_map(
    data, title, cmap,
    vmin=None, vmax=None, center=None,
    label="", export=False, filename="map.png",
):
    data = np.ma.masked_invalid(data)

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(STREAMLIT_BG)

    fig, ax = plt.subplots(figsize=(6, 5), facecolor=STREAMLIT_BG)
    ax.set_facecolor(STREAMLIT_BG)

    if center is None:
        im = ax.pcolormesh(
            lon, lat, data, cmap=cmap,
            shading="auto", vmin=vmin, vmax=vmax,
        )
    else:
        limit = max(abs(vmin or 0.0), abs(vmax or 0.0))
        if limit == 0:
            limit = 1.0
        im = ax.pcolormesh(
            lon, lat, data, cmap=cmap,
            shading="auto", vmin=-limit, vmax=limit,
        )

    add_up_cities(ax)
    ax.set_title(title, color="white", fontsize=14)
    ax.set_xlim(75.8, 85.7)
    ax.set_ylim(23.2, 31.0)
    ax.set_xlabel("Longitude", color="white", fontsize=11)
    ax.set_ylabel("Latitude", color="white", fontsize=11)
    ax.tick_params(colors="white", labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
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
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        st.download_button(
            "Download Map",
            data=buf.getvalue(),
            file_name=filename,
            mime="image/png",
            key=f"download_{filename}",
        )
    return fig


def render_map(**kwargs):
    fig = plot_map(**kwargs)
    st.pyplot(fig)
    plt.close(fig)


def sym_limit(masked_array, floor=1.0):
    if not np.any(~masked_array.mask):
        return floor
    v = float(np.ma.max(np.ma.abs(masked_array)))
    return v if v > 0 else floor


# ----------------------------------------------------
# Simulation Settings
# ----------------------------------------------------

st.subheader("Simulation Settings")

forecast_dates = pd.to_datetime(test.time.values[WINDOW_SIZE:]).date

record_years = pd.to_datetime(raw_ds["time"].values).year
climatology_last_year = int(record_years.max())
climatology_first_year = climatology_last_year - CLIMATOLOGY_YEARS + 1

c1, c2, c3 = st.columns([1, 1, 1])

with c1:
    import pandas as pd

    DEFAULT_DATE = pd.Timestamp("2024-07-25").date()

    selected_date = st.date_input(
        "Forecast Date",
        value=DEFAULT_DATE,
        min_value=forecast_dates[0],
        max_value=forecast_dates[-1],
    )

with c2:
    mc_samples = st.selectbox("MC Samples", [10, 20, 30, 50], index=2)

c4, c5 = st.columns(2)

with c4:
    temperature = st.slider(
        "Mean Temperature Change (°C)",
        min_value=-2.0, max_value=3.0, value=-2.0, step=0.5,
        format="%.1f °C",
        help=(
            "Area-weighted mean warming applied to the linear sensitivity. "
            "Redistributed across pixels using the observed warming pattern."
        ),
    )

with c5:
    rainfall = st.slider(
        "Rainfall Multiplier",
        min_value=0.5, max_value=2.0, value=1.0, step=0.1,
        format="%.1f×",
        help=(
            "Multiplicative factor applied AFTER the sensitivity has been "
            "added to the baseline: scenario = (climatology + dR/dT · ΔT) × "
            "multiplier. Also scales the CC reference used for validation."
        ),
    )

c6, c7 = st.columns(2)

with c6:
    scaling_mode = st.selectbox(
        "Precipitation Scaling",
        list(PRECIP_SCALING_MODES.keys()),
        index=list(PRECIP_SCALING_MODES.keys()).index(DEFAULT_SCALING_MODE),
    )

with c7:
    use_pattern = st.checkbox(
        "Use observed warming pattern",
        value=True,
        help=(
            "Off: every pixel warms uniformly. On: warming is redistributed "
            "by the observed trend pattern, preserving the requested mean."
        ),
    )

forecast_days = 1
cc_exponent = PRECIP_SCALING_MODES[scaling_mode]

st.write("")
run = st.button(
    "Compute Sensitivity Matrix",
    use_container_width=True,
    help=(
        "Runs the model at the control and at a +1 °C reference perturbation "
        "to compute dR/dT. After this, moving the temperature or rainfall "
        "slider updates the scenario instantly without any model calls."
    ),
)

st.divider()

with st.expander("Simulation Guide"):
    st.markdown(f"""
### Baseline
The baseline is the observed climatological mean rainfall for the same
calendar date across **{climatology_first_year}–{climatology_last_year}**
({CLIMATOLOGY_YEARS} samples per pixel). It is not a model forecast. Its
spread is the interannual standard deviation.

### Sensitivity (dR/dT)
Instead of running the model on every slider change, the dashboard stores
a **sensitivity matrix** — the per-grid rainfall response to a 1 °C
reference perturbation of the temperature channels:

    dR/dT (x) = [ R_warmed(x, ε) − R_control(x) ] / ε,   ε = {EPSILON_T:.1f} °C

Both `R_control` and `R_warmed` are ConvLSTM ensembles started from the
same initial condition; the rainfall channel is never modified, so the
response is the model's own. The stored `dR/dT` is in **mm/day per °C of
spatial-mean warming**, with the observed warming pattern baked into its
spatial structure.

### Scenario
Once `dR/dT` is stored, the scenario for any slider setting is:

    scenario = ( climatology + dR/dT · ΔT ) × multiplier

Clamped at zero. Pure arithmetic — moving the temperature or rainfall
slider is instant. Press "Compute Sensitivity Matrix" only when you
change the forecast date, horizon, MC sample count, or the
warming-pattern toggle.

### Forecast Horizon
For multi-day horizons, `dR/dT` is stored per day. Longer horizons take
longer to compute the first time, then all days are free at the sliders.

### Precipitation Scaling (Clausius-Clapeyron)
CC is used as an **independent yardstick**, not as an input. Warmer air
holds more moisture, giving a per-pixel expected multiplier:

    m(x) = [ e_s(T(x) + ΔT(x)) / e_s(T(x)) ] ** n

- **n = 1.0** — full thermodynamic scaling (~6–7 %/°C), right for extremes.
- **n ≈ 0.35** — ~2 %/°C, energy-constrained mean-rainfall response.

The comparison below reports how the stored sensitivity, evaluated at the
current slider ΔT, stacks up against CC.

### How to read the validation
The scenario's implied rainfall multiplier is `scenario / climatology`.
Comparing that against CC tells you whether the sensitivity's magnitude
matches thermodynamic expectation. Agreement means the ConvLSTM learned
a physically reasonable temperature–rainfall coupling. A near-zero
response means it barely uses temperature at all.

### Monte Carlo Samples
Uncertainty is estimated on `dR/dT` directly: each MC pass produces one
sensitivity field, and their spread is the uncertainty. The climatology
has no model uncertainty.
- 10 fast preview  · 20 balanced  · 30 recommended  · 50 most stable
""")


# ----------------------------------------------------
# Compute sensitivity on button press
# ----------------------------------------------------

def sensitivity_cache_key():
    return (
        str(selected_date),
        forecast_days,
        mc_samples,
        bool(use_pattern),
    )


if run:
    matches = np.where(forecast_dates == selected_date)[0]
    if matches.size == 0:
        st.error(
            f"{selected_date} is not present in the test record. "
            "Pick a date that exists in the dataset."
        )
        st.stop()

    sample = int(matches[0])
    X, _ = dataset[sample]
    sequence = X.numpy()

    grid_shape = (len(lat), len(lon))

    forecast_date_keys = tuple(
        (pd.Timestamp(selected_date) + pd.Timedelta(days=offset)).isoformat()
        for offset in range(forecast_days)
    )

    with st.spinner("Building climatological baseline..."):
        baseline_mean, baseline_std, baseline_counts = climatology_for_dates(
            raw_ds, "rainfall", forecast_date_keys,
        )
        tmax_clim, _, _ = climatology_for_dates(
            raw_ds, "tmax", forecast_date_keys
        )
        tmin_clim, _, _ = climatology_for_dates(
            raw_ds, "tmin", forecast_date_keys
        )

    baseline_temperature_all = (tmax_clim + tmin_clim) / 2.0
    base_temperature = baseline_temperature_all[0]

    if not np.isfinite(base_temperature).any():
        st.error(
            "No climatological temperature available for this date. "
            "Cannot evaluate the Clausius-Clapeyron response."
        )
        st.stop()

    base_temperature = np.where(
        np.isfinite(base_temperature),
        base_temperature,
        np.nanmean(base_temperature),
    )

    # Reference perturbation: ε = EPSILON_T °C mean, distributed by the pattern
    pattern = warming_pattern(raw_ds) if use_pattern else None
    ref_delta_temp_field = temperature_field(EPSILON_T, pattern, grid_shape)

    warmed_sequence = apply_temperature_perturbation(
        sequence, ref_delta_temp_field,
    )

    with st.spinner(
        f"Computing dR/dT (control + {EPSILON_T:.1f} °C reference run)..."
    ):
        (
            control_mean,
            control_std,
            sensitivity_mean,
            sensitivity_std,
            sens_confidence,
            collapsed,
        ) = mc_sensitivity(
            model, sequence, warmed_sequence, forecast_days, mc_samples,
        )

    if not has_dropout(model):
        st.warning(
            "No dropout layers found in ConvLSTM, so the Monte Carlo "
            "ensemble is deterministic. The sensitivity uncertainty will "
            "read zero."
        )
    elif collapsed:
        st.warning(
            "Every Monte Carlo sample was identical, so dropout was not "
            "active during inference. The sensitivity uncertainty map "
            "will be blank."
        )

    scale, _, is_linear = affine_params("rainfall")
    if not is_linear:
        st.warning(
            "The rainfall normalisation is not affine. The sensitivity is "
            "still computed in physical units, but verify ClimateNormalizer "
            "if a log transform has been added."
        )

    st.session_state.update({
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "baseline_counts": baseline_counts,
        "control_mean": control_mean,
        "control_std": control_std,
        "sensitivity_mean": sensitivity_mean,
        "sensitivity_std": sensitivity_std,
        "sens_confidence": sens_confidence,
        "base_temperature": base_temperature,
        "baseline_temperature_all": baseline_temperature_all,
        "warming_pattern_field": pattern if pattern is not None else None,
        "ref_delta_temp_field": ref_delta_temp_field,
        "result_forecast_days": forecast_days,
        "result_selected_date": pd.Timestamp(selected_date),
        "result_scaling_mode": scaling_mode,
        "result_cc_exponent": cc_exponent,
        "result_used_pattern": bool(pattern is not None),
        "result_mc_samples": mc_samples,
        "sensitivity_key": sensitivity_cache_key(),
    })


# ====================================================
# DISPLAY SAVED RESULTS
# ====================================================

if "sensitivity_mean" in st.session_state:

    baseline_all = st.session_state["baseline_mean"]
    baseline_std_all = st.session_state["baseline_std"]
    baseline_counts = st.session_state["baseline_counts"]

    control_all = st.session_state["control_mean"]
    sensitivity_all = st.session_state["sensitivity_mean"]
    sensitivity_std_all = st.session_state["sensitivity_std"]
    sens_confidence = st.session_state["sens_confidence"]

    base_temperature = st.session_state["base_temperature"]
    baseline_temperature_all = st.session_state["baseline_temperature_all"]
    stored_pattern = st.session_state["warming_pattern_field"]

    result_forecast_days = st.session_state["result_forecast_days"]
    result_date = st.session_state["result_selected_date"]
    result_scaling_mode = st.session_state["result_scaling_mode"]
    result_cc_exponent = st.session_state["result_cc_exponent"]
    result_used_pattern = st.session_state["result_used_pattern"]

    # Warn if the settings drifted from what dR/dT was computed for.
    # Temperature and rainfall sliders never invalidate the cache.
    if st.session_state.get("sensitivity_key") != sensitivity_cache_key():
        st.warning(
            "Forecast date, horizon, MC samples, or warming-pattern toggle "
            "changed since the sensitivity matrix was computed. Press "
            "**Compute Sensitivity Matrix** to refresh. Temperature and "
            "rainfall sliders do NOT require recomputation."
        )

    # -----------------------------------------------------
    # Derive scenario from cached sensitivity + current sliders
    # -----------------------------------------------------
    grid_shape = (len(lat), len(lon))
    delta_temp_field_now = temperature_field(
        temperature, stored_pattern, grid_shape,
    )
    applied_delta_mean = weighted_mean(delta_temp_field_now, lat)

    scenario_all = np.stack(
        [
            scenario_from_sensitivity(
                baseline_all[d], sensitivity_all[d], temperature, rainfall,
            )
            for d in range(result_forecast_days)
        ],
        axis=0,
    )

    # Scenario uncertainty propagates from dR/dT:
    #   scenario = (clim + sens · ΔT) · m
    #   → σ(scenario) ≈ |m · ΔT| · σ(sens)
    scenario_uncertainty_all = (
        np.abs(rainfall * temperature) * sensitivity_std_all
    )

    # CC field at the CURRENT slider ΔT.
    cc_field_now = cc_rainfall_multiplier(
        base_temperature, delta_temp_field_now, result_cc_exponent,
    )

    # For CC validation, compare pre-multiplier scenario against
    # climatology, so the multiplier applies to the CC reference only.
    pre_mult_scenario_all = np.stack(
        [
            scenario_from_sensitivity(
                baseline_all[d], sensitivity_all[d], temperature, 1.0,
            )
            for d in range(result_forecast_days)
        ],
        axis=0,
    )

    validation_all = [
        validate_against_cc(
            baseline_all[d],
            pre_mult_scenario_all[d],
            cc_field_now,
            rainfall,
            applied_delta_mean,
            lat,
        )
        for d in range(result_forecast_days)
    ]

    # -------------------------
    # Day selector
    # -------------------------
    day = 1
    idx = 0
    prediction_date = result_date

    valid_mask = ~np.isnan(test["rainfall"].isel(time=0).values)

    baseline = mask_field(baseline_all[idx], valid_mask)
    scenario = mask_field(scenario_all[idx], valid_mask)
    control = mask_field(control_all[idx], valid_mask)
    sensitivity = mask_field(sensitivity_all[idx], valid_mask)
    sensitivity_uncert = mask_field(sensitivity_std_all[idx], valid_mask)
    scenario_uncertainty = mask_field(scenario_uncertainty_all[idx], valid_mask)
    difference = scenario - baseline

    vresult = validation_all[idx]
    expected = mask_field(vresult["expected"], valid_mask)
    residual = mask_field(vresult["residual"], valid_mask)
    model_multiplier = mask_field(vresult["model_multiplier"], valid_mask)
    reference_multiplier = mask_field(vresult["reference_multiplier"], valid_mask)
    multiplier_error = mask_field(vresult["multiplier_error"], valid_mask)

    model_response_percent = (vresult["model_response"] - 1.0) * 100.0
    cc_response_percent = (vresult["cc_response"] - 1.0) * 100.0

    baseline_variability = mask_field(baseline_std_all[idx], valid_mask)
    delta_temp_masked = mask_field(delta_temp_field_now, valid_mask)

    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.ma.masked_invalid(
            np.abs(difference) / baseline_variability
        )

    # -------------------------
    # Forecast details
    # -------------------------
    st.subheader("Forecast Details")

    d1, d2, d3 = st.columns(3)
    d1.metric("Forecast Date", prediction_date.strftime("%d %b %Y"))
    d2.metric("Baseline Samples", f"{int(baseline_counts[idx])} yrs")
    d3.metric("MC Samples", st.session_state["result_mc_samples"])

    st.caption(
        f"Baseline: observed mean for this calendar date over "
        f"{climatology_first_year}–{climatology_last_year}. "
        f"Sensitivity computed at ε = {EPSILON_T:.1f} °C reference warming."
    )

    st.divider()

    # =====================================================
    # MAPS (above Simulation Summary per request)
    # =====================================================

    # Baseline / scenario temperature fields for the Temperature tab.
    # Scenario temperature = climatological daily-mean temperature plus
    # the applied ΔT field (which already carries the warming pattern).
    baseline_temp_day = baseline_temperature_all[idx]
    scenario_temp_day = baseline_temp_day + delta_temp_field_now

    baseline_temp_masked = mask_field(baseline_temp_day, valid_mask)
    scenario_temp_masked = mask_field(scenario_temp_day, valid_mask)

    tab_rain, tab_temp = st.tabs(["Rainfall", "Temperature"])

    # ------------------------------ Rainfall tab ------------------------------
    with tab_rain:

        # Row 1: Baseline / Scenario / Anomaly
        vmin = float(min(baseline.min(), scenario.min()))
        vmax = float(max(baseline.max(), scenario.max()))

        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("Baseline")
            render_map(
                data=baseline,
                title=f"{CLIMATOLOGY_YEARS}-Year Climatology",
                cmap="Blues",
                vmin=vmin, vmax=vmax,
                label="mm/day",
                export=True,
                filename=f"baseline_day_{day}.png",
            )

        with c2:
            st.subheader("Scenario")
            render_map(
                data=scenario,
                title=(
                    f"Climatology + dR/dT · {temperature:+.1f}°C, × {rainfall:.1f}"
                ),
                cmap="Blues",
                vmin=vmin, vmax=vmax,
                label="mm/day",
                export=True,
                filename=f"scenario_day_{day}.png",
            )

        with c3:
            st.subheader("Anomaly")
            anom_limit = sym_limit(difference)
            render_map(
                data=difference,
                title="Scenario − Climatology",
                cmap="RdBu_r",
                vmin=-anom_limit, vmax=anom_limit,
                label="mm/day",
                export=True,
                filename=f"anomaly_day_{day}.png",
            )

        st.divider()
        # Row 2: Sensitivity dR/dT + uncertainty
        st.subheader("Rainfall Sensitivity to Temperature (dR/dT)")

        s1, s2 = st.columns(2)

        with s1:
            st.markdown("**dR/dT (mm/day per °C of mean warming)**")
            s_limit = sym_limit(sensitivity)
            render_map(
                data=sensitivity,
                title="Rainfall Sensitivity",
                cmap="RdBu_r",
                vmin=-s_limit, vmax=s_limit,
                label="mm/day per °C",
                export=True,
                filename=f"sensitivity_day_{day}.png",
            )

            # Sensitivity NetCDF download
            sens_ds = xr.Dataset(
                {
                    "dR_dT": (
                        ("lat", "lon"),
                        np.array(sensitivity.filled(np.nan)),
                    ),
                    "dR_dT_std": (
                        ("lat", "lon"),
                        np.array(sensitivity_uncert.filled(np.nan)),
                    ),
                    "climatology_rainfall": (
                        ("lat", "lon"),
                        np.array(baseline.filled(np.nan)),
                    ),
                    "climatology_temperature": (
                        ("lat", "lon"),
                        np.array(baseline_temp_masked.filled(np.nan)),
                    ),
                },
                coords={"lat": lat, "lon": lon},
                attrs={
                    "epsilon_T_celsius": EPSILON_T,
                    "forecast_date": prediction_date.strftime("%Y-%m-%d"),
                    "forecast_day": day,
                    "pattern_applied": int(result_used_pattern),
                    "method": (
                        "finite difference (R_warmed - R_control) / epsilon, "
                        "with MC-Dropout uncertainty"
                    ),
                },
            )
            sbuf = io.BytesIO()
            sens_ds.to_netcdf(sbuf)
            sbuf.seek(0)
            st.download_button(
                label="Download Sensitivity (.nc)",
                data=sbuf.getvalue(),
                file_name=f"sensitivity_day_{day}.nc",
                mime="application/x-netcdf",
                key=f"download_sens_nc_{day}",
            )

        with s2:
            st.markdown("**dR/dT Uncertainty (MC-Dropout std)**")
            render_map(
                data=sensitivity_uncert,
                title="Sensitivity Uncertainty",
                cmap="inferno",
                vmin=0,
                vmax=float(sensitivity_uncert.max())
                if np.any(~sensitivity_uncert.mask) else 1.0,
                label="mm/day per °C",
                export=True,
                filename=f"sensitivity_uncert_day_{day}.png",
            )
        
        # Row: CC comparison maps
        st.subheader("Model Response vs Clausius-Clapeyron")

        model_percent = mask_field(
            (vresult["model_multiplier"] - 1.0) * 100.0, valid_mask
        )
        reference_percent = mask_field(
            (vresult["reference_multiplier"] - 1.0) * 100.0, valid_mask
        )
        error_percent = mask_field(
            vresult["multiplier_error"] * 100.0, valid_mask
        )

        shared_limit = max(
            sym_limit(model_percent, floor=0.0),
            sym_limit(reference_percent, floor=0.0),
            1.0,
        )

        g1, g2, g3 = st.columns(3)

        with g1:
            st.markdown("**Model**")
            render_map(
                data=model_percent,
                title="Model Rainfall Response",
                cmap="BrBG",
                vmin=-shared_limit, vmax=shared_limit,
                label="%",
                export=True,
                filename=f"model_response_day_{day}.png",
            )

        with g2:
            st.markdown("**Clausius-Clapeyron**")
            render_map(
                data=reference_percent,
                title="CC Expected Response",
                cmap="BrBG",
                vmin=-shared_limit, vmax=shared_limit,
                label="%",
                export=True,
                filename=f"cc_expected_day_{day}.png",
            )

        with g3:
            st.markdown("**Model − CC**")
            err_limit = sym_limit(error_percent)
            render_map(
                data=error_percent,
                title="Departure From CC",
                cmap="PuOr",
                vmin=-err_limit, vmax=err_limit,
                label="percentage points",
                export=True,
                filename=f"cc_departure_day_{day}.png",
            )

        st.caption(
            "'Model' here is the pre-multiplier sensitivity-derived scenario "
            "divided by climatology, evaluated at the current slider ΔT. Grey "
            "pixels were too dry in climatology for a ratio to be meaningful."
        )

        st.divider()

        # Row: Uncertainty maps
        st.subheader("Uncertainty")

        u1, u2 = st.columns(2)

        with u1:
            st.markdown("**Natural Variability (Baseline)**")
            render_map(
                data=baseline_variability,
                title="Interannual Std. Dev.",
                cmap="inferno",
                vmin=0,
                vmax=float(baseline_variability.max())
                if np.any(~baseline_variability.mask) else 1.0,
                label="mm/day",
                export=True,
                filename=f"baseline_variability_day_{day}.png",
            )

        with u2:
            st.markdown("**Scenario Uncertainty (from dR/dT)**")
            render_map(
                data=scenario_uncertainty,
                title="|multiplier × ΔT| · σ(dR/dT)",
                cmap="inferno",
                vmin=0,
                vmax=float(scenario_uncertainty.max())
                if np.any(~scenario_uncertainty.mask) else 1.0,
                label="mm/day",
                export=True,
                filename=f"scenario_uncertainty_day_{day}.png",
            )

        with st.expander("What do these two uncertainties mean?"):
            st.markdown("""
**Natural variability** is the standard deviation of observed rainfall on
this calendar date across the baseline years. It describes how much this
date varies year to year in the real world, independent of any model.

**Scenario uncertainty** is derived from the MC-Dropout spread on dR/dT.
Since scenario = (clim + sens · ΔT) · m, the uncertainty at any slider
setting is |m · ΔT| · σ(sens). It goes to zero when ΔT = 0.

A scenario anomaly smaller than the natural variability is not a
detectable climate signal, no matter how confident the model is.
""")

    # ---------------------------- Temperature tab ----------------------------
    with tab_temp:

        # Row 1: Baseline temperature / Scenario temperature
        # Shared colour scale so the warming shift is visible as a colour
        # shift rather than a rescaled palette.
        temp_finite = np.concatenate([
            np.asarray(baseline_temp_masked.compressed()),
            np.asarray(scenario_temp_masked.compressed()),
        ])
        if temp_finite.size > 0:
            temp_vmin = float(np.min(temp_finite))
            temp_vmax = float(np.max(temp_finite))
        else:
            temp_vmin, temp_vmax = 0.0, 1.0

        t1, t2 = st.columns(2)

        with t1:
            st.subheader("Baseline Temperature")
            render_map(
                data=baseline_temp_masked,
                title=f"{CLIMATOLOGY_YEARS}-Year Climatology (T_mean)",
                cmap="inferno",
                vmin=temp_vmin, vmax=temp_vmax,
                label="°C",
                export=True,
                filename=f"baseline_temp_day_{day}.png",
            )

        with t2:
            st.subheader("Scenario Temperature")
            render_map(
                data=scenario_temp_masked,
                title=(
                    f"Climatology + ΔT (mean {applied_delta_mean:+.2f} °C)"
                ),
                cmap="inferno",
                vmin=temp_vmin, vmax=temp_vmax,
                label="°C",
                export=True,
                filename=f"scenario_temp_day_{day}.png",
            )

        st.caption(
            "Baseline is the observed daily-mean temperature "
            "((tmax + tmin) / 2) averaged over "
            f"{climatology_first_year}–{climatology_last_year} for this "
            "calendar date. Scenario adds the applied ΔT field (which "
            "carries the warming pattern) to the baseline."
        )

        st.divider()

        # Applied ΔT field lives here (it is a temperature quantity).
        st.subheader("Applied Temperature Change")
        dt_limit = sym_limit(delta_temp_masked)
        render_map(
            data=delta_temp_masked,
            title=f"ΔT (spatial mean {applied_delta_mean:+.2f} °C)",
            cmap="RdBu_r",
            vmin=-dt_limit, vmax=dt_limit,
            label="°C",
            export=True,
            filename=f"delta_temperature_day_{day}.png",
        )

        with st.expander("How to read this tab"):
            st.markdown(f"""
**Baseline Temperature** is the observed daily-mean temperature climatology
for this calendar date. **Scenario Temperature** is that climatology with the
Applied ΔT field added — the slider drives it directly.

**dR/dT** is the model's estimated change in rainfall (mm/day) per **1 °C of
spatial-mean warming**, holding the initial condition fixed:

- **Blue** → the pixel gets wetter as it warms
- **Red** → the pixel gets drier as it warms
- **Near zero** → the model considers rainfall at that pixel insensitive
  to temperature over this horizon

The observed warming pattern is already baked into the field, so pixels
that warm more under the pattern also show a proportionally larger
rainfall response.

Uncertainty is the spread of dR/dT across
{st.session_state["result_mc_samples"]} Monte Carlo Dropout samples;
bright areas are where the sensitivity is unstable across samples and
should be treated with caution.
""")

    st.divider()

    # =====================================================
    # SIMULATION SUMMARY (below the maps)
    # =====================================================

    st.subheader("Simulation Summary")

    m1, m2, m3 = st.columns(3)
    m1.metric("Climatological Baseline", f"{baseline.mean():.3f} mm/day")
    m2.metric("Scenario Rainfall", f"{scenario.mean():.3f} mm/day")
    m3.metric("Mean Anomaly", f"{difference.mean():+.3f} mm/day")

    m4, m5, m6 = st.columns(3)
    m4.metric("Largest Increase", f"{difference.max():+.3f} mm/day")
    m5.metric("Largest Decrease", f"{difference.min():+.3f} mm/day")
    m6.metric(
        "Peak Scenario Uncertainty",
        f"{float(np.ma.max(scenario_uncertainty)):.3f}"
        if np.any(~scenario_uncertainty.mask) else "0.000",
    )

    st.divider()

    # Applied forcing
    st.subheader("Applied Forcing")

    f1, f2 = st.columns(2)
    f1.metric(
        "Mean ΔT (applied)",
        f"{applied_delta_mean:+.2f} °C",
        delta=f"requested {temperature:+.1f} °C",
        delta_color="off",
    )
    f2.metric(
        "ΔT Range Across Pixels",
        f"{delta_temp_masked.min():+.2f} to {delta_temp_masked.max():+.2f} °C",
    )
    st.caption(
        "Temperature is the only forcing that flows into dR/dT. The "
        "rainfall multiplier is applied afterwards; the CC diagnostics "
        "compare like-for-like against the pre-multiplier response."
    )

    st.divider()

    # CC validation
    st.subheader("Clausius-Clapeyron Validation")

    v1, v2, v3 = st.columns(3)
    v1.metric(
        "Model Rainfall Response",
        f"{model_response_percent:+.2f} %",
        delta=(
            f"{vresult['model_rate']:+.2f} %/°C"
            if np.isfinite(vresult["model_rate"]) else "no warming"
        ),
        delta_color="off",
    )
    v2.metric(
        "CC Expected Response",
        f"{cc_response_percent:+.2f} %",
        delta=(
            f"{vresult['cc_rate']:+.2f} %/°C"
            if np.isfinite(vresult["cc_rate"]) else "no warming"
        ),
        delta_color="off",
    )
    v3.metric(
        "Discrepancy",
        f"{model_response_percent - cc_response_percent:+.2f} pp",
    )

    v4, v5, v6 = st.columns(3)
    v4.metric(
        "Pattern Correlation",
        f"{vresult['pattern_correlation']:.3f}"
        if np.isfinite(vresult["pattern_correlation"]) else "n/a",
    )
    v5.metric(
        "Pixels Within ±25% of CC",
        f"{vresult['within_tolerance']:.1f} %"
        if np.isfinite(vresult["within_tolerance"]) else "n/a",
    )
    v6.metric("Mean Residual", f"{residual.mean():+.4f} mm/day")

    st.caption(
        f"Reference = climatology × CC × {rainfall:.1f} ({result_scaling_mode}). "
        f"Model response = (climatology + dR/dT · {temperature:+.1f}°C) / "
        f"climatology. Ratio diagnostics exclude climatology-dry pixels "
        f"(< {MIN_RAIN_FOR_RATIO} mm/day)."
    )

    if abs(temperature) < 1e-6:
        st.info(
            "ΔT = 0, so the scenario equals climatology × multiplier and the "
            "CC comparison is trivial. Move the temperature slider to "
            "exercise the sensitivity."
        )
    elif np.isfinite(vresult["model_rate"]) and np.isfinite(vresult["cc_rate"]):
        gap = vresult["model_rate"] - vresult["cc_rate"]
        if abs(gap) <= 1.0:
            st.success(
                f"The sensitivity implies {vresult['model_rate']:+.2f} %/°C, "
                f"consistent with the CC expectation of "
                f"{vresult['cc_rate']:+.2f} %/°C."
            )
        elif abs(vresult["model_rate"]) < 0.5:
            st.error(
                f"The sensitivity implies only "
                f"{vresult['model_rate']:+.2f} %/°C. The ConvLSTM barely "
                "responds to temperature — either it has not learned a "
                "temperature-rainfall relationship, or the reference "
                "perturbation is too small to move it."
            )
        else:
            st.warning(
                f"Sensitivity implies {vresult['model_rate']:+.2f} %/°C "
                f"against a CC expectation of {vresult['cc_rate']:+.2f} %/°C. "
                "Regional rainfall is governed by circulation as well as "
                "moisture, so a gap is not automatically a model failure, "
                "but it is worth understanding before trusting the scenario."
            )

    st.divider()

    # Confidence and significance
    st.subheader("Confidence and Significance")

    p1, p2, p3 = st.columns(3)
    p1.metric("Sensitivity Confidence", f"{sens_confidence:.1f}%")
    p2.metric(
        "Climatological Spread",
        f"±{baseline_variability.mean():.3f} mm/day",
    )
    p3.metric(
        "Signal-to-Noise",
        f"{float(np.ma.mean(snr)):.2f}"
        if np.any(~snr.mask) else "n/a",
    )

    st.caption(
        "Sensitivity confidence is derived from MC-Dropout spread on dR/dT. "
        "Signal-to-noise is the mean absolute anomaly divided by "
        "interannual std. Below 1 means the scenario change is within "
        "year-to-year variability."
    )

    # -------------------------
    # Statistics export
    # -------------------------
    stats = {
        "Metric": [
            "Prediction Date",
            "Forecast Day",
            "Baseline Period",
            "Baseline Samples (years)",
            "Requested Mean Temperature Change (°C)",
            "Applied Mean Temperature Change (°C)",
            "Warming Pattern",
            "Precipitation Scaling Mode",
            "Rainfall Multiplier",
            "Sensitivity Reference ε (°C)",
            "Sensitivity Confidence (%)",
            "Model Rainfall Response (%)",
            "Model Implied Scaling (%/°C)",
            "CC Expected Response (%)",
            "CC Expected Scaling (%/°C)",
            "Model minus CC (pp)",
            "Pattern Correlation With CC",
            "Pixels Within 25% of CC (%)",
            "Climatological Mean Rainfall (mm/day)",
            "Scenario Mean Rainfall (mm/day)",
            "Mean Anomaly (mm/day)",
            "Max Anomaly (mm/day)",
            "Min Anomaly (mm/day)",
            "Mean |dR/dT| (mm/day per °C)",
            "Max |dR/dT| (mm/day per °C)",
            "Climatological Spread (mm/day)",
            "Signal-to-Noise",
        ],
        "Value": [
            prediction_date.strftime("%Y-%m-%d"),
            day,
            f"{climatology_first_year}-{climatology_last_year}",
            int(baseline_counts[idx]),
            round(float(temperature), 2),
            round(float(applied_delta_mean), 3),
            "observed trend" if result_used_pattern else "uniform",
            result_scaling_mode,
            round(float(rainfall), 2),
            round(float(EPSILON_T), 2),
            round(float(sens_confidence), 2),
            round(float(model_response_percent), 3),
            round(float(vresult["model_rate"]), 3)
            if np.isfinite(vresult["model_rate"]) else "n/a",
            round(float(cc_response_percent), 3),
            round(float(vresult["cc_rate"]), 3)
            if np.isfinite(vresult["cc_rate"]) else "n/a",
            round(float(model_response_percent - cc_response_percent), 3),
            round(float(vresult["pattern_correlation"]), 4)
            if np.isfinite(vresult["pattern_correlation"]) else "n/a",
            round(float(vresult["within_tolerance"]), 2)
            if np.isfinite(vresult["within_tolerance"]) else "n/a",
            round(float(baseline.mean()), 3),
            round(float(scenario.mean()), 3),
            round(float(difference.mean()), 3),
            round(float(difference.max()), 3),
            round(float(difference.min()), 3),
            round(float(np.ma.mean(np.ma.abs(sensitivity))), 4),
            round(float(np.ma.max(np.ma.abs(sensitivity))), 4),
            round(float(baseline_variability.mean()), 3),
            round(float(np.ma.mean(snr)), 3)
            if np.any(~snr.mask) else "n/a",
        ],
    }

    stats_df = pd.DataFrame(stats)
    st.download_button(
        label="Download Statistics (CSV)",
        data=stats_df.to_csv(index=False).encode("utf-8"),
        file_name=f"agni_stats_day_{day}.csv",
        mime="text/csv",
    )

    # -------------------------
    # Interpretation
    # -------------------------
    st.subheader("Scenario Interpretation")

    change = float(difference.mean())
    noise = float(baseline_variability.mean())

    if change > 0:
        direction = (
            f"increased the average predicted rainfall by "
            f"**{change:.3f} mm/day** relative to the "
            f"{CLIMATOLOGY_YEARS}-year climatology"
        )
    elif change < 0:
        direction = (
            f"decreased the average predicted rainfall by "
            f"**{abs(change):.3f} mm/day** relative to the "
            f"{CLIMATOLOGY_YEARS}-year climatology"
        )
    else:
        direction = "produced no detectable change relative to the climatology"

    if noise > 0 and abs(change) < noise:
        significance = (
            "This anomaly is **smaller than the interannual variability** of "
            "this date, so it falls within the range the historical record "
            "already covers."
        )
    else:
        significance = (
            "This anomaly **exceeds the interannual variability** of this "
            "date, placing it outside the typical historical range."
        )

    st.info(f"""
### Summary

**Baseline:** {CLIMATOLOGY_YEARS}-year climatology
({climatology_first_year}–{climatology_last_year}), {int(baseline_counts[idx])} samples

**Mean Temperature Change:** {temperature:+.1f} °C
({'observed pattern' if result_used_pattern else 'uniform'})

**Rainfall Multiplier:** ×{rainfall:.1f}

**Precipitation Scaling (CC yardstick):** {result_scaling_mode}

**Scenario formula:** `(Climatology + dR/dT · ΔT) × Multiplier`

The selected scenario {direction}.

{significance}

Sensitivity Confidence: **{sens_confidence:.1f}%**
""")