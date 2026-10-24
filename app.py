"""Streamlit app for estimating wind farm outage energy from uploaded turbine data."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import methods_tabs
from baseline import park_means
from pipeline import fit_predict
from pipeline.evaluate import energy_interval
from preprocessing import (
    DATA_DIR,
    REFERENCES,
    SENSORS,
    STEP_HOURS,
    TARGETS,
    VAL,
    clean_turbine,
    load_power_curve,
    to_wide,
    wind_to_power,
)
from turbine_availability import (
    EXPECTED_TURBINES,
    get_available_turbines,
    load_turbine_data,
)


PRIMARY_GREEN = "#008b47"
PREDICTION_GREEN = "#77b90f"
OUTAGE_YELLOW = "#fad804"
DEFAULT_OUTAGE_PERIOD = (date(2026, 9, 1), date(2026, 9, 3))
# public part of the test set: test_set=True and park T known (VAL end is exclusive)
PUBLIC_TEST_PERIOD = (VAL[0].date(), (VAL[1] - pd.Timedelta(days=1)).date())

class UploadedTurbineFile(Protocol):
    name: str

    def getvalue(self) -> bytes: ...


@dataclass(frozen=True)
class AnalysisResult:
    predictions: dict[str, pd.DataFrame]
    energy_loss: dict[str, pd.DataFrame]
    turbine_types: list[str]
    available_turbines: list[str]
    context_energy: dict[str, pd.DataFrame] = field(default_factory=dict)


def apply_custom_theme() -> None:
    """Set the shared BayWa r.e. color palette and layout for Plotly charts."""
    pio.templates["baywa"] = go.layout.Template(
        layout=go.Layout(
            colorway=[PRIMARY_GREEN, PREDICTION_GREEN, OUTAGE_YELLOW],
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font={"family": "Arial, sans-serif", "color": "#20332a"},
            margin={"l": 48, "r": 24, "t": 48, "b": 48},
            xaxis={"gridcolor": "#e8eee9", "zerolinecolor": "#e8eee9"},
            yaxis={"gridcolor": "#e8eee9", "zerolinecolor": "#e8eee9"},
            hoverlabel={"bgcolor": "#ffffff", "font": {"color": "#20332a"}},
        )
    )
    pio.templates.default = "baywa"


def load_data(
    files: UploadedTurbineFile | Sequence[UploadedTurbineFile] | None,
) -> pd.DataFrame:
    """Parse raw per-turbine uploads and validate their required columns and identity."""
    if files is None:
        return pd.DataFrame()
    uploaded_files = list(files) if isinstance(files, (list, tuple)) else [files]
    if not uploaded_files:
        return pd.DataFrame()

    frames = []
    required_columns = {
        "turbine_name",
        "timestamp",
        "wind_speed",
        "wind_direction",
        "ambient_temperature",
        "air_pressure",
        "wind_speed_stddev",
        "test_set",
    }
    seen_turbines: set[str] = set()
    for uploaded_file in uploaded_files:
        raw = load_turbine_data(uploaded_file.name, uploaded_file.getvalue())
        missing_columns = required_columns - set(raw.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"{uploaded_file.name} is missing required raw-data columns: {missing}."
            )
        names = raw["turbine_name"].dropna().astype(str).str.strip().unique()
        if len(names) != 1:
            raise ValueError(
                f"{uploaded_file.name} must contain data for exactly one turbine."
            )
        turbine = names[0]
        if turbine not in (*REFERENCES, *TARGETS):
            raise ValueError(
                f"{uploaded_file.name} contains unknown turbine {turbine!r}; "
                f"expected one of {', '.join((*REFERENCES, *TARGETS))}."
            )
        if turbine in seen_turbines:
            raise ValueError(f"More than one uploaded file contains turbine {turbine}.")
        seen_turbines.add(turbine)
        raw["turbine_name"] = turbine
        frames.append(raw)

    return pd.concat(frames, ignore_index=True)


@dataclass(frozen=True)
class LocalTurbineFile:
    """A bundled data file with the same interface as a Streamlit upload."""

    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    def getvalue(self) -> bytes:
        return self.path.read_bytes()


@st.cache_data(max_entries=1, show_spinner=False)
def load_bundled_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw reference and target data shipped in data/dataset (the challenge files)."""
    files = {t: LocalTurbineFile(DATA_DIR / f"{t}_data.xlsx") for t in (*REFERENCES, *TARGETS)}
    return (
        load_data([files[t] for t in REFERENCES]),
        load_data([files[t] for t in TARGETS]),
    )


def run_bundled_analysis(period: tuple[date, date]) -> None:
    """Button callback: run the analysis on the challenge files for a fixed period."""
    st.session_state["outage_period"] = period
    try:
        raw_references, raw_target = load_bundled_data()
        st.session_state["available_turbines"] = get_available_turbines(raw_references)
        st.session_state["analysis_result"] = run_analysis(raw_references, raw_target, period)
        st.session_state["analysis_period"] = period
        st.session_state["main_tabs"] = "Dashboard & Results"
        st.session_state.pop("analysis_error", None)
    except (OSError, ValueError) as exc:
        st.session_state.pop("analysis_result", None)
        st.session_state["analysis_error"] = str(exc)


def preprocess_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Apply the shared preprocessing rules to each uploaded turbine."""
    if raw_data.empty:
        raise ValueError("No turbine data was uploaded.")
    return pd.concat(
        [
            clean_turbine(raw_data[raw_data["turbine_name"] == turbine].copy())
            for turbine in raw_data["turbine_name"].dropna().unique()
        ],
        ignore_index=True,
    )


def estimate_outage_period(
    wind_speed: pd.Series,
) -> tuple[date, date]:
    """Find the dates covered by the longest consecutive missing-wind-speed interval."""
    start, end, _ = _estimate_outage_interval(wind_speed)
    return start, end


def _estimate_outage_interval(
    wind_speed: pd.Series,
) -> tuple[date, date, int]:
    """Return the dates and number of steps in the longest missing-data interval."""
    missing = wind_speed.isna().to_numpy()
    best_start = best_end = None
    run_start = None
    for position, is_missing in enumerate(missing):
        if is_missing and run_start is None:
            run_start = position
        elif not is_missing and run_start is not None:
            if best_start is None or position - run_start > best_end - best_start:
                best_start, best_end = run_start, position
            run_start = None

    if run_start is not None and (
        best_start is None or len(missing) - run_start > best_end - best_start
    ):
        best_start, best_end = run_start, len(missing)
    if best_start is None:
        raise ValueError("No missing wind-speed interval was found in the target data.")

    timestamps = wind_speed.index
    return (
        timestamps[best_start].date(),
        timestamps[best_end - 1].date(),
        best_end - best_start,
    )


def estimate_outage_period_for_targets(
    wind_table: pd.DataFrame,
    target_names: Sequence[str],
) -> tuple[date, date]:
    """Choose the longest missing interval found on any uploaded target turbine."""
    candidates = []
    for turbine in target_names:
        try:
            candidates.append(_estimate_outage_interval(wind_table[turbine]))
        except ValueError:
            continue
    if not candidates:
        raise ValueError("No missing wind-speed interval was found in the target data.")
    start, end, _ = max(candidates, key=lambda interval: interval[2])
    return start, end


def get_power_curve(turbine_type: str) -> pd.DataFrame:
    """Load the supplied power curve for the selected turbine."""
    if turbine_type not in TARGETS:
        raise ValueError(f"Unknown target turbine: {turbine_type}.")
    return load_power_curve(turbine_type)


def calculate_predictions(
    sensor_tables: Mapping[str, pd.DataFrame],
    outage_period: tuple[date, date],
    model_params: Mapping[str, object],
) -> pd.DataFrame:
    """Fit the best pipeline model (pipeline/experiments.py BEST) outside the selected dates
    and predict the selected window, with a 95 % wind-speed band and expected power."""
    start_date, end_date = outage_period
    if end_date < start_date:
        raise ValueError("The outage end date must be on or after the start date.")
    turbine_type = model_params.get("turbine")
    if not isinstance(turbine_type, str) or turbine_type not in TARGETS:
        raise ValueError("A valid target turbine (T1, T2, or T3) is required.")

    end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    wind_table = sensor_tables["wind_speed"]
    window = wind_table.loc[
        (wind_table.index >= pd.Timestamp(start_date))
        & (wind_table.index < end_exclusive)
    ]
    if window.empty:
        raise ValueError("The selected outage period is outside the uploaded data range.")

    predicted = fit_predict(
        dict(sensor_tables),
        "best",
        (pd.Timestamp(start_date), end_exclusive),
        targets=[turbine_type],
        curve=get_power_curve(turbine_type),
    )[turbine_type].reindex(window.index)
    return pd.DataFrame(
        {
            "timestamp": window.index,
            "reference_wind_speed": park_means(window)["H"].to_numpy(),
            "predicted_wind_speed": predicted["pred"].to_numpy(),
            "predicted_wind_lo": predicted["lo"].to_numpy(),
            "predicted_wind_hi": predicted["hi"].to_numpy(),
            # expected power over the plausible wind range, not P(predicted wind)
            "predicted_power_kw": predicted["kw"].to_numpy(),
            "actual_wind_speed": window[turbine_type].to_numpy(),
        }
    )


def calculate_energy_loss(
    predictions: pd.DataFrame,
    power_curve: pd.DataFrame,
) -> pd.DataFrame:
    """Convert predicted and actual wind speeds to power and energy per step."""
    if "predicted_power_kw" in predictions:
        predicted_power_kw = predictions["predicted_power_kw"].to_numpy()
    else:
        predicted_power_kw = wind_to_power(
            predictions["predicted_wind_speed"], power_curve
        )
    actual_power_kw = wind_to_power(predictions["actual_wind_speed"], power_curve)
    return pd.DataFrame(
        {
            "timestamp": predictions["timestamp"],
            "predicted_power_kw": predicted_power_kw,
            "actual_power_kw": actual_power_kw,
            "loss_kwh": predicted_power_kw * STEP_HOURS,
            "actual_yield_kwh": actual_power_kw * STEP_HOURS,
        }
    )


def calculate_validation_metrics(
    predictions: pd.DataFrame,
    energy_loss: pd.DataFrame,
) -> dict[str, float] | None:
    """Calculate wind, power, and net energy-yield errors on paired observations."""
    paired = predictions.dropna(
        subset=["actual_wind_speed", "predicted_wind_speed"]
    )
    paired_energy = energy_loss.dropna(
        subset=["actual_power_kw", "predicted_power_kw"]
    )
    if paired.empty or paired_energy.empty:
        return None

    wind_errors = paired["predicted_wind_speed"] - paired["actual_wind_speed"]
    power_errors = (
        paired_energy["predicted_power_kw"] - paired_energy["actual_power_kw"]
    )
    energy_deviation = (
        paired_energy["loss_kwh"] - paired_energy["actual_yield_kwh"]
    ).sum()
    return {
        "wind_rmse": math.sqrt((wind_errors**2).mean()),
        "wind_mae": wind_errors.abs().mean(),
        "power_rmse": math.sqrt((power_errors**2).mean()),
        "power_mae": power_errors.abs().mean(),
        "energy_deviation_kwh": energy_deviation,
    }


def run_analysis(
    raw_references: pd.DataFrame,
    raw_target: pd.DataFrame,
    outage_period: tuple[date, date],
) -> AnalysisResult:
    """Preprocess uploaded turbines and run the OLS outage prediction."""
    reference_names = set(get_available_turbines(raw_references))
    missing_references = set(REFERENCES) - reference_names
    if missing_references:
        raise ValueError(
            "Upload all reference turbines required by the model: "
            f"{', '.join(sorted(missing_references))}."
        )

    target_names = get_available_turbines(raw_target)
    if not target_names or not set(target_names).issubset(TARGETS):
        raise ValueError("Upload at least one target turbine file containing T1, T2, or T3.")

    cleaned = preprocess_data(pd.concat([raw_references, raw_target], ignore_index=True))
    sensor_tables = {sensor: to_wide(cleaned, sensor) for sensor in SENSORS}
    wind_table = sensor_tables["wind_speed"]
    predictions: dict[str, pd.DataFrame] = {}
    energy_loss: dict[str, pd.DataFrame] = {}
    context_energy: dict[str, pd.DataFrame] = {}
    outage_start = pd.Timestamp(outage_period[0])
    outage_end = pd.Timestamp(outage_period[1]) + pd.Timedelta(days=1)
    context_start = outage_start - pd.Timedelta(days=2)
    context_end = outage_end + pd.Timedelta(days=2)
    context_window = wind_table.loc[
        (wind_table.index >= context_start)
        & (wind_table.index < context_end)
        & ((wind_table.index < outage_start) | (wind_table.index >= outage_end))
    ]
    for turbine in TARGETS:
        if turbine not in target_names:
            continue
        power_curve = get_power_curve(turbine)
        predictions[turbine] = calculate_predictions(
            sensor_tables,
            outage_period,
            {"turbine": turbine},
        )
        energy_loss[turbine] = calculate_energy_loss(
            predictions[turbine],
            power_curve,
        )
        context_energy[turbine] = pd.DataFrame(
            {
                "timestamp": context_window.index,
                "actual_yield_kwh": (
                    wind_to_power(context_window[turbine], power_curve)
                    * STEP_HOURS
                ),
            }
        )
    return AnalysisResult(
        predictions=predictions,
        energy_loss=energy_loss,
        turbine_types=[turbine for turbine in TARGETS if turbine in target_names],
        available_turbines=sorted(reference_names),
        context_energy=context_energy,
    )


def apply_page_styles() -> None:
    """Apply lightweight layout styling while keeping Streamlit's standard wide layout."""
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.8rem;
                padding-bottom: 3rem;
            }
            [data-testid="stMetric"] {
                background: #f6faf7;
                border: 1px solid #e4eee6;
                border-radius: 12px;
                padding: 1rem 1.1rem;
            }
            [data-testid="stMetricLabel"] {
                color: #52665a;
            }
            div[data-baseweb="tab-list"] {
                gap: 0.5rem;
            }
            button[data-baseweb="tab"] {
                border-radius: 8px 8px 0 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(
    selected_period: tuple[date, date],
    result: AnalysisResult | None,
    available_turbines: Sequence[str],
) -> None:
    """Render the outage estimate and any metrics supported by observed data."""
    st.subheader("Outage overview")
    if result is None:
        render_turbine_status(available_turbines, EXPECTED_TURBINES)
        st.info("Upload the wind data and submit it from the sidebar to start the calculation.")
        return
    turbine_label = ", ".join(result.turbine_types)
    # st.caption(
    #     f"Selected period: {selected_period[0]:%d %b %Y} – "
    #     f"{selected_period[1]:%d %b %Y} · Turbine: {turbine_label}"
    # )
    delta = selected_period[1] - selected_period[0]
    outage_hours = int(delta.total_seconds() / 3600)



    metric_rows = []
    total_predictions = pd.concat(result.predictions.values(), ignore_index=True)
    total_energy = pd.concat(result.energy_loss.values(), ignore_index=True)
    total_loss_kwh = total_energy["loss_kwh"].sum(min_count=1)
    total_loss_label = (
        f"{total_loss_kwh:,.1f} kWh"
        if pd.notna(total_loss_kwh)
        else "N/A"
    )
    total_range = energy_range_kwh(total_loss_kwh, result.turbine_types)
    total_range_label = (
        f"{total_range[0]:,.0f} – {total_range[1]:,.0f} kWh" if total_range else "N/A"
    )
    st.markdown(
    f"""
    <div style="
        background-color: #b42318;
        color: #ffffff;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin: 0.75rem 0 1.25rem;
    ">
        <div style="font-size: 1rem; font-weight: 600;">
            Total estimated energy yield loss
        </div>
        <div style="font-size: 2.5rem; font-weight: 700; line-height: 1.2;">
            {total_loss_label}
        </div>
        <div style="font-size: 1.05rem; font-weight: 600; margin-top: 0.25rem;">
            95 % range: {total_range_label}
        </div>
        <div style="font-size: 0.9rem; margin-top: 0.75rem; opacity: 0.9; line-height: 1.5;">
            Period: <strong>{selected_period[0]:%d %b %Y} – {selected_period[1]:%d %b %Y}</strong><br>
            Duration: <strong>{outage_hours} hours</strong>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
    total_metrics = calculate_validation_metrics(total_predictions, total_energy)
    render_turbine_status(available_turbines, EXPECTED_TURBINES)

    for turbine in result.turbine_types:
        turbine_predictions = result.predictions[turbine]
        turbine_energy = result.energy_loss[turbine]
        validation_metrics = calculate_validation_metrics(
            turbine_predictions, turbine_energy
        )
        metric_rows.append(
            _metric_summary_row(turbine, turbine_energy, validation_metrics, [turbine])
        )

    metric_rows.insert(
        0,
        _metric_summary_row("Total", total_energy, total_metrics, result.turbine_types),
    )
    st.markdown("### Metrics summary")

    context_frames = list(result.context_energy.values())
    context_energy = (
        pd.concat(context_frames, ignore_index=True)
        if context_frames
        else pd.DataFrame(columns=["timestamp", "actual_yield_kwh"])
    )
    actual_context_energy = (
        context_energy.groupby("timestamp", as_index=False)["actual_yield_kwh"]
        .sum(min_count=1)
    )
    estimated_outage_energy = (
        total_energy.groupby("timestamp", as_index=False)["loss_kwh"]
        .sum(min_count=1)
        .dropna(subset=["loss_kwh"])
    )
    aggregated_loss_kwh = estimated_outage_energy["loss_kwh"].sum(min_count=1)
    aggregated_loss_label = (
        f"{aggregated_loss_kwh:,.1f} kWh"
        if pd.notna(aggregated_loss_kwh)
        else "N/A"
    )
    # st.markdown(
    #     f"#### Aggregated energy around the outage · estimated outage loss: "
    #     f"**{aggregated_loss_label}**"
    # )
    context_figure = go.Figure()
    if not actual_context_energy.empty:
        chart_start = pd.Timestamp(selected_period[0]) - pd.Timedelta(days=2)
        chart_end = pd.Timestamp(selected_period[1]) + pd.Timedelta(days=3)
        chart_timestamps = pd.date_range(
            chart_start,
            chart_end,
            freq=pd.Timedelta(hours=STEP_HOURS),
            inclusive="left",
        )
        measured_energy = actual_context_energy.set_index("timestamp")[
            "actual_yield_kwh"
        ].reindex(chart_timestamps)
        context_figure.add_trace(
            go.Scatter(
                x=chart_timestamps,
                y=measured_energy,
                name="Observed energy yield",
                mode="lines",
                connectgaps=False,
                line={"color": PRIMARY_GREEN, "width": 2},
                hovertemplate="%{x}<br>%{y:.2f} kWh<extra>Observed energy yield</extra>",
            )
        )
    if not estimated_outage_energy.empty:
        context_figure.add_trace(
            go.Line(
                x=estimated_outage_energy["timestamp"],
                y=estimated_outage_energy["loss_kwh"],
                name="Estimated outage energy (loss)",
                marker_color=OUTAGE_YELLOW,
                hovertemplate="%{x}<br>%{y:.2f} kWh<extra>Estimated outage energy</extra>",
            )
        )
    context_figure.add_vrect(
        x0=pd.Timestamp(selected_period[0]),
        x1=pd.Timestamp(selected_period[1]) + pd.Timedelta(days=1),
        fillcolor=OUTAGE_YELLOW,
        opacity=0.12,
        line_width=0,
        annotation_text="Outage",
        annotation_position="top left",
    )
    context_figure.update_layout(
        template="baywa",
        height=370,
        barmode="overlay",
        xaxis_title="Time",
        yaxis_title="Energy per 10-minute interval (kWh)",
        legend={"orientation": "h", "y": 1.12},
    )
    st.plotly_chart(context_figure, width="stretch")

    missing_turbines = [
        turbine for turbine in TARGETS if turbine not in result.turbine_types
    ]
    if missing_turbines:
        st.caption(
            "Predictions are not available for all turbines "
            f"({', '.join(missing_turbines)} missing); aggregate values include "
            f"only the available turbines: {', '.join(result.turbine_types)}."
        )
    metrics_table = pd.DataFrame(metric_rows)
    st.dataframe(
        metrics_table,
        column_config={
            "Wind RMSE": st.column_config.TextColumn(
                "Wind RMSE",
                help="Root mean square error of wind speed, in m/s.",
            ),
            "Wind MAE": st.column_config.TextColumn(
                "Wind MAE",
                help="Mean absolute error of wind speed, in m/s.",
            ),
            "Power RMSE": st.column_config.TextColumn(
                "Power RMSE",
                help="Root mean square error of power, in kW.",
            ),
            "Power MAE": st.column_config.TextColumn(
                "Power MAE",
                help="Mean absolute error of power, in kW.",
            ),
            "Estimated energy": st.column_config.TextColumn(
                "Estimated energy",
                help="Estimated energy loss, in kWh.",
            ),
            "95 % range": st.column_config.TextColumn(
                "95 % range",
                help=(
                    "Range for the estimated energy, in kWh, from the energy errors of 83 "
                    "simulated 3-day outages with known truth."
                ),
            ),
            "Actual energy": st.column_config.TextColumn(
                "Actual energy",
                help="Observed actual energy yield, in kWh.",
            ),
            "Energy deviation": st.column_config.TextColumn(
                "Energy deviation",
                help="Estimated minus observed energy yield, in kWh.",
            ),
        },
        hide_index=True,
        width="stretch",
    )
    validation_columns = [
        "Wind RMSE",
        "Wind MAE",
        "Power RMSE",
        "Power MAE",
        "Actual energy",
        "Energy deviation",
    ]
    if metrics_table[validation_columns].eq("N/A").any().any():
        st.caption(
            "N/A means validation data is missing: during an actual outage, "
            "there are no observed turbine readings to compare against."
        )
    st.markdown("#### Estimated energy loss over time")
    total_loss_figure = go.Figure()
    chart_colors = ["#006b37", "#5b8f0b", "#c9ad00"]
    turbine_colors = {
        turbine: chart_colors[index]
        for index, turbine in enumerate(result.turbine_types)
    }
    for turbine in result.turbine_types:
        turbine_energy = result.energy_loss[turbine]
        total_loss_figure.add_trace(
            go.Bar(
                x=turbine_energy["timestamp"],
                y=turbine_energy["loss_kwh"],
                name=turbine,
                marker_color=turbine_colors[turbine],
                hovertemplate="%{x}<br>%{y:.2f} kWh<extra>%{fullData.name}</extra>",
            )
        )
    actual_energy = pd.concat(
        [
            result.energy_loss[turbine][["timestamp", "actual_yield_kwh"]]
            for turbine in result.turbine_types
        ],
        ignore_index=True,
    )
    actual_energy = (
        actual_energy.groupby("timestamp", as_index=False)["actual_yield_kwh"]
        .sum(min_count=1)
        .dropna(subset=["actual_yield_kwh"])
    )
    if not actual_energy.empty:
        total_loss_figure.add_trace(
            go.Line(
                x=actual_energy["timestamp"],
                y=actual_energy["actual_yield_kwh"],
                name="Actual energy yield",
                # mode="lines",
                line={"color": "#20332acd", "width": 1},
                # marker={"size": 6},
                connectgaps=False,
                hovertemplate="%{x}<br>%{y:.2f} kWh<extra>%{fullData.name}</extra>",
            )
        )
    total_loss_figure.update_layout(
        template="baywa",
        height=370,
        barmode="stack",
        xaxis_title="Time",
        yaxis_title="Estimated energy loss per 10-minute interval (kWh)",
        legend={"orientation": "h", "y": 1.12},
    )
    st.plotly_chart(total_loss_figure, width="stretch")
    st.markdown("#### Estimated wind speed over time")
    total_wind_speed_figure = go.Figure()
    for turbine in result.turbine_types:
        turbine_predictions = result.predictions[turbine]
        add_wind_band(
            total_wind_speed_figure,
            turbine_predictions,
            turbine_colors[turbine],
            name=f"{turbine} 95 % band",
            legendgroup=turbine,
        )
        total_wind_speed_figure.add_trace(
            go.Scatter(
                x=turbine_predictions["timestamp"],
                y=turbine_predictions["predicted_wind_speed"],
                name=turbine,
                mode="lines",
                legendgroup=turbine,
                line={"color": turbine_colors[turbine], "width": 2},
            )
        )
    total_wind_speed_figure.update_layout(
        template="baywa",
        height=370,
        xaxis_title="Time",
        yaxis_title="Estimated wind speed (m/s)",
        legend={"orientation": "h", "y": 1.12},
    )
    st.plotly_chart(total_wind_speed_figure, width="stretch")
    st.caption(
        "Shaded: 95 % band per 10-minute step. In 28 simulated outages it contained "
        "the true wind speed 94.8 % of the time."
    )
    with st.expander("How are the total metrics aggregated?"):
        st.write(
            "Wind-speed and power RMSE/MAE are calculated by pooling all available "
            "paired turbine/time observations before calculating the errors; they are "
            "not averages of the individual turbine metrics. Estimated energy and "
            "observed actual energy are summed across uploaded turbines. Energy "
            "deviation is summed only over observations with actual target readings."
        )
    if total_metrics is None:
        st.info("Actual target readings are unavailable for this period; accuracy metrics are N/A.")

    for turbine in result.turbine_types:
        turbine_predictions = result.predictions[turbine]
        turbine_energy = result.energy_loss[turbine]
        with st.expander(f"{turbine} detailed insights"):
            chart_column, loss_column = st.columns([1.6, 1])
            with chart_column:
                st.markdown("#### Wind speed")
                wind_figure = go.Figure()
                add_wind_band(wind_figure, turbine_predictions, turbine_colors[turbine])
                wind_figure.add_trace(
                    go.Scatter(
                        x=turbine_predictions["timestamp"],
                        y=turbine_predictions["actual_wind_speed"],
                        name="Actual",
                        mode="lines",
                        line={"color": "#20332a", "width": 2},
                    )
                )
                wind_figure.add_trace(
                    go.Scatter(
                        x=turbine_predictions["timestamp"],
                        y=turbine_predictions["predicted_wind_speed"],
                        name="Model prediction",
                        mode="lines",
                        line={"color": turbine_colors[turbine], "width": 2},
                    )
                )
                wind_figure.update_layout(
                    template="baywa",
                    height=370,
                    xaxis_title="Time",
                    yaxis_title="Wind speed (m/s)",
                    legend={"orientation": "h", "y": 1.12},
                )
                st.plotly_chart(wind_figure, width="stretch")

            with loss_column:
                st.markdown("#### Estimated energy")
                loss_figure = go.Figure(
                    go.Bar(
                        x=turbine_energy["timestamp"],
                        y=turbine_energy["loss_kwh"],
                        name="Estimated lost yield",
                        marker_color=turbine_colors[turbine],
                        hovertemplate="%{x}<br>%{y:.2f} kWh<extra></extra>",
                    )
                )
                loss_figure.update_layout(
                    template="baywa",
                    height=370,
                    xaxis_title="Time",
                    yaxis_title="Energy per 10-minute interval (kWh)",
                    showlegend=False,
                )
                st.plotly_chart(loss_figure, width="stretch")

    # st.markdown("#### Prediction Method")
    # st.markdown(
    #     "The model fits a separate ordinary least-squares relationship for each uploaded "
    #     "target turbine using the mean wind speeds from parks H and O. The selected "
    #     "outage dates are excluded from fitting; predicted wind speed is converted "
    #     "through that turbine's supplied power curve."
    # )

    if st.button("Draft Insurance Report", type="primary"):
        total_loss = total_energy["loss_kwh"].sum(min_count=1)
        if pd.isna(total_loss):
            st.warning("A report cannot be drafted because no energy estimate is available.")
        else:
            st.info(
                "**Subject:** Wind farm outage – preliminary energy loss estimate\n\n"
                f"Selected turbines: {', '.join(result.turbine_types)}\n\n"
                f"Period: {selected_period[0]:%d %b %Y} to "
                f"{selected_period[1]:%d %b %Y} (inclusive)\n\n"
                f"Estimated energy: {total_loss:,.1f} kWh"
                f" (95 % range: {total_range_label})\n\n"
                "This is a preliminary model estimate and requires review before use."
            )


def energy_range_kwh(
    estimated_kwh: float,
    turbines: Sequence[str],
) -> tuple[float, float] | None:
    """95 % range for an outage's estimated energy (pipeline.evaluate.energy_interval).

    One turbine uses its own backtest errors; several use the park errors, which is exact for
    T1-T3 together and a close approximation for two turbines.
    """
    if pd.isna(estimated_kwh) or not turbines:
        return None
    target = turbines[0] if len(turbines) == 1 else "park"
    try:
        return energy_interval(float(estimated_kwh), target=target)
    except FileNotFoundError:
        return None


def add_wind_band(
    figure: go.Figure,
    predictions: pd.DataFrame,
    color: str,
    name: str = "95 % band",
    legendgroup: str | None = None,
) -> None:
    """Shade the model's 95 % wind-speed band (lower edge, then upper edge filled to it)."""
    fill = f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.18)"
    common = {
        "x": predictions["timestamp"],
        "mode": "lines",
        "line": {"width": 0},
        "legendgroup": legendgroup,
        "hoverinfo": "skip",
    }
    figure.add_trace(
        go.Scatter(y=predictions["predicted_wind_lo"], showlegend=False, **common)
    )
    figure.add_trace(
        go.Scatter(
            y=predictions["predicted_wind_hi"],
            name=name,
            fill="tonexty",
            fillcolor=fill,
            **common,
        )
    )


def _metric_summary_row(
    label: str,
    energy_loss: pd.DataFrame,
    validation_metrics: dict[str, float] | None,
    turbines: Sequence[str],
) -> dict[str, str]:
    """Format the per-turbine or pooled metrics table row."""
    estimated_energy = energy_loss["loss_kwh"].sum(min_count=1)
    actual_energy = energy_loss["actual_yield_kwh"].sum(min_count=1)
    energy_range = energy_range_kwh(estimated_energy, turbines)
    return {
        "Turbine": label,
        "Wind RMSE": (
            f"{validation_metrics['wind_rmse']:.2f}" if validation_metrics else "N/A"
        ),
        "Wind MAE": (
            f"{validation_metrics['wind_mae']:.2f}" if validation_metrics else "N/A"
        ),
        "Power RMSE": (
            f"{validation_metrics['power_rmse']:.2f}" if validation_metrics else "N/A"
        ),
        "Power MAE": (
            f"{validation_metrics['power_mae']:.2f}" if validation_metrics else "N/A"
        ),
        "Estimated energy": (
            f"{estimated_energy:,.1f}" if pd.notna(estimated_energy) else "N/A"
        ),
        "95 % range": (
            f"{energy_range[0]:,.0f} – {energy_range[1]:,.0f}" if energy_range else "N/A"
        ),
        "Actual energy": (
            f"{actual_energy:,.1f}" if pd.notna(actual_energy) else "N/A"
        ),
        "Energy deviation": (
            f"{validation_metrics['energy_deviation_kwh']:+,.1f}"
            if validation_metrics else "N/A"
        ),
    }


def render_turbine_status(
    available_turbines: Sequence[str],
    expected_turbines: Sequence[str],
) -> None:
    """Show which expected turbines have data in the uploaded reference files."""
    st.subheader("Data Availability")
    columns = st.columns(len(expected_turbines))

    for column, turbine in zip(columns, expected_turbines):
        with column:
            if turbine in available_turbines:
                st.markdown(f"🟢 :green[**{turbine}**]")
            else:
                st.markdown(f"🔴 :red[**{turbine}**]")

    missing_turbines = set(expected_turbines) - set(available_turbines)
    if missing_turbines:
        st.warning(f"Missing data for: {', '.join(sorted(missing_turbines))}")


def render_methodology(result: AnalysisResult | None) -> None:
    """Explain the method (methods_tabs.py), then show metrics only for observed targets."""
    methods_tabs.render_methodology_story()

    st.markdown("#### Check on your selected period")
    st.caption(
        "Only where the uploaded target turbines have measured wind. "
        "Totals pool all turbine steps; energy deviation is predicted minus actual kWh."
    )
    if result is None:
        st.info("Submit uploaded data to view predictions and any available validation metrics.")
        return

    predictions = pd.concat(result.predictions.values(), ignore_index=True)
    energy_loss = pd.concat(result.energy_loss.values(), ignore_index=True)
    validation_metrics = calculate_validation_metrics(predictions, energy_loss)
    if validation_metrics is None:
        st.info("No observed target readings are available for validation in this period.")
        return

    paired = predictions.dropna(
        subset=["actual_wind_speed", "predicted_wind_speed"]
    )
    metric_columns = st.columns(5)
    metric_columns[0].metric(
        "Wind speed RMSE", f"{validation_metrics['wind_rmse']:.2f} m/s"
    )
    metric_columns[1].metric(
        "Wind speed MAE", f"{validation_metrics['wind_mae']:.2f} m/s"
    )
    metric_columns[2].metric(
        "Power RMSE", f"{validation_metrics['power_rmse']:.2f} kW"
    )
    metric_columns[3].metric(
        "Power MAE", f"{validation_metrics['power_mae']:.2f} kW"
    )
    metric_columns[4].metric(
        "Total energy-yield deviation",
        f"{validation_metrics['energy_deviation_kwh']:+,.1f} kWh",
        help="Predicted minus actual energy yield, summed over timestamps with observed target data.",
    )

    st.markdown("#### Actual vs. predicted wind speed")
    scatter_figure = go.Figure(
        go.Scatter(
            x=paired["actual_wind_speed"],
            y=paired["predicted_wind_speed"],
            mode="markers",
            name="Observed validation points",
            marker={"color": PRIMARY_GREEN, "size": 9, "opacity": 0.8},
        )
    )
    scatter_figure.update_layout(
        template="baywa",
        height=430,
        xaxis_title="Actual target wind speed (m/s)",
        yaxis_title="Predicted target wind speed (m/s)",
        showlegend=False,
    )
    st.plotly_chart(scatter_figure, width="stretch")


def main() -> None:
    image_directory = Path(__file__).resolve().parent / "app" / "images"

    st.set_page_config(
        page_title="Munich Climate Week - BayWa r.e. | Wind Farm Outage",
        page_icon=str(image_directory / "QUENTE.png"),
        layout="wide",
    )
    apply_custom_theme()
    apply_page_styles()

    with st.sidebar:
        col1, col2 = st.columns([3, 2])

        with col1:
            st.image(str(image_directory / "MUCClimateWeekLogo.png"), use_container_width=True)
        with col2:
            st.image(str(image_directory / "QUENTE.png"), use_container_width=True)
        st.image(str(image_directory / "LogoBayWa.webp"), width="stretch")

        st.markdown("## Analysis setup")
        st.markdown("**Quick start with the challenge data**")
        with st.container(horizontal=True):
            st.button(
                "Public test set (4–9 Sep)",
                icon=":material/fact_check:",
                on_click=run_bundled_analysis,
                args=(PUBLIC_TEST_PERIOD,),
                help="Bundled challenge files; park T is known here, so accuracy metrics are shown.",
                width="stretch",
            )
            st.button(
                "Judged outage (1–3 Sep)",
                icon=":material/bolt:",
                on_click=run_bundled_analysis,
                args=(DEFAULT_OUTAGE_PERIOD,),
                help="Bundled challenge files; park T is missing here (hidden test set).",
                width="stretch",
            )
        st.markdown("**Or upload your own data**")
        st.caption("Configure the data and outage period for your analysis.")
        reference_file = st.file_uploader(
            "Reference turbines H1–H5 and O1–O2 (one raw file per turbine)",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            key="reference_wind_data",
        )
        target_files = st.file_uploader(
            "Target turbine data (upload any of T1, T2, and T3)",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            key="affected_wind_data",
        )
        st.caption(
            "Upload the original per-turbine data files, including turbine_name, timestamp, "
            "sensor columns, and test_set. The app applies the project's cleaning rules."
        )

        # if reference_file is not None or target_files:
        #     st.info("The baseline requires one file for each reference turbine H1–H5 and O1–O2.")

        st.session_state.setdefault("outage_period", DEFAULT_OUTAGE_PERIOD)
        if not target_files:
            st.session_state.pop("outage_upload_fingerprint", None)
            st.session_state.pop("outage_estimation_error", None)
        else:
            upload_fingerprint = tuple(
                (uploaded_file.name, hashlib.sha256(uploaded_file.getvalue()).hexdigest())
                for uploaded_file in target_files
            )
            if st.session_state.get("outage_upload_fingerprint") != upload_fingerprint:
                st.session_state["outage_upload_fingerprint"] = upload_fingerprint
                st.session_state.pop("outage_estimation_error", None)
                try:
                    target_raw = load_data(target_files)
                    target_names = get_available_turbines(target_raw)
                    if not target_names or not set(target_names).issubset(TARGETS):
                        raise ValueError("Upload at least one target turbine file for T1, T2, or T3.")
                    target_data = preprocess_data(target_raw)
                    target_wide = to_wide(target_data)
                    st.session_state["outage_period"] = estimate_outage_period_for_targets(
                        target_wide,
                        target_names,
                    )
                except ValueError as exc:
                    st.session_state["outage_estimation_error"] = str(exc)

        if "outage_estimation_error" in st.session_state:
            st.warning(
                "The outage period could not be estimated automatically: "
                f"{st.session_state['outage_estimation_error']}"
            )
        elif target_files:
            st.success(
                "The outage period was estimated from the longest missing interval "
                "among uploaded target turbines. "
                "You can adjust it below."
            )

        selected_dates = st.date_input(
            "Outage period",
            key="outage_period",
        )

        if (
            isinstance(selected_dates, tuple)
            and len(selected_dates) == 2
            and selected_dates[0] is not None
            and selected_dates[1] is not None
        ):
            selected_period = (selected_dates[0], selected_dates[1])
        else:
            selected_period = DEFAULT_OUTAGE_PERIOD

        if st.button(
            "Submit data and start calculation",
            type="primary",
            width="stretch",
        ):
            st.session_state.pop("analysis_result", None)
            st.session_state.pop("analysis_period", None)
            try:
                raw_references = load_data(reference_file)
                st.session_state["available_turbines"] = get_available_turbines(
                    raw_references
                )
                if not target_files:
                    raise ValueError(
                        "Upload at least one target turbine file (T1, T2, or T3)."
                    )
                raw_target = load_data(target_files)
                with st.spinner("Cleaning uploaded data and fitting the model..."):
                    result = run_analysis(
                        raw_references=raw_references,
                        raw_target=raw_target,
                        outage_period=selected_period,
                    )
                st.session_state["analysis_result"] = result
                st.session_state["analysis_period"] = selected_period
                st.session_state["main_tabs"] = "Dashboard & Results"
                st.session_state.pop("analysis_error", None)
            except ValueError as exc:
                st.session_state["analysis_error"] = str(exc)

        if "analysis_error" in st.session_state:
            st.error(st.session_state["analysis_error"])

    st.session_state.setdefault("available_turbines", [])
    available_turbines: list[str] = st.session_state.get("available_turbines", [])
    result = st.session_state.get("analysis_result")
    analysis_period = st.session_state.get("analysis_period", selected_period)

    st.title("Wind Farm Outage Analysis")
    st.caption("BayWa r.e. · Munich Climate Week Hackathon Challenge 1")
    dashboard_tab, test_tab, methodology_tab, tried_tab = st.tabs(
        [
            "Dashboard & Results",
            "Test Set Results",
            "Methodology & Validation",
            "Things We Tried but Didn't Work Out",
        ],
        default="Dashboard & Results",
        key="main_tabs",
        on_change="rerun",
    )

    if dashboard_tab.open:
        with dashboard_tab:
            render_dashboard(
                analysis_period,
                result,
                available_turbines,
            )

    if test_tab.open:
        with test_tab:
            methods_tabs.render_test_results()

    if methodology_tab.open:
        with methodology_tab:
            render_methodology(result)

    if tried_tab.open:
        with tried_tab:
            methods_tabs.render_tried()


if __name__ == "__main__":
    main()
