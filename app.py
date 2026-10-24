"""Streamlit app for estimating wind farm outage energy from uploaded turbine data."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from baseline import OLSBaseline, park_means
from preprocessing import (
    REFERENCES,
    STEP_HOURS,
    TARGETS,
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

class UploadedTurbineFile(Protocol):
    name: str

    def getvalue(self) -> bytes: ...


@dataclass(frozen=True)
class AnalysisResult:
    predictions: pd.DataFrame
    energy_loss: pd.DataFrame
    turbine_type: str
    available_turbines: list[str]


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
    return timestamps[best_start].date(), timestamps[best_end - 1].date()


def get_power_curve(turbine_type: str) -> pd.DataFrame:
    """Load the supplied power curve for the selected turbine."""
    if turbine_type not in TARGETS:
        raise ValueError(f"Unknown target turbine: {turbine_type}.")
    return load_power_curve(turbine_type)


def calculate_predictions(
    wind_table: pd.DataFrame,
    outage_period: tuple[date, date],
    model_params: Mapping[str, object],
) -> pd.DataFrame:
    """Fit the OLS baseline outside the selected dates and predict the selected window."""
    start_date, end_date = outage_period
    if end_date < start_date:
        raise ValueError("The outage end date must be on or after the start date.")
    turbine_type = model_params.get("turbine")
    if not isinstance(turbine_type, str) or turbine_type not in TARGETS:
        raise ValueError("A valid target turbine (T1, T2, or T3) is required.")

    end_exclusive = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    model = OLSBaseline(targets=[turbine_type]).fit(
        wind_table,
        exclude=(start_date, end_exclusive),
    )
    window = wind_table.loc[
        (wind_table.index >= pd.Timestamp(start_date))
        & (wind_table.index < end_exclusive)
    ]
    if window.empty:
        raise ValueError("The selected outage period is outside the uploaded data range.")

    predicted = model.predict(window)[turbine_type]
    return pd.DataFrame(
        {
            "timestamp": window.index,
            "reference_wind_speed": park_means(window)["H"].to_numpy(),
            "predicted_wind_speed": predicted.to_numpy(),
            "actual_wind_speed": window[turbine_type].to_numpy(),
        }
    )


def calculate_energy_loss(
    predictions: pd.DataFrame,
    power_curve: pd.DataFrame,
) -> pd.DataFrame:
    """Convert predicted wind speed to predicted energy per 10-minute step."""
    power_kw = wind_to_power(predictions["predicted_wind_speed"], power_curve)
    return pd.DataFrame(
        {
            "timestamp": predictions["timestamp"],
            "loss_kwh": power_kw * STEP_HOURS,
        }
    )


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
            "Upload all reference turbines required by the baseline: "
            f"{', '.join(sorted(missing_references))}."
        )

    target_names = get_available_turbines(raw_target)
    if len(target_names) != 1 or target_names[0] not in TARGETS:
        raise ValueError("Upload one target turbine file containing T1, T2, or T3.")
    selected_turbine = target_names[0]

    cleaned = preprocess_data(pd.concat([raw_references, raw_target], ignore_index=True))
    wind_table = to_wide(cleaned)
    predictions = calculate_predictions(
        wind_table,
        outage_period,
        {"turbine": selected_turbine},
    )
    energy_loss = calculate_energy_loss(
        predictions,
        get_power_curve(selected_turbine),
    )
    return AnalysisResult(
        predictions=predictions,
        energy_loss=energy_loss,
        turbine_type=selected_turbine,
        available_turbines=sorted(reference_names),
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
    turbine_type: str | None,
    result: AnalysisResult | None,
    available_turbines: Sequence[str],
) -> None:
    """Render the outage estimate and any metrics supported by observed data."""
    st.subheader("Outage overview")
    turbine_label = (
        turbine_type
        if turbine_type is not None
        else "inferred from affected data on submission"
    )
    st.caption(
        f"Selected period: {selected_period[0]:%d %b %Y} – "
        f"{selected_period[1]:%d %b %Y} · Turbine: {turbine_label}"
    )

    render_turbine_status(available_turbines, EXPECTED_TURBINES)

    if result is None:
        st.info("Upload the wind data and submit it from the sidebar to start the calculation.")
        return

    predictions = result.predictions
    energy_loss = result.energy_loss
    paired = predictions.dropna(subset=["actual_wind_speed", "predicted_wind_speed"])
    if not paired.empty:
        errors = paired["predicted_wind_speed"] - paired["actual_wind_speed"]
        metric_columns = st.columns(2)
        metric_columns[0].metric("RMSE wind speed", f"{math.sqrt((errors ** 2).mean()):.2f} m/s")
        metric_columns[1].metric("MAE wind speed", f"{errors.abs().mean():.2f} m/s")
    else:
        st.info("Actual target readings are unavailable for this period; accuracy metrics are not shown.")

    total_loss = float(energy_loss["loss_kwh"].sum(min_count=1))
    if not energy_loss.empty and not math.isnan(total_loss):
        st.metric("Estimated energy loss", f"{total_loss:,.1f} kWh")
    else:
        st.warning("Energy loss could not be estimated because reference wind-speed data is missing.")

    chart_column, loss_column = st.columns([1.6, 1])
    with chart_column:
        st.markdown("#### Wind Speed Visualization")
        wind_figure = go.Figure()
        wind_figure.add_trace(
            go.Scatter(
                x=predictions["timestamp"],
                y=predictions["actual_wind_speed"],
                name="Actual",
                mode="lines",
                line={"color": PRIMARY_GREEN, "width": 2},
            )
        )
        wind_figure.add_trace(
            go.Scatter(
                x=predictions["timestamp"],
                y=predictions["predicted_wind_speed"],
                name="Baseline prediction",
                mode="lines",
                line={"color": PREDICTION_GREEN, "width": 2},
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
        st.markdown("#### Energy Loss Calculation")
        loss_figure = go.Figure(
            go.Bar(
                x=energy_loss["timestamp"],
                y=energy_loss["loss_kwh"],
                name="Estimated lost yield",
                marker_color=OUTAGE_YELLOW,
                hovertemplate="%{x}<br>%{y:.2f} kWh<extra></extra>",
            )
        )
        loss_figure.update_layout(
            template="baywa",
            height=370,
            xaxis_title="Time",
            yaxis_title="Lost yield (kWh / 10 min)",
            showlegend=False,
        )
        st.plotly_chart(loss_figure, width="stretch")

    st.markdown("#### Prediction Method")
    st.markdown(
        "The model fits a separate ordinary least-squares relationship for the selected "
        "target turbine using the mean wind speeds from parks H and O. The selected "
        "outage dates are excluded from fitting; predicted wind speed is converted "
        "through that turbine's supplied power curve."
    )

    if st.button("Draft Insurance Report", type="primary"):
        if math.isnan(total_loss):
            st.warning("A report cannot be drafted because no energy estimate is available.")
        else:
            st.info(
                "**Subject:** Wind farm outage – preliminary energy loss estimate\n\n"
                f"Selected turbine: {turbine_type}\n\n"
                f"Period: {selected_period[0]:%d %b %Y} to "
                f"{selected_period[1]:%d %b %Y} (inclusive)\n\n"
                f"Estimated energy: {total_loss:,.1f} kWh\n\n"
                "This is a preliminary model estimate and requires review before use."
            )


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
    """Describe the implemented baseline and show metrics only for observed targets."""
    st.subheader("Methodology & Validation")
    st.markdown("### Prediction Method")
    st.markdown(
        "Raw turbine uploads are cleaned with the project's preprocessing rules. "
        "For each target turbine, the OLS baseline predicts wind speed from the "
        "mean park H and park O wind speeds. The selected outage window is excluded "
        "from model fitting. Predicted wind speed is mapped to power using the "
        "selected turbine's supplied power curve, then multiplied by 1/6 hour per step."
    )
    st.markdown("### Accuracy and Uncertainty")
    st.markdown(
        "Accuracy is calculated only for timestamps in the selected period where "
        "the uploaded target turbine has an observed wind-speed value. Periods "
        "without target observations have no accuracy metrics."
    )
    if result is None:
        st.info("Submit uploaded data to view predictions and any available validation metrics.")
        return

    paired = result.predictions.dropna(
        subset=["actual_wind_speed", "predicted_wind_speed"]
    )
    if paired.empty:
        st.info("No observed target readings are available for validation in this period.")
        return

    errors = paired["predicted_wind_speed"] - paired["actual_wind_speed"]
    metric_columns = st.columns(2)
    metric_columns[0].metric("Wind speed RMSE", f"{math.sqrt((errors ** 2).mean()):.2f} m/s")
    metric_columns[1].metric("Wind speed MAE", f"{errors.abs().mean():.2f} m/s")

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
    st.set_page_config(
        page_title="Munich Climate Week - BayWa r.e. | Wind Farm Outage",
        page_icon="🌬️",
        layout="wide",
    )
    apply_custom_theme()
    apply_page_styles()

    with st.sidebar:
        image_directory = Path(__file__).resolve().parent / "app" / "images"
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.image(str(image_directory / "MUCClimateWeekLogo.png"), use_container_width=True)
        st.image(str(image_directory / "LogoBayWa.webp"), width="stretch")

        st.markdown("## Analysis setup")
        st.caption("Configure the data and outage period for your analysis.")
        reference_file = st.file_uploader(
            "Reference turbines H1–H5 and O1–O2 (one raw file per turbine)",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            key="reference_wind_data",
        )
        affected_file = st.file_uploader(
            "Affected turbine data (one raw T1, T2, or T3 file)",
            type=["csv", "xlsx", "xls"],
            key="affected_wind_data",
        )
        st.caption(
            "Upload the original per-turbine data files, including turbine_name, timestamp, "
            "sensor columns, and test_set. The app applies the project's cleaning rules."
        )

        if reference_file is not None or affected_file is not None:
            st.info("The baseline requires one file for each reference turbine H1–H5 and O1–O2.")

        st.session_state.setdefault("outage_period", DEFAULT_OUTAGE_PERIOD)
        if affected_file is None:
            st.session_state.pop("outage_upload_fingerprint", None)
            st.session_state.pop("outage_estimation_error", None)
        else:
            upload_fingerprint = (
                affected_file.name,
                hashlib.sha256(affected_file.getvalue()).hexdigest(),
            )
            if st.session_state.get("outage_upload_fingerprint") != upload_fingerprint:
                st.session_state["outage_upload_fingerprint"] = upload_fingerprint
                st.session_state.pop("outage_estimation_error", None)
                try:
                    target_raw = load_data(affected_file)
                    target_names = get_available_turbines(target_raw)
                    if len(target_names) != 1 or target_names[0] not in TARGETS:
                        raise ValueError(
                            "Upload one affected turbine file for T1, T2, or T3."
                        )
                    target_data = preprocess_data(target_raw)
                    target_wide = to_wide(target_data)
                    st.session_state["outage_period"] = estimate_outage_period(
                        target_wide[target_names[0]]
                    )
                except ValueError as exc:
                    st.session_state["outage_estimation_error"] = str(exc)

        if "outage_estimation_error" in st.session_state:
            st.warning(
                "The outage period could not be estimated automatically: "
                f"{st.session_state['outage_estimation_error']}"
            )
        elif affected_file is not None:
            st.success(
                "The outage period was estimated from the affected turbine data. "
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
                if affected_file is None:
                    raise ValueError(
                        "Upload one affected turbine file (T1, T2, or T3)."
                    )
                raw_target = load_data(affected_file)
                with st.spinner("Cleaning uploaded data and fitting the baseline..."):
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
    analysis_turbine = result.turbine_type if result is not None else None

    st.title("Wind Farm Outage Analysis")
    st.caption("BayWa r.e. · Munich Climate Week Hackathon Challenge 1")
    dashboard_tab, methodology_tab, tried_tab = st.tabs(
        [
            "Dashboard & Results",
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
                analysis_turbine,
                result,
                available_turbines,
            )

    if methodology_tab.open:
        with methodology_tab:
            render_methodology(result)

    if tried_tab.open:
        with tried_tab:
            st.subheader("Things We Tried but Didn't Work Out")
            st.caption(
                "Document approaches that did not work so the team can avoid "
                "repeating them."
            )


if __name__ == "__main__":
    main()
