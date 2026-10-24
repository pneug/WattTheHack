"""Streamlit frontend skeleton for the BayWa r.e. wind farm outage challenge."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st


PRIMARY_GREEN = "#008b47"
PREDICTION_GREEN = "#77b90f"
OUTAGE_YELLOW = "#fad804"
DEFAULT_OUTAGE_PERIOD = (date(2025, 1, 15), date(2025, 1, 16))

TURBINE_MODELS = ("T1", "T2", "T3")


@dataclass(frozen=True)
class AnalysisResult:
    predictions: pd.DataFrame
    energy_loss: pd.DataFrame


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


def load_data(file: object | Sequence[object] | None) -> pd.DataFrame:
    """Return illustrative data until the data-ingestion backend is connected."""
    timestamps = pd.date_range(
        start="2025-01-14 00:00",
        periods=72,
        freq="10min",
        tz="Europe/Berlin",
    )
    reference_speeds = [
        round(7.4 + 1.5 * math.sin(index / 8) + 0.4 * math.cos(index / 3), 2)
        for index in range(len(timestamps))
    ]
    affected_speeds = [
        round(max(0.0, speed * 0.92 + 0.25 * math.sin(index / 5)), 2)
        for index, speed in enumerate(reference_speeds)
    ]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "reference_wind_speed": reference_speeds,
            "affected_wind_speed": affected_speeds,
        }
    )


def estimate_outage_period(df: pd.DataFrame) -> tuple[date, date]:
    """Return a fixed example outage period until automatic detection is implemented."""
    return DEFAULT_OUTAGE_PERIOD


def get_power_curve(turbine_type: str) -> pd.DataFrame:
    """Return an illustrative power curve for the selected turbine."""
    return pd.DataFrame(
        {
            "wind_speed": [0, 3, 5, 7, 9, 11, 13, 15, 25],
            "power_kw": [0, 0, 180, 520, 1050, 1650, 2000, 2000, 0],
        }
    )


def calculate_predictions(
    df_reference: pd.DataFrame,
    outage_period: tuple[date, date],
    model_params: Mapping[str, float],
) -> pd.DataFrame:
    """Return illustrative actual and predicted wind speeds for the selected period."""
    start_date, end_date = outage_period
    timestamps = pd.date_range(
        start=pd.Timestamp(start_date, tz="Europe/Berlin"),
        end=pd.Timestamp(end_date + timedelta(days=1), tz="Europe/Berlin")
        - pd.Timedelta(minutes=10),
        freq="10min",
    )
    reference_speeds = [
        round(7.2 + 1.6 * math.sin(index / 15) + 0.35 * math.cos(index / 7), 2)
        for index in range(len(timestamps))
    ]
    predicted_speeds = [
        round(max(0.0, speed * 0.94 + 0.18 * math.sin(index / 9)), 2)
        for index, speed in enumerate(reference_speeds)
    ]
    actual_speeds = [
        round(max(0.0, predicted + 0.25 * math.cos(index / 6)), 2)
        for index, predicted in enumerate(predicted_speeds)
    ]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "reference_wind_speed": reference_speeds,
            "predicted_wind_speed": predicted_speeds,
            "actual_wind_speed": actual_speeds,
        }
    )


def calculate_energy_loss(
    predictions: pd.DataFrame,
    power_curve: pd.DataFrame,
) -> pd.DataFrame:
    """Return illustrative energy-loss values until power conversion is implemented."""
    losses = [
        round(0.62 + 0.18 * (1 + math.sin(index / 11)), 2)
        for index in range(len(predictions))
    ]
    return pd.DataFrame(
        {
            "timestamp": predictions["timestamp"],
            "loss_kwh": losses,
        }
    )


def check_if_ground_truth_available(outage_period: tuple[date, date]) -> bool:
    """Report mock ground-truth availability for the dashboard example."""
    return True


@st.cache_data
def get_validation_scatter_data() -> tuple[list[float], list[float]]:
    """Return fixed example data for the validation scatter plot."""
    reference_speeds = [round(4.2 + index * 0.22, 2) for index in range(30)]
    affected_speeds = [
        round(speed * 0.91 + 0.35 * math.sin(index / 2.5), 2)
        for index, speed in enumerate(reference_speeds)
    ]
    return reference_speeds, affected_speeds


def run_analysis(
    reference_files: Sequence[object],
    affected_file: object,
    outage_period: tuple[date, date],
    turbine_type: str,
    model_params: Mapping[str, float],
) -> AnalysisResult:
    """Submit the selected inputs to the replaceable backend calculation pipeline."""
    reference_data = load_data(reference_files)
    load_data(affected_file)
    predictions = calculate_predictions(reference_data, outage_period, model_params)
    energy_loss = calculate_energy_loss(predictions, get_power_curve(turbine_type))
    return AnalysisResult(predictions=predictions, energy_loss=energy_loss)


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
    turbine_type: str,
    result: AnalysisResult | None,
) -> None:
    """Render dashboard metrics, visualizations, and report placeholder."""
    st.subheader("Outage overview")
    st.caption(
        f"Selected period: {selected_period[0]:%d %b %Y} – "
        f"{selected_period[1]:%d %b %Y} · Turbine: {turbine_type}"
    )

    if result is None:
        st.info("Upload the wind data and submit it from the sidebar to start the calculation.")
        return

    has_ground_truth = check_if_ground_truth_available(selected_period)
    if has_ground_truth:
        metric_columns = st.columns(3)
        metric_columns[0].metric("RMSE wind speed", "0.42 m/s", help="Illustrative mock metric")
        metric_columns[1].metric("MAE wind speed", "0.31 m/s", help="Illustrative mock metric")
        metric_columns[2].metric(
            "Energy yield deviation",
            "4.8%",
            help="Illustrative mock metric",
        )
    else:
        st.info("Ground-truth data is not available for the selected period.")

    predictions = result.predictions
    energy_loss = result.energy_loss
    if not predictions.empty and not energy_loss.empty:
        total_loss = energy_loss["loss_kwh"].sum()
        st.metric("Estimated energy loss", f"{total_loss:,.1f} kWh")

    chart_column, loss_column = st.columns([1.6, 1])
    with chart_column:
        st.markdown("#### Wind Speed Visualization")
        wind_figure = go.Figure()
        wind_figure.add_trace(
            go.Scatter(
                x=predictions["timestamp"],
                y=predictions["actual_wind_speed"],
                name="Actual (mock ground truth)",
                mode="lines",
                line={"color": PRIMARY_GREEN, "width": 2},
            )
        )
        wind_figure.add_trace(
            go.Scatter(
                x=predictions["timestamp"],
                y=predictions["predicted_wind_speed"],
                name="Predicted (mock)",
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

    st.caption("Illustrative mock results. Replace the backend pipeline before using real data.")
    st.markdown("#### Prediction Method")
    st.markdown("[To be added later]")

    if st.button("Generate Insurance Report (Email)", type="primary"):
        st.info(
            "**Subject:** Wind farm outage – preliminary energy loss estimate\n\n"
            f"Dear Claims Team,\n\n"
            f"A mock outage for {selected_period[0]:%d %b %Y} to "
            f"{selected_period[1]:%d %b %Y} is shown in this demo. "
            f"The illustrative estimated energy loss is {total_loss:,.1f} kWh.\n\n"
            "This draft contains placeholder values and is not an insurance claim."
        )


def render_methodology() -> None:
    """Render fixed test KPIs and placeholders without describing an unbuilt model."""
    st.subheader("Methodology & Validation")
    st.markdown("### Prediction Method")
    st.markdown("[To be added later]")
    st.markdown("### Accuracy and Uncertainty")
    st.markdown("[To be added later]")

    st.markdown("#### Fixed test dataset metrics")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Wind speed RMSE", "0.46 m/s")
    metric_columns[1].metric("Wind speed MAE", "0.34 m/s")
    metric_columns[2].metric("Power RMSE", "142 kW")
    metric_columns[3].metric("Power MAE", "96 kW")
    st.caption("Fixed illustrative values; replace with validated test-set results.")

    st.markdown("#### Correlation Scatter Plot")
    reference_speeds, affected_speeds = get_validation_scatter_data()
    scatter_figure = go.Figure(
        go.Scatter(
            x=reference_speeds,
            y=affected_speeds,
            mode="markers",
            name="Mock paired observations",
            marker={"color": PRIMARY_GREEN, "size": 9, "opacity": 0.8},
        )
    )
    scatter_figure.update_layout(
        template="baywa",
        height=430,
        xaxis_title="Reference wind farm (m/s)",
        yaxis_title="Affected wind farm (m/s)",
        showlegend=False,
    )
    st.plotly_chart(scatter_figure, width="stretch")
    st.caption("Illustrative mock data only.")


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
            "Reference wind farm data (multiple files allowed)",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            key="reference_wind_data",
        )
        affected_file = st.file_uploader(
            "Affected wind farm data (single file)",
            type=["csv", "xlsx", "xls"],
            key="affected_wind_data",
        )
        st.caption(
            "Supported formats: CSV, Excel (.xlsx, .xls). "
            "Expected data: Berlin timestamps and 10-minute average wind speeds."
        )

        automatic_selection = st.checkbox("Enable automatic turbine selection")
        turbine_type = st.selectbox(
            "Turbine model",
            options=TURBINE_MODELS,
            disabled=automatic_selection,
        )

        if reference_file is not None or affected_file is not None:
            st.info("Uploads received. CSV processing will be connected by the backend.")

        st.session_state.setdefault("outage_period", DEFAULT_OUTAGE_PERIOD)
        if st.button("Estimate period automatically", width="stretch"):
            reference_files = reference_file or []
            st.session_state["outage_period"] = estimate_outage_period(
                load_data(reference_files)
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
            if not reference_file or affected_file is None:
                st.session_state.pop("analysis_result", None)
                st.session_state.pop("analysis_period", None)
                st.session_state.pop("analysis_turbine", None)
                st.session_state["analysis_error"] = (
                    "Upload at least one reference file and one affected wind data file."
                )
            else:
                with st.spinner("Submitting data and calculating energy loss..."):
                    result = run_analysis(
                        reference_files=reference_file,
                        affected_file=affected_file,
                        outage_period=selected_period,
                        turbine_type=turbine_type,
                        model_params={
                            "automatic_turbine_selection": float(automatic_selection)
                        },
                    )
                st.session_state["analysis_result"] = result
                st.session_state["analysis_period"] = selected_period
                st.session_state["analysis_turbine"] = turbine_type
                st.session_state["main_tabs"] = "Dashboard & Results"
                st.session_state.pop("analysis_error", None)

        if "analysis_error" in st.session_state:
            st.error(st.session_state["analysis_error"])

    result = st.session_state.get("analysis_result")
    analysis_period = st.session_state.get("analysis_period", selected_period)
    analysis_turbine = st.session_state.get("analysis_turbine", turbine_type)

    st.title("Wind Farm Outage Analysis")
    st.caption("BayWa r.e. · Energy loss estimation workspace")
    dashboard_tab, methodology_tab = st.tabs(
        ["Dashboard & Results", "Methodology & Validation"],
        default="Dashboard & Results",
        key="main_tabs",
        on_change="rerun",
    )

    if dashboard_tab.open:
        with dashboard_tab:
            render_dashboard(analysis_period, analysis_turbine, result)

    if methodology_tab.open:
        with methodology_tab:
            render_methodology()


if __name__ == "__main__":
    main()
