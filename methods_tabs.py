"""Content of the "Test Set Results", "Methodology & Validation" and "Things We Tried but Didn't Work Out" tabs.

Kept out of app.py so the dashboard and these explanation tabs can be edited without git overlap.
Figures come from reports/app_figures/*.json (regenerate: `python -m pipeline.figures`), so the app
needs no backtest step files at runtime. Source for all numbers: docs/insights.md, experiments/LOG.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

FIGURES = Path(__file__).resolve().parent / "reports" / "app_figures"


@st.cache_resource(max_entries=16)
def _figure(name: str) -> go.Figure | None:
    path = FIGURES / f"{name}.json"
    if not path.exists():
        return None
    return go.Figure(json.loads(path.read_text(encoding="utf-8")))


def _chart(name: str, alt: str) -> None:
    fig = _figure(name)
    if fig is None:
        st.caption(f"Figure `{name}` missing: run `python -m pipeline.figures`.")
        return
    st.plotly_chart(fig, theme=None, key=f"methods_{name}", alt=alt)


def render_methodology_story() -> None:
    """How the model works, what it adds to the baseline and the four key insights (pitch storyline)."""
    st.subheader("Methodology")
    st.markdown(
        "**Idea:** when park T is down, its neighbours H and O still measure the wind. "
        "We learn how T's wind relates to theirs, predict T's wind during the outage, "
        "and turn it into lost energy with T's power curve."
    )
    with st.container(horizontal=True):
        st.metric("1 · Learn", "T vs H and O", "all days without outage", delta_color="off", border=True)
        st.metric("2 · Predict", "T's wind", "every 10 minutes", delta_color="off", border=True)
        st.metric("3 · Convert", "lost kWh", "via the power curve", delta_color="off", border=True)

    st.markdown("#### Baseline vs our model")
    st.table(
        pd.DataFrame(
            [
                ("Neighbour wind", "10-minute values", "1-hour averages (gusts are noise)"),
                ("Wind direction", "ignored", "own factor for each of 8 directions"),
                ("Wind → power", "power of the single best guess", "power averaged over the likely winds"),
                ("Uncertainty", "none", "95 % band per step, 95 % range for the energy"),
            ],
            columns=["", "Baseline", "Our model"],
        ),
        hide_index=True,
        alt="Comparison of the baseline and our model",
    )
    st.latex(r"v_T = a_{\text{dir}} + b_{\text{dir}} \cdot \overline{v}_H^{\,1h} + c \cdot \overline{v}_O^{\,1h}")
    st.caption("Still one transparent formula per turbine: every coefficient can be printed and explained.")

    st.markdown("#### Four things we learned")

    st.markdown("**1 · Test on many outages, not one lucky week.** We hid 3 days of park T at a time and "
                "predicted them blind, 28 times. Error in park energy: **11.0 % → 8.5 %**.")
    _chart("distribution", "Park energy deviation per simulated outage, baseline vs final model")

    st.markdown("**2 · Average the power, not the wind.** The power curve bends upward, so a gust adds more "
                "than a lull removes. Using only the best-guess wind underestimated energy by **8.5 %**; "
                "averaging the power fixes it (**+1.4 %**).")
    _chart("power_evidence", "Mean power by predicted wind speed: plain curve, averaged curve and actual")

    st.markdown("**3 · What remains is weather luck, so we show a range.** An outage's energy error is its "
                "3-day average wind error, about ±11 %. No neighbour data can remove it, so we report a range.")
    with st.container(horizontal=True):
        st.metric("Lost energy, 1–3 Sep", "322 MWh", border=True)
        st.metric("95 % range", "262–387 MWh", border=True,
                  help="From the energy errors of 83 simulated outages (one starting every day).")

    st.markdown("**4 · Transparent beat black box.** Gradient boosting reached 9.7 %, a neural network 9.1 %; "
                "our formula 8.5 %, and every number is explainable.")
    _chart("blackbox", "Power RMSE vs park energy error for all experiments, black box highlighted")

    with st.expander("Details", icon=":material/info:"):
        st.markdown(
            "- **Direction:** T gets 84 % (north) to 93 % (south-west) of H's wind: terrain and park layout.\n"
            "- **Safety rules:** park T is never used during the outage; if an input is missing, that step falls "
            "back to the baseline formula, so there are no gaps.\n"
            "- **The 95 % wind band** contained the true wind 94.8 % of the time."
        )
        _chart("direction", "Share of park H wind reaching park T by wind direction")


TRIED = pd.DataFrame(
    [
        ("Black box: gradient boosting", "9.7 %", "Less accurate and not explainable"),
        ("Black box: neural network / TabPFN", "9.1 % / 9.2 %", "Less accurate and not explainable"),
        ("Temperature, pressure, time of day, turbulence", "11.0–11.6 %", "The neighbours' wind already holds this information"),
        ("Weight recent days more", "11.0–11.6 %", "Fewer data points, more noise"),
        ("Curved relations instead of a straight line", "10.7–11.9 %", "A straight line is good enough"),
        ("Correct with the errors just before and after the outage", "10.9 % (worse)", "Errors fade within 3 hours"),
        ("Automatic choice of reference turbines", "8.3 % (tie)", "Same accuracy; one turbine (O1) carries most information"),
    ],
    columns=["Idea", "Park energy error", "Why it didn't help"],
)


def render_tried() -> None:
    st.subheader("Things We Tried but Didn't Work Out")
    st.markdown(
        "About 50 variants, all tested on the same 28 simulated outages. "
        "**Baseline 11.0 %, our model 8.5 %** (error in park energy, lower is better). "
        "Lesson: more inputs or a more complex model added numbers, not information."
    )
    st.table(TRIED, hide_index=True, alt="Approaches we tried, their park energy error and why they didn't help")
    _chart("leaderboard", "Park energy error of the best variant per idea vs baseline and final model")


RUNS = Path(__file__).resolve().parent / "experiments" / "runs"
FINAL, BASELINE = "E12_dir8_smooth60_power", "E00_baseline"


@st.cache_data(max_entries=4)
def _test_rows(exp_id: str) -> pd.DataFrame:
    """Rows T1-T3 and "All" for the public test window (4-9 Sep, "val" in experiments/runs/<id>.json)."""
    pw = pd.DataFrame(json.loads((RUNS / f"{exp_id}.json").read_text(encoding="utf-8"))["per_window"])
    val = pw[pw["window"] == "val"].set_index("target")
    t = val.drop(index="park")
    n = t["n"]
    park = val.loc["park"]
    rows = t[["ws_rmse", "ws_mae", "p_rmse", "p_mae", "e_pred_kwh", "e_act_kwh", "e_dev_pct", "cover"]].copy()
    # "All": errors pooled over every turbine step, energy summed over the park
    rows.loc["All"] = [
        (n @ t["ws_rmse"] ** 2 / n.sum()) ** 0.5, n @ t["ws_mae"] / n.sum(),
        (n @ t["p_rmse"] ** 2 / n.sum()) ** 0.5, n @ t["p_mae"] / n.sum(),
        park["e_pred_kwh"], park["e_act_kwh"], park["e_dev_pct"], n @ t["cover"] / n.sum(),
    ]
    return rows


def _test_table(rows: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Turbine": rows.index,
        "Wind RMSE (m/s)": rows["ws_rmse"].map("{:.2f}".format),
        "Wind MAE (m/s)": rows["ws_mae"].map("{:.2f}".format),
        "Power RMSE (kW)": rows["p_rmse"].map("{:,.0f}".format),
        "Power MAE (kW)": rows["p_mae"].map("{:,.0f}".format),
        "Predicted energy (MWh)": (rows["e_pred_kwh"] / 1000).map("{:,.1f}".format),
        "Actual energy (MWh)": (rows["e_act_kwh"] / 1000).map("{:,.1f}".format),
        "Energy deviation": rows["e_dev_pct"].map("{:+.1f} %".format),
        "95 % band coverage": (100 * rows["cover"]).map("{:.1f} %".format),
    })


def render_test_results() -> None:
    """Challenge deliverable 2: accuracy on the public part of the test set (4-9 Sep 2026)."""
    st.subheader("Test set results")
    st.markdown(
        "Accuracy as defined in the challenge, on the **public part of the test set: 4–9 Sep 2026** "
        "(6 days, 864 ten-minute steps per turbine, park T known). We treat it like an outage: the model is "
        "fitted without these days and predicts them blind. The hidden part (1–3 Sep) is scored by the judges."
    )
    final, base = _test_rows(FINAL), _test_rows(BASELINE)
    a, b = final.loc["All"], base.loc["All"]

    st.markdown("##### All turbines, final model (change vs baseline)")
    with st.container(horizontal=True):
        st.metric("Wind speed RMSE", f"{a.ws_rmse:.2f} m/s", f"{a.ws_rmse - b.ws_rmse:+.2f} m/s",
                  delta_color="inverse", border=True)
        st.metric("Wind speed MAE", f"{a.ws_mae:.2f} m/s", f"{a.ws_mae - b.ws_mae:+.2f} m/s",
                  delta_color="inverse", border=True)
        st.metric("Power RMSE", f"{a.p_rmse:,.0f} kW", f"{a.p_rmse - b.p_rmse:+,.0f} kW",
                  delta_color="inverse", border=True)
        st.metric("Power MAE", f"{a.p_mae:,.0f} kW", f"{a.p_mae - b.p_mae:+,.0f} kW",
                  delta_color="inverse", border=True)
        st.metric("Summed energy deviation", f"{a.e_dev_pct:+.1f} %",
                  f"{(a.e_pred_kwh - a.e_act_kwh) / 1000:+,.1f} MWh · baseline {b.e_dev_pct:+.1f} %",
                  delta_color="off", border=True,
                  help="Predicted minus actual park energy, summed over 4-9 Sep, relative to actual.")

    st.markdown("##### Per turbine, final model")
    st.table(_test_table(final), hide_index=True, alt="Test set metrics per turbine for the final model")
    with st.expander("Baseline per turbine, for comparison", icon=":material/compare_arrows:"):
        st.table(_test_table(base), hide_index=True, alt="Test set metrics per turbine for the baseline")

    t_final = final.drop(index="All")["e_dev_pct"]
    t_base = base.drop(index="All")["e_dev_pct"]
    st.info(
        f"The baseline's park total looks better only because its turbine errors cancel "
        f"(T1 {t_base['T1']:+.1f} %, T3 {t_base['T3']:+.1f} %). Per turbine the final model is much closer: "
        f"average absolute energy deviation {t_base.abs().mean():.1f} % → {t_final.abs().mean():.1f} %. "
        "One 6-day window is a weak judge; over 28 simulated 3-day outages the park energy error is "
        "8.5 % (final) vs 11.0 % (baseline), see Methodology.",
        title="How to read the energy deviation",
        icon=":material/info:",
    )
    st.caption(
        "Power is compared with the power curve applied to the measured wind. All turbines: wind and power errors "
        f"pooled over every turbine step; energy summed over the park. Source: experiments/runs/{FINAL}.json and "
        f"{BASELINE}.json (`python -m pipeline.run <id>`)."
    )
