"""Content of the "Methodology & Validation" and "Things We Tried but Didn't Work Out" tabs.

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
    """The final model, how we tested it and the four key insights (pitch storyline)."""
    st.subheader("Methodology & Validation")
    st.markdown(
        "We rebuild park T's wind speed during an outage from the neighbouring parks H and O, "
        "turn it into power with the turbine's power curve and add up the lost energy. "
        "All numbers below come from **28 simulated 3-day outages** where the true values are known."
    )

    with st.container(horizontal=True):
        st.metric("Park energy error", "8.5 %", "-2.5 pts vs baseline (11.0 %)", delta_color="inverse", border=True,
                  help="Mean absolute deviation of the park's total energy per simulated outage (what the judges score).")
        st.metric("Energy bias", "+1.4 %", "baseline -8.5 %", delta_color="off", border=True,
                  help="Average signed deviation: is the model systematically too high or too low?")
        st.metric("Wind speed RMSE", "1.20 m/s", "-0.06 m/s vs baseline", delta_color="inverse", border=True)
        st.metric("95 % band coverage", "94.8 %", "target 95 %", delta_color="off", border=True,
                  help="Share of true wind speeds inside the shown 95 % band.")

    st.markdown("#### The model in one sentence")
    st.latex(r"v_T = a_{\text{dir}} + b_{\text{dir}} \cdot \overline{v}_H^{\,1h} + c \cdot \overline{v}_O^{\,1h}")
    st.markdown(
        "- One formula per turbine; $\\overline{v}^{\\,1h}$ is the 1-hour average wind of each neighbouring park.\n"
        "- The offset $a$ and the H factor $b$ take one value for each of 8 wind directions.\n"
        "- Every coefficient can be printed and explained. The outage itself is never used for fitting."
    )

    st.markdown("#### 1. We tested on 28 simulated outages, not one lucky week")
    st.markdown(
        "We hid 3 days of park T at a time, predicted them blind and compared with the truth, 28 times. "
        "The official validation week alone was misleading: the baseline's park total was off by only +0.1 %, "
        "but only because T1 (+19 %) and T3 (-8 %) cancelled out. A repeat with an outage starting every day "
        "(83 outages) confirmed the result: 10.8 % (baseline) → 8.3 % (final)."
    )
    _chart("distribution", "Park energy deviation per simulated outage, baseline vs final model")

    st.markdown("#### 2. Average the power, not the wind")
    st.markdown(
        "Between 3 and 11 m/s, where 97 % of the energy is made, the power curve bends upward: a gust above the "
        "predicted wind adds more power than an equal lull below it removes. Feeding the single best guess into the "
        "curve therefore underestimated park energy by **8.5 %**. We average the power over the plausible wind range "
        "instead; the bias drops to **+1.4 %**. This one change took the park error from 11.0 % to 9.0 %."
    )
    _chart("power_evidence", "Mean power by predicted wind speed: plain curve, averaged curve and actual")

    st.markdown("#### 3. What remains is weather luck, so we show a range")
    st.markdown(
        "The park energy error of an outage is simply its **3-day average wind error**. Wind errors fade within "
        "about 3 hours and the three turbines are wrong together, so three days leave a random offset worth "
        "about ±11 % of energy per outage. The neighbours carry no information to remove it, so we report the "
        "lost energy with a 95 % range built from the errors of 83 simulated outages (one starting every day)."
    )
    with st.container(horizontal=True):
        st.metric("Lost energy, judged outage 1–3 Sep", "322 MWh", border=True)
        st.metric("95 % range", "262–387 MWh", border=True,
                  help="From the energy errors of 83 simulated outages (one starting every day).")
        st.metric("Range check", "93 %", "target 95 %", delta_color="off", border=True,
                  help="Built from the other outages only, the range contained the true energy in 93 % of 83 daily outages.")
    _chart("bias_noise", "Park energy error split into bias and noise, baseline vs final model")

    st.markdown("#### 4. The transparent formula beat the black box")
    st.markdown(
        "Gradient boosting with all inputs reached 9.7 % park energy error and its 95 % band covered only 89 %. "
        "The transparent formula reached 8.5 % with every number explainable."
    )
    _chart("blackbox", "Power RMSE vs park energy error for all experiments, black box highlighted")

    with st.expander("More details: direction, smoothing, uncertainty bands, safety rules", icon=":material/info:"):
        st.markdown(
            "- **Wind direction:** T gets 84 % (north / north-east) to 93 % (east / south-west) of H's wind, "
            "because of terrain and park layout.\n"
            "- **Gusts are noise:** 10-minute values of the neighbours agree less with T (correlation 0.80) than "
            "3-hour averages (0.89), so we use 1-hour averages.\n"
            "- **Uncertainty bands:** the width of the 10-minute band grows with wind speed; it covers 94.8 % of true values.\n"
            "- **The judged outage is favourable:** 1–3 Sep is 89 % south-west wind and windy, the conditions in which "
            "the model tracks T best (6.3 % error in windy outages vs 10.7 % in calm ones).\n"
            "- **Safety rules in code:** park T is never used during the outage; if an extra input is missing at a step, "
            "that step falls back to the plain baseline formula, so the prediction never has gaps."
        )
        _chart("direction", "Share of park H wind reaching park T by wind direction")
        _chart("smoothing", "Effect of averaging the neighbours over 30 to 120 minutes")
        _chart("calibration", "Coverage of the 95 % wind-speed band per turbine")


TRIED = pd.DataFrame(
    [
        ("Weight recent days more", "The relation drifts from July to September",
         "11.0–11.6 %", "The drift is not local; less data only adds noise"),
        ("Stretch the prediction to T's natural variability (industry standard)", "Plain regression flattens extremes",
         "Bias -2 %, but wind error 1.26 → 1.33 m/s", "Adds noise at every step; averaging the power removes the bias without that cost"),
        ("Curved relations (log, square root, piecewise)", "Calm and strong wind might behave differently",
         "10.7–11.9 %", "A straight line is good enough"),
        ("Time of day, temperature, air pressure, turbulence", "Atmospheric stability changes wind profiles",
         "11.0–11.6 %", "No information beyond the neighbours' wind speed"),
        ("Black box (gradient boosting, all inputs)", "Captures any pattern automatically",
         "9.7 % (final: 8.5 %)", "The transparent model was better and is explainable"),
        ("Correct with the error just before and after the outage", "Tracks the local weather situation",
         "7.8 → 10.9 % (worse)", "Errors fade within 3 hours; the edges say nothing about the middle"),
        ("Extra park-O factor per wind direction; learn from hours with the same direction", "Lean on the upwind park",
         "8.5–8.9 % (no gain)", "Direction is already captured; extra terms add numbers, not information"),
        ("Automatic choice of reference turbines", "Fewer inputs, bonus criterion",
         "8.3 % vs 8.3 % (tie)", "Same accuracy with 2–3 references: O1 carries most of the information"),
        ("Wind direction for the park total", "Strong effect per turbine (13.8 → 11.8 %)",
         "11.0 → 10.9 %", "Fixes differences between turbines, which cancel in the park sum; kept for wind and power accuracy"),
    ],
    columns=["Idea", "Why it looked promising", "Park energy error (baseline 11.0 %)", "Our reading"],
)


def render_tried() -> None:
    st.subheader("Things We Tried but Didn't Work Out")
    st.markdown(
        "About 50 variants, each tested on the same 28 simulated outages. Most \"obvious\" additions added numbers "
        "to the model without adding information. Two things mattered: the right inputs (1-hour averages plus wind "
        "direction) and converting wind to power correctly."
    )
    st.table(TRIED, hide_index=True, alt="Approaches we tried, why, the result and our interpretation")
    _chart("leaderboard", "Park energy error of the best variant per idea vs baseline and final model")
