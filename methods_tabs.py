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


# The app focuses on the final model: our own baseline stays in the legend (click to show), not in the title.
_HIDE_TRACES = {"distribution": "Baseline", "blackbox": "Baseline OLS"}
_SUB = "<br><span style='font-size:13px;color:#6b7785'>{}</span>"
_TITLES = {
    "distribution": "<b>Park energy error per simulated outage: centred near zero (median -0.6 %)</b>"
                    + _SUB.format("One dot per simulated 3-day outage (28), hover for the window"),
    "blackbox": "<b>The transparent final model beats the black box on park energy error (8.5 % vs 9.7 %)</b>"
                + _SUB.format("All experiments: power RMSE vs park energy error"),
}


@st.cache_resource(max_entries=16)
def _figure(name: str) -> go.Figure | None:
    path = FIGURES / f"{name}.json"
    if not path.exists():
        return None
    fig = go.Figure(json.loads(path.read_text(encoding="utf-8")))
    if name in _HIDE_TRACES:
        fig.update_traces(visible="legendonly", selector={"name": _HIDE_TRACES[name]})
    if name in _TITLES:
        fig.update_layout(title_text=_TITLES[name])
    return fig


def _chart(name: str, alt: str) -> None:
    fig = _figure(name)
    if fig is None:
        st.caption(f"Figure `{name}` missing: run `python -m pipeline.figures`.")
        return
    st.plotly_chart(fig, theme=None, key=f"methods_{name}", alt=alt)


def render_methodology_story() -> None:
    """How the model works, what makes it accurate and the four key insights (pitch storyline)."""
    st.subheader("Methodology")
    st.markdown("When park T is down, we rebuild its wind from the neighbouring parks H and O. "
                "Four choices make our model accurate:")

    cards = [
        (":material/blur_on:", "Smooth out gusts",
         "We use the neighbours' **1-hour average** wind, not 10-minute values.",
         "Gusts at H and O don't reach T at the same moment."),
        (":material/explore:", "Respect the wind direction",
         "Each of **8 directions** gets its own share of H's wind.",
         "T gets 84 % (north) to 93 % (south-west) of H's wind."),
        (":material/bolt:", "Average the power",
         "We average the power over **all likely winds**, not the power of one best guess.",
         "Removes an 8.5 % underestimate of energy."),
        (":material/straighten:", "Show the uncertainty",
         "A **95 % band** every 10 minutes and a **95 % range** for the lost energy.",
         "The band held the true wind 94.8 % of the time."),
    ]
    for column, (icon, title, text, why) in zip(st.columns(len(cards)), cards):
        with column.container(border=True, height="stretch"):
            st.markdown(f"#### {icon}")
            st.markdown(f"**{title}**")
            st.markdown(text)
            st.caption(why)

    st.markdown("#### The formula in plain words")
    with st.container(border=True):
        st.markdown(
            "**T's wind** = :blue-badge[offset for this wind direction] "
            "+ :blue-badge[share for this direction] × :green-badge[park H, 1-hour average] "
            "+ :orange-badge[fixed share] × :green-badge[park O, 1-hour average]"
        )
        st.markdown(
            "**Lost energy** = sum over the outage of :violet-badge[average power over the likely winds]"
        )
        st.caption("One formula per turbine, fitted on all days without outage. Every number in it can be "
                   "printed and explained.")
    with st.expander("As an equation", icon=":material/functions:"):
        st.latex(r"v_T = a_{\text{dir}} + b_{\text{dir}} \cdot \overline{v}_H^{\,1h} + c \cdot \overline{v}_O^{\,1h}")

    st.markdown("#### Four things we learned")

    st.markdown("**1 · Test on many outages, not one lucky week.** We hid 3 days of park T at a time and "
                "predicted them blind, 28 times. Average error in park energy: **8.5 %**.")
    _chart("distribution", "Park energy deviation per simulated outage for the final model")

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
            "back to a simpler formula without direction, so there are no gaps.\n"
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
    with st.container(border=True):
        st.markdown("**TL;DR**")
        st.markdown(
            "1. **Black boxes lost:** gradient boosting, neural network and TabPFN were all less accurate "
            "than our transparent formula.\n"
            "2. **More sensors added nothing:** temperature, pressure or turbulence carry no information "
            "beyond the neighbours' wind.\n"
            "3. **The rest is weather luck:** not even the hours right before and after an outage help, "
            "because errors fade within 3 hours."
        )
    st.caption("About 50 variants, all tested on the same 28 simulated outages. Our model: 8.5 % average error "
               "in park energy (lower is better).")
    st.table(TRIED, hide_index=True, alt="Approaches we tried, their park energy error and why they didn't help")
    _chart("leaderboard", "Park energy error of the best variant per idea vs baseline and final model")


RUNS = Path(__file__).resolve().parent / "experiments" / "runs"
FINAL = "E12_dir8_smooth60_power"


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
        "Energy deviation (MWh)": ((rows["e_pred_kwh"] - rows["e_act_kwh"]) / 1000).map("{:+,.1f}".format),
        "Energy deviation (%)": rows["e_dev_pct"].map("{:+.1f} %".format),
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
    final = _test_rows(FINAL)
    a = final.loc["All"]

    st.markdown("##### All turbines")
    with st.container(horizontal=True):
        st.metric("Wind speed RMSE", f"{a.ws_rmse:.2f} m/s", border=True)
        st.metric("Wind speed MAE", f"{a.ws_mae:.2f} m/s", border=True)
        st.metric("Power RMSE", f"{a.p_rmse:,.0f} kW", border=True)
        st.metric("Power MAE", f"{a.p_mae:,.0f} kW", border=True)
        st.metric("Summed energy deviation", f"{(a.e_pred_kwh - a.e_act_kwh) / 1000:+,.1f} MWh", border=True,
                  help=f"Predicted minus actual park energy over 4–9 Sep: "
                       f"{a.e_pred_kwh / 1000:,.1f} vs {a.e_act_kwh / 1000:,.1f} MWh.")
        st.metric("Summed energy deviation (relative)", f"{a.e_dev_pct:+.1f} %", border=True,
                  help="Summed energy deviation relative to the actual park energy.")

    st.markdown("##### Per turbine")
    st.table(_test_table(final), hide_index=True, alt="Test set metrics per turbine")
    st.caption(
        "Power is compared with the power curve applied to the measured wind. All turbines: wind and power errors "
        "pooled over every turbine step; energy summed over the park. One 6-day window is a small sample: over 28 "
        f"simulated 3-day outages the average park energy error is 8.5 % (see Methodology). "
        f"Source: experiments/runs/{FINAL}.json."
    )
