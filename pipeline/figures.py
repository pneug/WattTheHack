"""Plotly figures for the experiment report. One function per figure, each returns a go.Figure."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pipeline.experiments import BEST

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"

BASE = "E00_baseline"
INK, MUTED, GRID = "#1f2933", "#6b7785", "#e6e9ee"
C_BASE, C_BEST, C_ALT, C_GREY = "#8896a5", "#0b7a75", "#d9822b", "#c9ced6"
TURB = ["T1", "T2", "T3"]


def load_results() -> pd.DataFrame:
    d = pd.read_csv(EXP / "results.csv")
    d = d.sort_values("run_at").groupby("id", as_index=False).last()
    return d.set_index("id", drop=False)


def load_run(eid: str) -> dict:
    return json.loads((EXP / "runs" / f"{eid}.json").read_text(encoding="utf-8"))


def style(fig, title, subtitle, height=440):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b><br><span style='font-size:13px;color:{MUTED}'>{subtitle}</span>",
                   x=0.0, xanchor="left", y=0.95),
        height=height, template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", size=13, color=INK),
        margin=dict(l=70, r=30, t=100, b=60), legend=dict(orientation="h", y=-0.15),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def fig_leaderboard(res=None):
    res = load_results() if res is None else res
    rows = [res.loc[BASE]]
    for g in [f"E0{i}" for i in range(1, 9)]:
        sub = res[res.id.str.startswith(g + "_")]
        rows.append(sub.loc[sub.bt_abs_edev.idxmin()])
    rows.append(res.loc[BEST])
    d = pd.DataFrame(rows).sort_values("bt_abs_edev", ascending=False)
    base = res.loc[BASE, "bt_abs_edev"]
    col = [C_BEST if i == BEST else C_BASE if i == BASE else C_GREY for i in d.id]
    labels = [f"{r.id.split('_')[0]}: {r.desc}" for r in d.itertuples()]
    fig = go.Figure(go.Bar(
        y=labels, x=d.bt_abs_edev, orientation="h", marker_color=col, width=0.55,
        text=[f"{v:.1f} %" for v in d.bt_abs_edev], textposition="outside",
        customdata=np.c_[d.bt_ws_rmse, d.bt_p_rmse, d.id],
        hovertemplate="%{customdata[2]}<br>|energy dev| %{x:.2f} %<br>wind RMSE %{customdata[0]:.3f} m/s"
                      "<br>power RMSE %{customdata[1]:.0f} kW<extra></extra>"))
    fig.add_vline(x=base, line=dict(color=INK, width=1, dash="dot"),
                  annotation_text="baseline", annotation_position="top")
    best = res.loc[BEST, "bt_abs_edev"]
    fig.update_xaxes(title="Mean |energy deviation| per turbine and outage (%), lower is better",
                     range=[0, base * 1.2])
    fig.update_yaxes(automargin=True)
    return style(fig, f"Combining wind direction, smoothed neighbours and a power fix cuts the energy error from {base:.1f} % to {best:.1f} %",
                 "Best variant per round-1 idea (E01–E08) versus baseline E00 and final model, 28 simulated outages",
                 height=520)


def fig_bias(res=None):
    res = load_results() if res is None else res
    ids = [BASE, "E03_power_gauss", "E09_power_gauss_bins", "E10_dir8_lags", BEST]
    cols = [C_BASE, C_ALT, "#e9b872", "#7fa6c9", C_BEST]
    fig = go.Figure()
    for i, c in zip(ids, cols):
        fig.add_bar(name=f"{i.split('_')[0]}: {res.loc[i, 'desc']}"[:60], x=TURB,
                    y=[res.loc[i, f"bt_edev_{t}"] for t in TURB], marker_color=c,
                    hovertemplate=f"{i}<br>%{{x}}: %{{y:.1f}} %<extra></extra>")
    fig.update_layout(barmode="group")
    fig.add_hline(y=0, line=dict(color=INK, width=1))
    fig.update_yaxes(title="Mean signed energy deviation (%)")
    b = res.loc[BASE, [f"bt_edev_{t}" for t in TURB]]
    m = res.loc[BEST, [f"bt_edev_{t}" for t in TURB]]
    return style(fig, f"Plain power-curve mapping loses {abs(b.max()):.0f}–{abs(b.min()):.0f} % of energy; the final model is within {m.abs().max():.1f} %",
                 "Signed energy bias per turbine, mean over backtest outages (negative = underestimate)", height=480)


def _windows(eid, kind="bt", target="park"):
    d = pd.DataFrame(load_run(eid)["per_window"])
    return d[(d.kind == kind) & (d.target == target)]


def fig_distribution():
    fig = go.Figure()
    for eid, name, c in [(BASE, "Baseline", C_BASE), (BEST, "Final model", C_BEST)]:
        w = _windows(eid)
        fig.add_trace(go.Box(y=w.e_dev_pct, name=name, boxpoints="all", jitter=0.4, pointpos=0,
                             marker=dict(color=c, size=6, opacity=0.8), line=dict(color=c, width=1.5),
                             fillcolor="rgba(0,0,0,0)", text=w.window,
                             hovertemplate="%{text}<br>%{y:.1f} %<extra>" + name + "</extra>"))
    fig.add_hline(y=0, line=dict(color=INK, width=1, dash="dot"))
    fig.update_yaxes(title="Park energy deviation (%)")
    fig.update_layout(showlegend=False)
    mb, mx = _windows(BASE).e_dev_pct, _windows(BEST).e_dev_pct
    return style(fig, f"The final model centres near zero (median {mx.median():+.1f} %) versus {mb.median():+.1f} % for the baseline",
                 "Park-level energy deviation, one dot per simulated 3-day outage (28), hover for window", height=460)


def _cover(eid):
    d = pd.DataFrame(load_run(eid)["per_window"])
    d = d[(d.kind == "bt") & d.target.isin(TURB)]
    rows = {t: (np.average(x.cover, weights=x.n), np.average(x.width, weights=x.n))
            for t, x in d.groupby("target")}
    allc = (np.average(d.cover, weights=d.n), np.average(d.width, weights=d.n))
    return rows, allc


def fig_calibration():
    fig = go.Figure()
    for eid, name, c in [(BASE, "Baseline", C_BASE), (BEST, "Final model", C_BEST)]:
        rows, _ = _cover(eid)
        fig.add_trace(go.Scatter(
            x=TURB, y=[rows[t][0] * 100 for t in TURB], mode="markers", name=name,
            marker=dict(color=c, size=13), customdata=[rows[t][1] for t in TURB],
            hovertemplate=name + " %{x}<br>coverage %{y:.1f} %<br>mean width %{customdata:.2f} m/s<extra></extra>"))
    fig.add_hline(y=95, line=dict(color=INK, width=1, dash="dot"), annotation_text="95 % target")
    fig.update_yaxes(title="Share of true wind speeds inside the 95 % band (%)", range=[85, 100])
    (cb, wb), (cx, wx) = _cover(BASE)[1], _cover(BEST)[1]
    return style(fig, f"Uncertainty bands are honest: coverage {cb*100:.1f} % (baseline) and {cx*100:.1f} % (final) against a 95 % target",
                 f"Backtest coverage of the 95 % band per turbine; mean band width {wb:.1f} m/s (baseline) vs {wx:.1f} m/s (final)",
                 height=420)


def fig_example(window="val", target="T1"):
    s = pd.read_csv(EXP / "runs" / f"{BEST}_steps.csv", parse_dates=["timestamp"])
    s = s[(s.window == window) & (s.target == target)].sort_values("timestamp")
    fig = go.Figure()
    fig.add_scatter(x=s.timestamp, y=s.hi, line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=s.timestamp, y=s.lo, line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(11,122,117,0.15)", name="95 % band", hoverinfo="skip")
    fig.add_scatter(x=s.timestamp, y=s.actual, name="Actual", line=dict(color=INK, width=1))
    fig.add_scatter(x=s.timestamp, y=s.pred, name="Reconstructed", line=dict(color=C_BEST, width=1.5))
    fig.update_yaxes(title="Wind speed (m/s)")
    rmse = float(np.sqrt(np.mean((s.pred - s.actual) ** 2)))
    cov = float(((s.actual >= s.lo) & (s.actual <= s.hi)).mean() * 100)
    return style(fig, f"On the validation outage, {target} is tracked within {rmse:.1f} m/s RMSE and {cov:.0f} % of actuals sit in the band",
                 f"Final model, turbine {target}, validation window: actual vs reconstructed wind speed with 95 % band", height=440)


def fig_blackbox(res=None):
    res = load_results() if res is None else res
    hl = {BASE: ("Baseline OLS", C_BASE), "E08_hgb": ("Gradient boosting (black box)", C_ALT),
          "E08_ridge_refs": ("Ridge, more reference sensors", "#7fa6c9"), BEST: ("Final model (transparent)", C_BEST)}
    other = res[~res.id.isin(hl)]
    fig = go.Figure(go.Scatter(x=other.bt_p_rmse, y=other.bt_abs_edev, mode="markers", name="Other experiments",
                               marker=dict(color=C_GREY, size=8), text=other.id,
                               hovertemplate="%{text}<br>power RMSE %{x:.0f} kW<br>|energy dev| %{y:.2f} %<extra></extra>"))
    for i, (lab, c) in hl.items():
        r = res.loc[i]
        fig.add_scatter(x=[r.bt_p_rmse], y=[r.bt_abs_edev], mode="markers+text", name=lab, text=[lab],
                        textposition="bottom center" if i == "E08_hgb" else "top center",
                        marker=dict(color=c, size=13), showlegend=False,
                        hovertemplate=f"{i}<br>power RMSE %{{x:.0f}} kW<br>|energy dev| %{{y:.2f}} %<extra></extra>")
    fig.update_xaxes(title="Power RMSE (kW), lower is better")
    fig.update_yaxes(title="Mean |energy deviation| (%)")
    h, b = res.loc["E08_hgb"], res.loc[BEST]
    if b.bt_abs_edev <= h.bt_abs_edev:
        t = f"The transparent final model beats the black box on energy error ({b.bt_abs_edev:.1f} % vs {h.bt_abs_edev:.1f} %)"
    else:
        t = f"The black box wins on energy error ({h.bt_abs_edev:.1f} % vs {b.bt_abs_edev:.1f} %), at a cost in explainability"
    return style(fig, t, "All experiments: power RMSE vs energy error; highlighted are baseline, gradient boosting, ridge and final model", height=500)


def _tldr(res):
    b, x = res.loc[BASE], res.loc[BEST]
    gain = (1 - x.bt_abs_edev / b.bt_abs_edev) * 100
    h = res.loc["E08_hgb"]
    return [
        f"Energy error per turbine and outage fell from {b.bt_abs_edev:.1f} % to {x.bt_abs_edev:.1f} % ({gain:.0f} % relative) with the final model {BEST}.",
        f"The baseline underestimated energy by {abs(b.bt_edev_T1):.1f} / {abs(b.bt_edev_T2):.1f} / {abs(b.bt_edev_T3):.1f} % (T1/T2/T3); the final model's bias is {x.bt_edev_T1:+.1f} / {x.bt_edev_T2:+.1f} / {x.bt_edev_T3:+.1f} %.",
        f"Gradient boosting reaches {h.bt_abs_edev:.1f} %, the transparent final model {x.bt_abs_edev:.1f} %; 95 % band coverage is {x.bt_cover*100:.1f} % (baseline {b.bt_cover*100:.1f} %).",
    ]


def _plain(obj):
    """Plotly figure dict -> JSON-safe Python with plain lists (no base64 arrays), floats rounded."""
    if isinstance(obj, dict):
        if "bdata" in obj and "dtype" in obj:  # plotly's base64 typed array
            import base64
            arr = np.frombuffer(base64.b64decode(obj["bdata"]), dtype=obj["dtype"])
            return _plain(arr.reshape(obj["shape"]) if "shape" in obj else arr)
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.datetime64):
            return [str(pd.Timestamp(v)) for v in obj]
        return _plain(obj.tolist())
    if isinstance(obj, (float, np.floating)):
        return None if np.isnan(obj) else round(float(obj), 4)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(obj))
    return obj


def _fig_script(fig: go.Figure, div_id: str) -> str:
    """One JSON line per trace and an indented layout, so git diffs stay readable."""
    d = _plain(fig.to_plotly_json())
    traces = ",\n".join(json.dumps(t, ensure_ascii=False) for t in d["data"])
    layout = json.dumps(d["layout"], indent=1, ensure_ascii=False)
    return "\n".join([
        f'<div class="fig" id="{div_id}"></div>',
        "<script>",
        f'Plotly.newPlot("{div_id}", [',
        traces,
        "],",
        layout + ",",
        '{"responsive": true});',
        "</script>",
    ])


def build_report(path="reports/experiments.html"):
    from plotly.offline import get_plotlyjs_version

    res = load_results()
    figs = [fig_leaderboard(res), fig_bias(res), fig_distribution(), fig_calibration(), fig_example(), fig_blackbox(res)]
    for f in figs:
        print(f.layout.title.text.split("<br>")[0].replace("<b>", "").replace("</b>", ""))
    css = [
        f"body {{font-family: 'Segoe UI', Helvetica, Arial, sans-serif; color: {INK}; max-width: 980px; "
        "margin: 0 auto; padding: 24px 16px; background: #fff}",
        f"h1 {{margin: 0 0 4px}} .sub {{color: {MUTED}; margin: 0 0 16px}}",
        f".tldr {{background: #f4f7f8; border-left: 4px solid {C_BEST}; padding: 8px 20px; margin-bottom: 24px}}",
        ".tldr li {margin: 6px 0} .fig {margin-bottom: 28px}",
    ]
    html = "\n".join([
        "<!doctype html>", '<html lang="en">', "<head>", '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Experiment results</title>",
        f'<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js" charset="utf-8"></script>',
        "<style>", *css, "</style>", "</head>", "<body>",
        "<h1>Reconstructing turbine wind speed: experiment results</h1>",
        '<p class="sub">About 40 configurations tested on 28 simulated 3-day outages plus the 4–9 Sep validation window.</p>',
        '<div class="tldr"><b>TL;DR</b>', "<ul>", *[f"  <li>{t}</li>" for t in _tldr(res)], "</ul>", "</div>",
        *[_fig_script(f, f"fig{k + 1}") for k, f in enumerate(figs)],
        "</body>", "</html>", "",
    ])
    out = Path(path)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build_report())
