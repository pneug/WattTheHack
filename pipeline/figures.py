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
        rows.append(sub.loc[sub.bt_abs_edev_park.idxmin()])
    rows.append(res.loc[BEST])
    d = pd.DataFrame(rows).sort_values("bt_abs_edev_park", ascending=False)
    base = res.loc[BASE, "bt_abs_edev_park"]
    col = [C_BEST if i == BEST else C_BASE if i == BASE else C_GREY for i in d.id]
    labels = [f"{r.id.split('_')[0]}: {r.desc}" for r in d.itertuples()]
    fig = go.Figure(go.Bar(
        y=labels, x=d.bt_abs_edev_park, orientation="h", marker_color=col, width=0.55,
        text=[f"{v:.1f} %" for v in d.bt_abs_edev_park], textposition="outside",
        customdata=np.c_[d.bt_ws_rmse, d.bt_p_rmse, d.id],
        hovertemplate="%{customdata[2]}<br>park |energy dev| %{x:.2f} %<br>wind RMSE %{customdata[0]:.3f} m/s"
                      "<br>power RMSE %{customdata[1]:.0f} kW<extra></extra>"))
    fig.add_vline(x=base, line=dict(color=INK, width=1, dash="dot"),
                  annotation_text="baseline", annotation_position="top")
    best = res.loc[BEST, "bt_abs_edev_park"]
    top = d[~d.id.isin([BASE, BEST])].iloc[-1]
    fig.update_xaxes(title="Mean |park energy deviation| per outage (%), lower is better",
                     range=[0, base * 1.2])
    fig.update_yaxes(automargin=True)
    return style(fig, f"Park energy error {base:.1f} % → {best:.1f} %; the best single change is {top.desc} ({top.bt_abs_edev_park:.1f} %)",
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
    fig = go.Figure(go.Scatter(x=other.bt_p_rmse, y=other.bt_abs_edev_park, mode="markers", name="Other experiments",
                               marker=dict(color=C_GREY, size=8), text=other.id,
                               hovertemplate="%{text}<br>power RMSE %{x:.0f} kW<br>park |energy dev| %{y:.2f} %<extra></extra>"))
    for i, (lab, c) in hl.items():
        r = res.loc[i]
        fig.add_scatter(x=[r.bt_p_rmse], y=[r.bt_abs_edev_park], mode="markers+text", name=lab, text=[lab],
                        textposition="bottom center" if i == "E08_hgb" else "top center",
                        marker=dict(color=c, size=13), showlegend=False,
                        hovertemplate=f"{i}<br>power RMSE %{{x:.0f}} kW<br>park |energy dev| %{{y:.2f}} %<extra></extra>")
    fig.update_xaxes(title="Power RMSE (kW), lower is better")
    fig.update_yaxes(title="Mean |park energy deviation| (%)")
    h, b = res.loc["E08_hgb"], res.loc[BEST]
    if b.bt_abs_edev_park <= h.bt_abs_edev_park:
        t = f"The transparent final model beats the black box on park energy error ({b.bt_abs_edev_park:.1f} % vs {h.bt_abs_edev_park:.1f} %)"
    else:
        t = f"The black box wins on park energy error ({h.bt_abs_edev_park:.1f} % vs {b.bt_abs_edev_park:.1f} %), at a cost in explainability"
    return style(fig, t, "All experiments: power RMSE vs energy error; highlighted are baseline, gradient boosting, ridge and final model", height=500)


# ---------------------------------------------------------------- judged metrics: final vs baseline
# The judges score energy as the absolute deviation of the PARK total (the whole park is out, insurance pays the total).

# (label, summary key, per-window column, rows: "turb" or "park", unit, number format)
JUDGED = [
    ("Wind speed RMSE", "bt_ws_rmse", "ws_rmse", "turb", "m/s", ".2f"),
    ("Wind speed MAE", "bt_ws_mae", "ws_mae", "turb", "m/s", ".2f"),
    ("Power RMSE", "bt_p_rmse", "p_rmse", "turb", "kW", ".0f"),
    ("Power MAE", "bt_p_mae", "p_mae", "turb", "kW", ".0f"),
    ("Energy deviation, park total (judged)", "bt_abs_edev_park", "e_dev_pct", "park", "%", ".1f"),
    ("Energy deviation, per turbine (not judged)", "bt_abs_edev", "e_dev_pct", "turb", "%", ".1f"),
]


def _paired(a=BASE, b=None):
    """Backtest windows of two runs side by side (suffixes _a, _b)."""
    b = BEST if b is None else b
    pa, pb = pd.DataFrame(load_run(a)["per_window"]), pd.DataFrame(load_run(b)["per_window"])
    m = pa.merge(pb, on=["window", "target", "kind"], suffixes=("_a", "_b"))
    return m[m.kind == "bt"]


def _win_share(m, col, rows):
    d = m[m.target == "park"] if rows == "park" else m[m.target.isin(TURB)]
    a, b = d[f"{col}_a"], d[f"{col}_b"]
    if col == "e_dev_pct":
        a, b = a.abs(), b.abs()
    return float((b < a).mean()), len(d)


def _park_stats(eid):
    """Park energy error over the backtest outages: bias (mean signed), noise (std), judged mean |error|."""
    e = _windows(eid).e_dev_pct
    return {"bias": float(e.mean()), "noise": float(e.std()), "mae": float(e.abs().mean())}


def fig_scorecard(res=None):
    res = load_results() if res is None else res
    a, b = res.loc[BASE], res.loc[BEST]
    labels, chg, text = [], [], []
    for lab, key, _, _, unit, f in JUDGED:
        labels.append(lab)
        chg.append((b[key] / a[key] - 1) * 100)
        text.append(f"{a[key]:{f}} → {b[key]:{f}} {unit} ({chg[-1]:+.0f} %)")
    fig = go.Figure(go.Bar(y=labels, x=chg, orientation="h", width=0.55, text=text, textposition="outside",
                           marker_color=[C_BEST if c < 0 else C_ALT for c in chg],
                           hovertemplate="%{y}<br>%{text}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=INK, width=1))
    fig.update_xaxes(title="Change versus baseline (%), negative = better", range=[min(chg) * 1.9, 8])
    fig.update_yaxes(autorange="reversed", automargin=True)
    small = chg[:4]
    return style(fig, f"Better on every judged metric: park energy error falls by {-chg[4]:.0f} %, the four tracking metrics by {-max(small):.0f}–{-min(small):.0f} %",
                 "Final model vs baseline, pooled over 28 simulated 3-day outages (labels: baseline → final)", height=440)


def fig_winrate():
    m = _paired()
    labels, share, text = [], [], []
    for lab, _, col, rows, _, _ in JUDGED:
        s, n = _win_share(m, col, rows)
        labels.append(lab)
        share.append(s * 100)
        text.append(f"{s * 100:.0f} % ({round(s * n)} of {n})")
    fig = go.Figure(go.Bar(y=labels, x=share, orientation="h", width=0.55, text=text, textposition="outside",
                           marker_color=[C_BEST if s >= 75 else "#7fbfba" for s in share],
                           hovertemplate="%{y}<br>final better in %{text}<extra></extra>"))
    fig.add_vline(x=50, line=dict(color=INK, width=1, dash="dot"), annotation_text="coin flip", annotation_position="top")
    fig.update_xaxes(title="Share of simulated outages where the final model beats the baseline (%)", range=[0, 112])
    fig.update_yaxes(autorange="reversed", automargin=True)
    return style(fig, f"Wind and power RMSE improve in {min(share[:3]):.0f}–{max(share[:3]):.0f} % of outages; power MAE and park energy only in "
                      f"{min(share[3:5]):.0f}–{max(share[3:5]):.0f} %",
                 "Per outage and turbine (park rows: per outage); a reliable gain is far right of the 50 % line", height=440)


def fig_bias_noise():
    models = [(BASE, "Baseline"), ("E10_dir8_smooth60", "+ direction, smoothing"),
              ("E09_power_gauss_bins", "+ power fix only"), (BEST, "Final model")]
    st = {eid: _park_stats(eid) for eid, _ in models}
    names = [n for _, n in models]
    fig = go.Figure()
    for key, name, c in [("bias", "Bias (mean signed error)", C_ALT), ("noise", "Noise (std across outages)", C_GREY),
                         ("mae", "Mean |error| (what is judged)", C_BEST)]:
        y = [st[eid][key] for eid, _ in models]
        fig.add_bar(name=name, x=names, y=y, marker_color=c, text=[f"{v:+.1f}" if key == "bias" else f"{v:.1f}" for v in y],
                    textposition="outside", hovertemplate="%{x}<br>" + name + ": %{y:.1f} %<extra></extra>")
    fig.update_layout(barmode="group")
    fig.add_hline(y=0, line=dict(color=INK, width=1))
    fig.update_yaxes(title="Park energy error over 28 outages (%)")
    b, x = st[BASE], st[BEST]
    return style(fig, f"The park error fell because the bias went away ({b['bias']:+.1f} → {x['bias']:+.1f} %); the noise stayed "
                      f"(±{b['noise']:.0f} → ±{x['noise']:.0f} %)",
                 "Park energy deviation per simulated outage, split into bias and noise; only the power fix moves the bias", height=460)


def _bias_noise_text():
    b, x = _park_stats(BASE), _park_stats(BEST)
    se = x["noise"] / np.sqrt(len(_windows(BEST)))
    return (f"Each outage's park energy error = <b>bias</b> (the same direction every time, caused by the method) + <b>noise</b> "
            f"(the weather luck of that window, sign changes from outage to outage). With absolute scoring the bias is added to every "
            f"outage, so removing it lowers the expected error whatever the hidden window's weather. The baseline was {b['bias']:+.1f} % "
            f"on average (it under-predicted almost every outage); the final model is {x['bias']:+.1f} %, which is within the "
            f"±{2 * se:.0f} % uncertainty of that average, so we do not chase it further.")


def _weighting_text(res):
    """Numbers behind the 'how to weight the metrics' bullets."""
    rho = res[[k for _, k, *_ in JUDGED]].rank().corr()
    m = _paired()
    win_rmse = m[m.target.isin(TURB)].groupby("window").ws_rmse_b.mean()
    park = m[m.target == "park"].set_index("window").e_dev_pct_b.abs()
    r_win = float(np.corrcoef(win_rmse.loc[park.index], park)[0, 1])
    x = _park_stats(BEST)
    e10, d8 = res.loc["E10_dir8_smooth60"], res.loc["E06_dir8"]
    v = pd.DataFrame(load_run(BASE)["per_window"])
    v = v[v.kind == "val"].set_index("target")["e_dev_pct"]
    return [
        f"<b>Four of the five metrics measure the same thing.</b> Across all {len(res)} experiments, the rank correlation of wind RMSE with wind MAE is "
        f"{rho.loc['bt_ws_rmse', 'bt_ws_mae']:.2f}, with power MAE {rho.loc['bt_ws_rmse', 'bt_p_mae']:.2f} and with power RMSE {rho.loc['bt_ws_rmse', 'bt_p_rmse']:.2f}. "
        "A better wind reconstruction moves all four; the gains are small (4–7 %) and, except for power MAE, steady across outages.",
        f"<b>The park energy deviation is the odd one out: the largest gain and the noisiest.</b> Per outage it spreads by ±{x['noise']:.0f} % (std) "
        f"around a bias of {x['bias']:+.1f} %, and it is unrelated to how well the wind was tracked in that outage (r = {r_win:+.2f}). "
        "On a single 3-day hidden window luck dominates; what we control is the bias.",
        f"<b>In the park total, turbine errors cancel.</b> On 4–9 Sep the baseline park total was {v['park']:+.1f} % while T1 was {v['T1']:+.1f} %. "
        f"Wind direction cuts the per-turbine energy error ({res.loc[BASE, 'bt_abs_edev']:.1f} → {d8.bt_abs_edev:.1f} %) but hardly the park total "
        f"({res.loc[BASE, 'bt_abs_edev_park']:.1f} → {d8.bt_abs_edev_park:.1f} %): it fixes how energy is split between turbines, which the park sum forgives. "
        "It still pays off on the four tracking metrics.",
        f"<b>Power MAE pulls slightly against E[P(v)].</b> MAE rewards the median, E[P(v)] targets the mean: adding it moves power MAE "
        f"{e10.bt_p_mae:.0f} → {res.loc[BEST, 'bt_p_mae']:.0f} kW, while power RMSE ({e10.bt_p_rmse:.0f} → {res.loc[BEST, 'bt_p_rmse']:.0f} kW) "
        f"and park energy ({e10.bt_abs_edev_park:.1f} → {res.loc[BEST, 'bt_abs_edev_park']:.1f} %) improve. One metric slightly worse, two clearly better.",
        "<b>How we weight them:</b> (1) an unbiased park energy sum first: the judges score the park total, it is the insurance number, and its bias "
        "is the only part of its error we can reduce; (2) the four tracking metrics as one group, where direction and smoothing pay off; "
        "(3) never tune on a single window (4–9 Sep or the hidden one), always on the 28-outage backtest.",
    ]


# ---------------------------------------------------------------- the three changes in the final model

SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def fig_direction():
    from pipeline import stages
    from pipeline.core import load_data

    d = load_data()
    x = stages.FEATURES["direction"](d)
    ang = np.degrees(np.arctan2(x.dir_sin, x.dir_cos)) % 360
    sec = (ang + 22.5) // 45 % 8
    h = stages.f_park_means(d)["H"]
    t = d["wind_speed"][TURB].mean(axis=1)
    ok = sec.notna() & h.notna() & t.notna() & (h > 3)  # above cut-in, where energy is made
    g = pd.DataFrame({"s": sec[ok], "T": t[ok], "H": h[ok]}).groupby("s")
    ratio, share = g["T"].sum() / g["H"].sum(), g.size() / ok.sum() * 100
    lab = [SECTORS[int(s)] for s in ratio.index]
    fig = go.Figure(go.Bar(x=lab, y=ratio.values, width=0.6, marker_color=C_BEST,
                           text=[f"{v:.2f}" for v in ratio.values], textposition="outside",
                           customdata=share.values,
                           hovertemplate="wind from %{x}<br>T / H = %{y:.3f}<br>%{customdata:.0f} % of the time<extra></extra>"))
    fig.add_hline(y=float(t[ok].sum() / h[ok].sum()), line=dict(color=INK, width=1, dash="dot"),
                  annotation_text="all directions", annotation_position="top left")
    fig.update_yaxes(title="Park T wind / park H wind", range=[0.75, 1.0])
    fig.update_xaxes(title="Wind direction at park H (8 sectors)")
    return style(fig, f"Park T gets {ratio.min():.0%} to {ratio.max():.0%} of park H's wind, depending on where the wind comes from",
                 f"Ratio of mean wind speed T / H per direction sector (H above 3 m/s); most wind comes from SW and W "
                 f"({share.iloc[5]:.0f} % and {share.iloc[6]:.0f} % of the time)", height=420)


def fig_power_evidence():
    from preprocessing import load_power_curve, wind_to_power

    s = pd.read_csv(EXP / "runs" / f"{BEST}_steps.csv")
    s = s[s.window.str.startswith("bt")].dropna(subset=["pred", "actual"])
    curve = load_power_curve()
    s["p_act"], s["p_plain"] = wind_to_power(s.actual, curve), wind_to_power(s.pred, curve)
    s["bin"] = np.floor(s.pred) + 0.5
    g = s.groupby("bin").agg(p_act=("p_act", "mean"), p_plain=("p_plain", "mean"), p_exp=("kw", "mean"), n=("kw", "size"))
    g = g[g.n >= 30]
    fig = go.Figure()
    for col, name, st in [("p_plain", "Plain curve P(v̂)", dict(color=C_BASE, width=2, dash="dash")),
                          ("p_exp", "Averaged over uncertainty E[P(v)]", dict(color=C_BEST, width=2.5))]:
        fig.add_scatter(x=g.index, y=g[col], mode="lines", name=name, line=st,
                        hovertemplate="predicted %{x:.1f} m/s<br>%{y:,.0f} kW<extra>" + name + "</extra>")
    fig.add_scatter(x=g.index, y=g.p_act, mode="markers", name="Actual (mean)", marker=dict(color=INK, size=9),
                    customdata=g.n, hovertemplate="predicted %{x:.1f} m/s<br>actual %{y:,.0f} kW<br>%{customdata} steps<extra></extra>")
    fig.update_xaxes(title="Predicted wind speed (1 m/s bins)")
    fig.update_yaxes(title="Mean power per 10 min (kW)")
    k = ((g.p_act - g.p_plain).abs() * g.n).idxmax()  # biggest gap weighted by how often it occurs
    r = g.loc[k]
    return style(fig, f"Where most energy is made, the plain curve gives too little power; averaging over the uncertainty is much closer "
                      f"({k - 0.5:.0f}–{k + 0.5:.0f} m/s: {r.p_plain:,.0f} vs {r.p_exp:,.0f} kW, actual {r.p_act:,.0f} kW)",
                 "Backtest steps grouped by predicted wind speed: mean actual power vs the two ways of converting the prediction",
                 height=460)


def fig_smoothing(res=None):
    res = load_results() if res is None else res
    ids = [(10, BASE), (30, "E02_smooth30min"), (60, "E02_smooth60min"), (120, "E02_smooth120min")]
    a = res.loc[BASE]
    fig = go.Figure()
    for key, name, c in [("bt_ws_rmse", "Wind speed RMSE", C_BEST), ("bt_abs_edev_park", "Park energy deviation", C_ALT)]:
        y = [(res.loc[i, key] / a[key] - 1) * 100 for _, i in ids]
        fig.add_scatter(x=[m for m, _ in ids], y=y, mode="lines+markers", name=name,
                        line=dict(color=c, width=2), marker=dict(size=9),
                        hovertemplate="%{x} min<br>%{y:+.1f} % vs raw 10-min<extra>" + name + "</extra>")
    fig.add_hline(y=0, line=dict(color=INK, width=1, dash="dot"))
    fig.update_xaxes(title="Averaging window of the neighbours' wind (min; 10 = raw)", type="log",
                     tickvals=[10, 30, 60, 120])
    fig.update_yaxes(title="Change versus raw 10-min input (%)")
    r60 = (res.loc["E02_smooth60min", "bt_ws_rmse"] / a.bt_ws_rmse - 1) * 100
    e60 = (res.loc["E02_smooth60min", "bt_abs_edev_park"] / a.bt_abs_edev_park - 1) * 100
    return style(fig, f"Averaging the neighbours lowers wind RMSE ({r60:+.1f} % at 60 min) but not the park energy error ({e60:+.1f} %)",
                 "Smoothing alone (no direction, no power fix), backtest change versus the raw 10-min baseline", height=420)


def _cards(res):
    """The three changes in the final model: (title, assumption, result, figure)."""
    r = res
    b = r.loc[BASE]
    return [
        ("1. Power averaged over the uncertainty, E[P(v)]",
         "The power curve is convex at 3–8 m/s, where most energy is made: a single best-guess wind speed gives less power than the average over the plausible "
         "speeds (Jensen's inequality). The spread grows with wind speed, so it is estimated per wind-speed bin.",
         f"The main lever for the judged park total: alone it cuts the park energy error {b.bt_abs_edev_park:.1f} → {r.loc['E09_power_gauss_bins', 'bt_abs_edev_park']:.1f} %, "
         f"on top of direction + smoothing {r.loc['E10_dir8_smooth60', 'bt_abs_edev_park']:.1f} → {r.loc[BEST, 'bt_abs_edev_park']:.1f} %. "
         f"Energy bias per turbine {b.bt_edev_T1:+.1f} / {b.bt_edev_T2:+.1f} / {b.bt_edev_T3:+.1f} % → {r.loc[BEST, 'bt_edev_T1']:+.1f} / "
         f"{r.loc[BEST, 'bt_edev_T2']:+.1f} / {r.loc[BEST, 'bt_edev_T3']:+.1f} %.",
         fig_power_evidence()),
        ("2. Wind direction (8 sectors)",
         "Hills and wakes shelter park T differently depending on where the wind comes from, so one fixed T/H ratio is wrong for some directions.",
         f"Alone: wind RMSE {b.bt_ws_rmse:.3f} → {r.loc['E06_dir8', 'bt_ws_rmse']:.3f} m/s, power RMSE {b.bt_p_rmse:.0f} → {r.loc['E06_dir8', 'bt_p_rmse']:.0f} kW, "
         f"per-turbine energy error {b.bt_abs_edev:.1f} → {r.loc['E06_dir8', 'bt_abs_edev']:.1f} %. The park total barely moves "
         f"({b.bt_abs_edev_park:.1f} → {r.loc['E06_dir8', 'bt_abs_edev_park']:.1f} %), because it mostly corrects how energy is split between turbines. "
         "Explained as: one offset and one H factor per compass sector.",
         fig_direction()),
        ("3. Neighbours averaged over 60 min",
         "10-min gusts at H and O are local and not shared with T (correlation rises from 0.80 at 10 min to 0.89 at 3 h), so averaging removes noise from the inputs.",
         f"Wind RMSE {r.loc['E06_dir8', 'bt_ws_rmse']:.3f} → {r.loc['E10_dir8_smooth60', 'bt_ws_rmse']:.3f} m/s on top of direction, power RMSE "
         f"{r.loc['E06_dir8', 'bt_p_rmse']:.0f} → {r.loc['E10_dir8_smooth60', 'bt_p_rmse']:.0f} kW. Same accuracy as ±20 min lags, but one coefficient per park "
         "instead of five collinear ones, so it stays explainable.",
         fig_smoothing(res)),
    ]


def _failures(res):
    r, b = res, res.loc[BASE]
    e01 = r.loc[r.index.str.startswith("E01")].bt_abs_edev_park
    e06 = r.loc[["E06_hour", "E06_stddev", "E06_temp", "E06_pressure"]].bt_abs_edev_park
    return [
        f"<b>Weighting recent days (E01):</b> park energy error {e01.min():.1f}–{e01.max():.1f} % vs {b.bt_abs_edev_park:.1f} %. We assumed the T/H relation drifts by month. "
        "Likely reason: the monthly shift in the fit reflects different wind regimes, not a changing site relation; fewer data points only add noise.",
        f"<b>Non-linear fits (E05 log, sqrt, piecewise):</b> no gain; log moves the 4–9 Sep park energy to {r.loc['E05_log', 'val_edev_park']:+.1f} %. "
        "Insight: wind speed relates linearly between parks; the non-linearity sits in the power curve and is handled there (E[P(v)]).",
        f"<b>Variance-ratio method (E04):</b> removes the bias ({r.loc['E04_variance_ratio', 'bt_edev_T1']:+.1f} % for T1) but adds noise to every step "
        f"(wind RMSE {b.bt_ws_rmse:.3f} → {r.loc['E04_variance_ratio', 'bt_ws_rmse']:.3f}). Insight: fix the bias in the power conversion, not by inflating the wind.",
        f"<b>Hour of day, turbulence, temperature, pressure (E06):</b> park energy error {e06.min():.1f}–{e06.max():.1f} %, no gain. Insight: they carry no "
        "information beyond the neighbours' wind; leaving them out keeps the model simpler, and we can show why.",
        f"<b>Out-of-sample and conformal bands (E07):</b> coverage {r.loc['E07_conformal', 'bt_cover'] * 100:.1f} % vs {b.bt_cover * 100:.1f} % with the training spread. "
        "Insight: the training residuals were already honest; estimating the spread per wind-speed bin is what tightens the band.",
        f"<b>Gradient boosting (E08):</b> park energy error {r.loc['E08_hgb', 'bt_abs_edev_park']:.1f} % vs {r.loc[BEST, 'bt_abs_edev_park']:.1f} % for the final model, "
        f"with only {r.loc['E08_hgb', 'bt_cover'] * 100:.0f} % band coverage. Insight: there is no hidden structure a black box finds that the transparent features miss.",
    ]


def _tldr(res):
    b, x = res.loc[BASE], res.loc[BEST]
    gain = (1 - x.bt_abs_edev_park / b.bt_abs_edev_park) * 100
    sb, sx = _park_stats(BASE), _park_stats(BEST)
    h = res.loc["E08_hgb"]
    return [
        f"Park energy error (the judged energy metric) fell from {b.bt_abs_edev_park:.1f} % to {x.bt_abs_edev_park:.1f} % per outage ({gain:.0f} % relative) "
        f"with the final model {BEST}; wind and power RMSE fell {(1 - x.bt_ws_rmse / b.bt_ws_rmse) * 100:.0f} % and {(1 - x.bt_p_rmse / b.bt_p_rmse) * 100:.0f} %.",
        f"The gain comes from removing the bias: the baseline under-predicted the park energy by {abs(sb['bias']):.1f} % on average, the final model is "
        f"{sx['bias']:+.1f} %; the outage-to-outage noise (±{sx['noise']:.0f} %) is weather luck and stays.",
        f"Gradient boosting reaches {h.bt_abs_edev_park:.1f} %, the transparent final model {x.bt_abs_edev_park:.1f} %; 95 % band coverage is "
        f"{x.bt_cover * 100:.1f} % (baseline {b.bt_cover * 100:.1f} %).",
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
    n_fig = [0]

    def fig(f):
        n_fig[0] += 1
        print(f.layout.title.text.split("<br>")[0].replace("<b>", "").replace("</b>", ""))
        return _fig_script(f, f"fig{n_fig[0]}")

    def ul(items, cls="notes"):
        return "\n".join([f'<ul class="{cls}">', *[f"  <li>{t}</li>" for t in items], "</ul>"])

    def card(title, why, result, f):
        return "\n".join(['<div class="card">', f"<h3>{title}</h3>",
                          f'<p><span class="tag">Why we expected a gain</span> {why}</p>',
                          f'<p><span class="tag">Result</span> {result}</p>', "</div>", fig(f)])

    body = [
        "<h2>1. Final model vs baseline on the judged metrics</h2>",
        fig(fig_scorecard(res)), fig(fig_winrate()),
        "<h3>Why the park error fell: bias vs noise</h3>", f"<p>{_bias_noise_text()}</p>", fig(fig_bias_noise()),
        "<h3>What the pattern means, and how to weight the metrics</h3>", ul(_weighting_text(res)),
        "<h2>2. The three changes that make the final model</h2>",
        f'<p class="sub">Final model {BEST}: T = a + b·H + c·O on 60-min averaged neighbours, with an offset and H factor per wind '
        "direction sector, and power averaged over the uncertainty. Each change was first tested alone against the baseline.</p>",
        *[card(*c) for c in _cards(res)],
        "<h2>3. What did not work, and what it tells us</h2>", ul(_failures(res)),
        fig(fig_leaderboard(res)),
        "<h2>4. Robustness checks</h2>",
        fig(fig_bias(res)), fig(fig_distribution()), fig(fig_calibration()), fig(fig_example()), fig(fig_blackbox(res)),
    ]
    css = [
        f"body {{font-family: 'Segoe UI', Helvetica, Arial, sans-serif; color: {INK}; max-width: 980px; "
        "margin: 0 auto; padding: 24px 16px; background: #fff; line-height: 1.45}",
        f"h1 {{margin: 0 0 4px}} .sub {{color: {MUTED}; margin: 0 0 16px}}",
        f"h2 {{margin: 40px 0 8px; padding-top: 12px; border-top: 1px solid {GRID}}} h3 {{margin: 18px 0 6px}}",
        f".tldr {{background: #f4f7f8; border-left: 4px solid {C_BEST}; padding: 8px 20px; margin-bottom: 24px}}",
        ".tldr li, .notes li {margin: 6px 0} .fig {margin-bottom: 28px}",
        f".card {{border-left: 4px solid {C_BEST}; padding: 2px 16px; margin: 20px 0 8px; background: #f7faf9}}",
        f".card p {{margin: 6px 0}} .tag {{font-weight: 600; color: {C_BEST}; margin-right: 4px}}",
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
        *body,
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
