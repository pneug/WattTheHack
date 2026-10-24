"""Leaderboard in experiments/LOG.md from experiments/results.csv (latest row per id)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline import experiments

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results.csv"
LOG = ROOT / "experiments" / "LOG.md"
START, END = "<!-- leaderboard:start -->", "<!-- leaderboard:end -->"

COLS = [("bt_ws_rmse", "bt ws RMSE", "{:.3f}"), ("bt_p_rmse", "bt P RMSE [kW]", "{:.1f}"),
        ("bt_abs_edev", "bt mean abs E dev [%]", "{:.2f}"), ("bt_abs_edev_park", "bt park abs E dev [%]", "{:.2f}")]


def _f(v, fmt) -> str:
    return "" if pd.isna(v) else fmt.format(v)


def leaderboard() -> str:
    df = pd.read_csv(RESULTS).sort_values("run_at", kind="stable").groupby("id", as_index=False).last()
    df = df.sort_values("bt_abs_edev")
    head = ["id", "change (desc)"] + [c[1] for c in COLS] + ["T1/T2/T3 bias [%]", "val ws RMSE",
                                                             "val park E dev [%]", "95% cover", "best"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in df.iterrows():
        bias = "/".join(_f(r[f"bt_edev_T{i}"], "{:+.1f}") for i in (1, 2, 3))
        cells = [r["id"], str(r["desc"]).replace("|", "/")] + [_f(r[k], f) for k, _, f in COLS]
        cells += [bias, _f(r["val_ws_rmse"], "{:.3f}"), _f(r["val_edev_park"], "{:+.2f}"),
                  _f(r["bt_cover"], "{:.2f}"), "**BEST**" if r["id"] == experiments.BEST else ""]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def update_log() -> None:
    text = LOG.read_text(encoding="utf-8")
    block = f"{START}\n{leaderboard()}\n{END}"
    if START in text and END in text:
        a, b = text.index(START), text.index(END) + len(END)
        text = text[:a] + block + text[b:]
    else:
        text += "\n" + block + "\n"
    LOG.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    update_log()
    print(leaderboard())
