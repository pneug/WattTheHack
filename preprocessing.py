"""Load and clean the BayWa r.e. wind turbine dataset.

Usage:
    python preprocessing.py                # writes data/processed/*.csv
    from preprocessing import load_clean   # in notebooks / backend

See docs/preprocessing.md for the reasoning behind every rule.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data" / "dataset"
OUT_DIR = Path(__file__).parent / "data" / "processed"

TARGETS = ["T1", "T2", "T3"]
REFERENCES = ["H1", "H2", "H3", "H4", "H5", "O1", "O2"]
TURBINES = TARGETS + REFERENCES

FREQ = "10min"
STEP_HOURS = 10 / 60
START, END = pd.Timestamp("2026-07-01 00:00"), pd.Timestamp("2026-09-30 23:50")
HIDDEN = (pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-04"))  # scored by judges, T is NaN
VAL = (pd.Timestamp("2026-09-04"), pd.Timestamp("2026-09-10"))  # test_set=True but T is known

SENSORS = ["wind_speed", "wind_direction", "ambient_temperature", "air_pressure", "wind_speed_stddev"]
# Physically plausible ranges; anything outside becomes NaN.
VALID_RANGE = {
    "wind_speed": (0.0, 40.0),
    "wind_direction": (0.0, 360.0),
    "ambient_temperature": (-30.0, 50.0),
    "air_pressure": (900.0, 1100.0),
    "wind_speed_stddev": (0.0, 10.0),
}
STUCK_STEPS = 12  # identical non-zero wind speed for >= 2 h counts as a frozen sensor


def park_of(turbine: str) -> str:
    return turbine[0]


def load_raw(turbine: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read one turbine file exactly as delivered."""
    return pd.read_excel(data_dir / f"{turbine}_data.xlsx")


def clean_turbine(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning rules to one turbine. Returns a full 10-min grid with flag columns."""
    turbine = raw["turbine_name"].iloc[0]
    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="first")

    # 1. Reindex to the full grid: 3-4 Aug rows are absent from the files, not NaN.
    grid = pd.date_range(START, END, freq=FREQ, name="timestamp")
    df = df.set_index("timestamp").reindex(grid)
    df["flag_row_absent"] = df["turbine_name"].isna()
    df["turbine_name"] = turbine
    df["park"] = park_of(turbine)

    # 2. Sensor dropout: every sensor reads exactly 0 at once (seen in park T). Not calm wind.
    zero_cols = [c for c in ["wind_speed", "wind_direction", "ambient_temperature", "wind_speed_stddev"]
                 if df[c].notna().any()]
    df["flag_dropout"] = (df[zero_cols] == 0).all(axis=1)
    df.loc[df["flag_dropout"], SENSORS] = np.nan

    # 3. Out-of-range values become NaN.
    df["flag_out_of_range"] = False
    for col, (lo, hi) in VALID_RANGE.items():
        bad = df[col].notna() & ~df[col].between(lo, hi)
        df.loc[bad, col] = np.nan
        df["flag_out_of_range"] |= bad
    df["wind_direction"] = df["wind_direction"] % 360  # 360 deg == 0 deg

    # 4. Frozen sensor: same non-zero wind speed repeated for >= 2 h.
    ws = df["wind_speed"]
    run_id = (ws != ws.shift()).cumsum()
    run_len = ws.groupby(run_id).transform("size")
    df["flag_stuck"] = ws.notna() & (ws > 0) & (run_len >= STUCK_STEPS)
    df.loc[df["flag_stuck"], "wind_speed"] = np.nan

    # 5. Placeholder columns: a constant value carries no information (park O air pressure).
    for col in ["air_pressure", "wind_speed_stddev"]:
        if df[col].nunique() <= 1:
            df[col] = np.nan

    # 6. Split label from the timestamp (test_set is NaN on re-added rows).
    ts = df.index
    df["split"] = "train"
    df.loc[(ts >= VAL[0]) & (ts < VAL[1]), "split"] = "val"
    df.loc[(ts >= HIDDEN[0]) & (ts < HIDDEN[1]), "split"] = "hidden"
    df = df.drop(columns="test_set")

    df["flag_missing_ws"] = df["wind_speed"].isna()
    return df.reset_index()


def load_clean(turbines: list[str] = TURBINES, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """All turbines, cleaned, in long format (one row per turbine and timestamp)."""
    return pd.concat([clean_turbine(load_raw(t, data_dir)) for t in turbines], ignore_index=True)


def to_wide(long: pd.DataFrame, var: str = "wind_speed") -> pd.DataFrame:
    """Pivot one variable to timestamp x turbine (columns in TURBINES order)."""
    wide = long.pivot(index="timestamp", columns="turbine_name", values=var)
    return wide[[t for t in TURBINES if t in wide.columns]]


def fill_short_gaps(series: pd.Series, max_steps: int = 2) -> pd.Series:
    """Linearly interpolate gaps of at most `max_steps` (default 20 min). Longer gaps stay NaN.

    Meant for reference features only. Never fill the target: NaN means unknown, not zero.
    """
    is_na = series.isna()
    gap_id = (~is_na).cumsum()
    gap_len = is_na.groupby(gap_id).transform("sum")
    filled = series.interpolate(limit_area="inside")
    return filled.where(~is_na | (gap_len <= max_steps))


def load_power_curve(turbine: str = "T1", data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Power curve as columns wind_speed [m/s], power [kW]."""
    pc = pd.read_excel(data_dir / f"{turbine}_power_curve.xlsx")
    return pc[["wind_speed", "power"]].sort_values("wind_speed").reset_index(drop=True)


def wind_to_power(ws, curve: pd.DataFrame) -> np.ndarray:
    """Map wind speed to power [kW] by linear interpolation on the curve.

    0 kW below the first curve point and above the last one (26 m/s, assumed cut-out).
    NaN wind speed stays NaN.
    """
    ws = np.asarray(ws, dtype=float)
    p = np.interp(ws, curve["wind_speed"], curve["power"], left=0.0, right=0.0)
    return np.where(np.isnan(ws), np.nan, p)


def energy_kwh(power_kw) -> float:
    """Sum 10-min power [kW] to energy [kWh], ignoring NaN."""
    return float(np.nansum(power_kw) * STEP_HOURS)


def quality_report(long: pd.DataFrame) -> pd.DataFrame:
    """One row per turbine: what was found and fixed."""
    g = long.groupby("turbine_name", sort=False)
    known = long["split"] != "hidden"
    # NaN inside the delivered rows (excludes the absent 3-4 Aug rows and the hidden window)
    sensor_gap = long["flag_missing_ws"] & known & ~long["flag_row_absent"]
    by_turbine = sensor_gap.groupby(long["turbine_name"], sort=False)
    rep = pd.DataFrame({
        "rows_absent": g["flag_row_absent"].sum(),
        "nan_ws_in_files": by_turbine.sum(),
        "sensor_dropouts": g["flag_dropout"].sum(),
        "out_of_range": g["flag_out_of_range"].sum(),
        "stuck_sensor": g["flag_stuck"].sum(),
        "longest_sensor_gap_h": by_turbine.apply(_longest_gap_hours),
        "pressure_usable": g["air_pressure"].apply(lambda s: s.notna().any()),
        "stddev_usable": g["wind_speed_stddev"].apply(lambda s: s.notna().any()),
    })
    missing_known = (long["flag_missing_ws"] & known).groupby(long["turbine_name"]).sum()
    rep["valid_ws_pct"] = (100 * (1 - missing_known / known.groupby(long["turbine_name"]).sum())).round(1)
    return rep.reindex([t for t in TURBINES if t in rep.index])


def _longest_gap_hours(is_missing: pd.Series) -> float:
    """Longest run of True values, in hours."""
    best = run = 0
    for v in is_missing.to_numpy():
        run = run + 1 if v else 0
        best = max(best, run)
    return round(best * STEP_HOURS, 1)


if __name__ == "__main__":
    long = load_clean()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long.to_csv(OUT_DIR / "clean_long.csv", index=False)
    to_wide(long).to_csv(OUT_DIR / "wind_speed_wide.csv")
    rep = quality_report(long)
    rep.to_csv(OUT_DIR / "quality_report.csv")
    print(rep.to_string())
    print(f"\nWrote {len(long):,} rows to {OUT_DIR}")
