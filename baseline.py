"""OLS baseline: reconstruct park T wind speed from the park H and O averages.

    T_i(t) = a_i + b_i * H(t) + c_i * O(t)      (one fit per turbine, then clipped at 0)

Usage:
    from baseline import OLSBaseline
    model = OLSBaseline().fit(ws, exclude=("2026-09-01", "2026-09-04"))
    pred = model.predict(ws)                      # timestamp x T1..T3 wind speed [m/s]
    loss = model.energy_loss(ws, power_curve)     # per step and turbine [kWh]

`ws` is the wide table from preprocessing.to_wide(load_clean()). See docs/baseline.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from preprocessing import STEP_HOURS, TARGETS, load_power_curve, wind_to_power

H_COLS = ["H1", "H2", "H3", "H4", "H5"]
O_COLS = ["O1", "O2"]


def park_means(ws: pd.DataFrame) -> pd.DataFrame:
    """Average of the turbines that report in each step (a missing turbine is skipped)."""
    return pd.DataFrame({"H": ws[H_COLS].mean(axis=1), "O": ws[O_COLS].mean(axis=1)})


class OLSBaseline:
    """Three numbers per turbine (a, b, c), fitted by least squares and then kept static."""

    def __init__(self, targets: list[str] = TARGETS):
        self.targets = targets
        self.coef_: dict[str, tuple[float, float, float]] = {}
        self.sigma_: dict[str, float] = {}

    def fit(self, ws: pd.DataFrame, exclude: tuple | None = None) -> "OLSBaseline":
        """Fit on all steps where target, H and O are known, except the `exclude` window [start, end)."""
        X = park_means(ws)
        keep = pd.Series(True, index=ws.index)
        if exclude is not None:
            keep = ~((ws.index >= pd.Timestamp(exclude[0])) & (ws.index < pd.Timestamp(exclude[1])))
        for t in self.targets:
            ok = keep & X.notna().all(axis=1) & ws[t].notna()
            A = np.column_stack([np.ones(ok.sum()), X[ok].values])
            beta, *_ = np.linalg.lstsq(A, ws.loc[ok, t].values, rcond=None)
            self.coef_[t] = tuple(float(b) for b in beta)
            self.sigma_[t] = float((ws.loc[ok, t].values - A @ beta).std())  # residual spread [m/s]
        return self

    def predict(self, ws: pd.DataFrame) -> pd.DataFrame:
        """Predicted wind speed [m/s] per target; NaN where H or O is missing."""
        X = park_means(ws)
        out = {t: (a + b * X["H"] + c * X["O"]).clip(lower=0) for t, (a, b, c) in self.coef_.items()}
        return pd.DataFrame(out)

    def coefficients(self) -> pd.DataFrame:
        """Table of a, b, c and residual sigma per turbine (for display)."""
        return pd.DataFrame(self.coef_, index=["a", "b (H)", "c (O)"]).T.assign(
            sigma=pd.Series(self.sigma_))

    def power(self, ws: pd.DataFrame, power_curve: pd.DataFrame | None = None) -> pd.DataFrame:
        """Predicted power [kW] per target through the power curve."""
        curve = load_power_curve() if power_curve is None else power_curve
        pred = self.predict(ws)
        return pred.apply(lambda s: pd.Series(wind_to_power(s, curve), index=s.index))

    def energy_loss(self, ws: pd.DataFrame, power_curve: pd.DataFrame | None = None) -> pd.DataFrame:
        """Predicted energy [kWh] per step and target (= the lost energy when applied to the outage)."""
        return self.power(ws, power_curve) * STEP_HOURS


def run_outage(ws: pd.DataFrame, start, end, power_curve: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fit without the outage window [start, end) and return the predictions inside it.

    Columns: <T>_pred (m/s), <T>_actual (m/s, NaN in the hidden window), <T>_loss_kwh.
    """
    model = OLSBaseline().fit(ws, exclude=(start, end))
    win = ws[(ws.index >= pd.Timestamp(start)) & (ws.index < pd.Timestamp(end))]
    pred, kwh = model.predict(win), model.energy_loss(win, power_curve)
    out = pd.concat({"pred": pred, "actual": win[model.targets], "loss_kwh": kwh}, axis=1)
    out.columns = [f"{t}_{k}" if k != "loss_kwh" else f"{t}_loss_kwh" for k, t in out.columns]
    return out


if __name__ == "__main__":
    from preprocessing import load_clean, to_wide

    ws = to_wide(load_clean())
    res = run_outage(ws, "2026-09-04", "2026-09-10")  # validation window: actual T is known
    m = OLSBaseline().fit(ws, exclude=("2026-09-04", "2026-09-10"))
    print(m.coefficients().round(3))
    for t in TARGETS:
        e = (res[f"{t}_pred"] - res[f"{t}_actual"]).dropna()
        print(f"{t}: RMSE {np.sqrt((e ** 2).mean()):.2f} m/s, predicted {res[f'{t}_loss_kwh'].sum() / 1000:,.1f} MWh")
