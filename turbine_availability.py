"""Helpers for uploaded turbine availability data."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd

EXPECTED_TURBINES = ["H1", "H2", "H3", "H4", "H5", "O1", "O2"]


def get_available_turbines(df: pd.DataFrame) -> list[str]:
    """Return sorted unique turbine names, or an empty list if none are present."""
    if "turbine_name" not in df.columns:
        return []

    turbine_names = df["turbine_name"].dropna().astype(str).str.strip()
    return sorted(name for name in turbine_names.unique() if name)


def load_turbine_data(file_name: str, file_content: bytes) -> pd.DataFrame:
    """Parse an uploaded CSV or Excel file, including European CSV decimals."""
    extension = Path(file_name).suffix.lower()
    file_buffer = BytesIO(file_content)

    if extension == ".csv":
        first_line = next(
            (line for line in file_content.decode("utf-8-sig").splitlines() if line.strip()),
            "",
        )
        separator = max((",", ";", "|", "\t"), key=first_line.count)
        return pd.read_csv(file_buffer, sep=separator, decimal=",")
    if extension in {".xls", ".xlsx"}:
        return pd.read_excel(file_buffer)

    raise ValueError(f"Unsupported turbine data file type: {extension or 'unknown'}")
