"""Checkpoint integrity and invalidation, without remote model calls."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from pipeline import side_cache


class SideCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.index = pd.date_range("2026-01-01", periods=3, freq="10min", tz="Europe/Berlin", name="time")
        self.data = {"wind_speed": pd.DataFrame({"H1": [1., 2., 3.], "T1": [4., 5., 6.]}, index=self.index)}
        self.config = {"model": {"name": "tabpfn_api", "random_state": 4}}
        self.cache = side_cache.SideBacktestCache(self.config, self.data, targets=["T1"], root=self.temporary.name)
        self.frame = pd.DataFrame({key: [1.2, np.nan, 3.4] for key in side_cache.COLUMNS}, index=self.index)
        self.frame.attrs = {"model": {"type": "dummy"}, "resid": [0.2, -0.3], "fitted": [3., 4.]}
        self.window = ("bt_0", self.index[0], self.index[-1] + pd.Timedelta(minutes=10))

    def test_roundtrip_preserves_timestamps_columns_and_calibration_attrs(self):
        self.cache.save(*self.window, {"T1": self.frame})
        result = self.cache.load(*self.window)
        pd.testing.assert_frame_equal(result["T1"], self.frame)
        self.assertEqual(result["T1"].attrs, self.frame.attrs)
        self.assertIsNone(self.cache.load("another-window", *self.window[1:]))
        payload = self.cache._path(*self.window).read_text()
        self.assertNotIn("random_state", payload)

    def test_fingerprint_invalidates_reference_target_schema_config_and_source(self):
        original = side_cache.fingerprint(self.config, self.data)
        for column in ("H1", "T1"):
            changed = {"wind_speed": self.data["wind_speed"].copy()}
            changed["wind_speed"].iloc[0, changed["wind_speed"].columns.get_loc(column)] += 1
            self.assertNotEqual(original, side_cache.fingerprint(self.config, changed))
        changed = {"wind_speed": self.data["wind_speed"].astype("float32")}
        self.assertNotEqual(original, side_cache.fingerprint(self.config, changed))
        changed = {"wind_speed": self.data["wind_speed"].rename(columns={"H1": "H2"})}
        self.assertNotEqual(original, side_cache.fingerprint(self.config, changed))
        changed = {"wind_speed": self.data["wind_speed"].set_axis(self.index + pd.Timedelta(seconds=1))}
        self.assertNotEqual(original, side_cache.fingerprint(self.config, changed))
        self.assertNotEqual(original, side_cache.fingerprint({"model": {"random_state": 5}}, self.data))
        source = Path(self.temporary.name) / "source.py"
        source.write_text("first")
        with patch.object(side_cache, "SOURCE_PATHS", [source]):
            first = side_cache.fingerprint(self.config, self.data)
            source.write_text("second")
            self.assertNotEqual(first, side_cache.fingerprint(self.config, self.data))

    def test_corruption_or_wrong_target_and_columns_is_a_safe_miss(self):
        self.cache.save(*self.window, {"T1": self.frame})
        path = self.cache._path(*self.window)
        original = path.read_text()
        for damage in ("{partial", "targets", "columns"):
            if damage == "{partial":
                path.write_text(damage)
            else:
                payload = json.loads(original)
                if damage == "targets":
                    payload["targets"]["unexpected"] = payload["targets"].pop("T1")
                else:
                    frame = json.loads(payload["targets"]["T1"]["frame"])
                    frame["columns"][0] = "wrong"
                    payload["targets"]["T1"]["frame"] = json.dumps(frame)
                path.write_text(json.dumps(payload))
            with self.assertWarnsRegex(RuntimeWarning, "invalid checkpoint ignored"):
                self.assertIsNone(self.cache.load(*self.window))

    def test_interrupted_atomic_replace_never_publishes_partial_result(self):
        with patch.object(side_cache.os, "replace", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                self.cache.save(*self.window, {"T1": self.frame})
        self.assertIsNone(self.cache.load(*self.window))
        self.assertEqual(list(self.cache.directory.iterdir()), [])
        self.cache.save(*self.window, {"T1": self.frame})
        old = self.cache._path(*self.window).read_bytes()
        changed = self.frame.copy()
        changed.iloc[0, 0] = 10
        with patch.object(side_cache.os, "replace", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                self.cache.save(*self.window, {"T1": changed})
        self.assertEqual(old, self.cache._path(*self.window).read_bytes())


if __name__ == "__main__":
    unittest.main()
