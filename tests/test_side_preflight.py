"""Shape-only preflight accounting and no-inference checks."""

import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from pipeline import side_preflight as preflight


class PreflightTests(unittest.TestCase):
    def test_exact_plan_counts_calibration_and_chunks(self):
        index = pd.date_range("2026-07-01", periods=100, freq="10min")
        data = {"wind_speed": pd.DataFrame({t: np.ones(100) for t in preflight.TARGETS}, index=index)}
        X = pd.DataFrame({"H": np.ones(100)}, index=index)
        cfg = {"model": {"name": "tabpfn_api", "max_train_rows": 30,
                          "calibration_rows": 10, "batch_size": 6},
               "features": [], "weights": "none"}
        windows = [("bt_test", index[80], index[90])]
        with patch.object(preflight.stages, "build_X", return_value=X), \
             patch.object(preflight.evaluate, "windows", return_value=windows):
            plan = preflight.build_plan(cfg, data)
        self.assertEqual(plan["fit_calls"], 3)
        self.assertEqual(plan["query_rows"], 60)
        self.assertEqual(plan["predict_calls"], 12)
        self.assertEqual(plan["shape_pairs"], [
            dict(context_rows=30, query_rows=4, features=1, calls=6),
            dict(context_rows=30, query_rows=6, features=1, calls=6)])

    def test_quote_uses_only_zero_surrogates_and_counts_all_calls(self):
        client = types.SimpleNamespace(set_access_token=Mock(), get_api_usage=Mock(return_value="quota available"),
                                       __version__="fake", fit=Mock(), predict=Mock())
        def estimate(train, query, **kwargs):
            self.assertFalse(train.any())
            self.assertFalse(query.any())
            self.assertEqual(train.shape, (30, 2))
            self.assertEqual(query.shape, (5, 2))
            cost = 10 if kwargs["operation"] == "cache_predict" else 20
            return dict(estimated_cost=cost, pricing_version="quota_v3", inputs=kwargs)
        client.estimate_cost = Mock(side_effect=estimate)
        plan = dict(shape_pairs=[dict(context_rows=30, query_rows=5, features=2, calls=3)],
                    model_version="v3.5", n_estimators=8)
        with patch.object(preflight, "_load_api_token", return_value="fake-secret"):
            result = preflight.quote_plan(plan, client)
        self.assertEqual(result["total_estimated_cost"], {"cache_predict": 30, "predict": 60})
        self.assertEqual(client.estimate_cost.call_count, 2)
        client.fit.assert_not_called()
        client.predict.assert_not_called()

    def test_remote_errors_do_not_expose_token(self):
        client = types.SimpleNamespace(set_access_token=Mock(side_effect=RuntimeError("fake-secret")))
        with patch.object(preflight, "_load_api_token", return_value="fake-secret"):
            with self.assertRaises(RuntimeError) as error:
                preflight.quote_plan({"shape_pairs": []}, client)
        self.assertNotIn("fake-secret", str(error.exception))
        self.assertTrue(error.exception.__suppress_context__)


if __name__ == "__main__":
    unittest.main()
