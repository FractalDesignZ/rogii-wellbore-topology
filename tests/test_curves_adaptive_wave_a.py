"""Wave A: LAS null decode, gap features, dual adaptive conf×lifetime."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwt.curves import (  # noqa: E402
    conf_run_lifetime,
    fill_short_gaps,
    gap_run_lengths,
    gr_gap_features,
)
from rwt.gates import (  # noqa: E402
    adaptive_residual_mask,
    apply_adaptive_residual,
    select_adaptive_conf_threshold,
)
from rwt.load import normalize_curve_nulls  # noqa: E402


class TestLasNulls(unittest.TestCase):
    def test_sentinel_to_nan(self) -> None:
        x = np.array([80.0, -999.25, 90.0, -9999.0, 70.0])
        y = normalize_curve_nulls(x)
        self.assertTrue(np.isnan(y[1]))
        self.assertTrue(np.isnan(y[3]))
        self.assertAlmostEqual(float(y[0]), 80.0)

    def test_gr_absurd_floor(self) -> None:
        x = np.array([80.0, -100.0, 50.0])
        y = normalize_curve_nulls(x, absurd_le=-50.0)
        self.assertTrue(np.isnan(y[1]))
        self.assertAlmostEqual(float(y[2]), 50.0)


class TestGapFeatures(unittest.TestCase):
    def test_gap_run_lengths(self) -> None:
        valid = np.array([True, False, False, True, False])
        runs = gap_run_lengths(valid)
        np.testing.assert_array_equal(runs, [0, 2, 2, 0, 1])

    def test_fill_short_gaps_only(self) -> None:
        # long gap of 4 with max_gap=2 → no fill; short gap of 2 → fill
        x = np.array([1.0, np.nan, np.nan, 4.0, np.nan, np.nan, np.nan, np.nan, 9.0])
        filled, imp = fill_short_gaps(x, max_gap=2)
        self.assertTrue(imp[1] and imp[2])
        self.assertFalse(imp[4])  # long gap untouched
        self.assertTrue(np.isnan(filled[4]))
        self.assertAlmostEqual(float(filled[1]), 2.0, places=5)

    def test_gr_gap_features_keys(self) -> None:
        gr = np.array([10.0, np.nan, np.nan, 20.0])
        f = gr_gap_features(gr, short_gap=5)
        self.assertIn("gr_is_null", f)
        self.assertEqual(float(f["gr_is_null"][1]), 1.0)
        self.assertEqual(float(f["gr_gap_run"][1]), 2.0)


class TestConfLifetime(unittest.TestCase):
    def test_lifetime_runs(self) -> None:
        conf = np.array([0.9, 0.9, 0.1, 0.8, 0.8, 0.8])
        life = conf_run_lifetime(conf, 0.35)
        np.testing.assert_array_equal(life, [2, 2, 0, 3, 3, 3])

    def test_dual_mask_requires_lifetime(self) -> None:
        conf = np.array([0.9, 0.9, 0.9, 0.1])
        m1 = adaptive_residual_mask(4, pf_confidence=conf, conf_threshold=0.35, min_lifetime=0)
        m3 = adaptive_residual_mask(4, pf_confidence=conf, conf_threshold=0.35, min_lifetime=3)
        np.testing.assert_array_equal(m1, [True, True, True, False])
        # run of 3 high-conf → all three fire; min_life=3 ok
        np.testing.assert_array_equal(m3, [True, True, True, False])
        m4 = adaptive_residual_mask(4, pf_confidence=conf, conf_threshold=0.35, min_lifetime=4)
        np.testing.assert_array_equal(m4, [False, False, False, False])

    def test_select_prefers_lifetime_when_spikes_hurt(self) -> None:
        # residual perfect only on a long high-conf run; short spike residual is disaster
        y = np.array([10.0, 10.0, 10.0, 10.0, 20.0, 20.0])
        prior = np.array([10.0, 10.0, 10.0, 10.0, 20.0, 20.0])
        res = np.array([10.0, 10.0, 10.0, 10.0, 50.0, 50.0])  # wrecks last two
        conf = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9])  # all high conf
        # without lifetime, thr alone cannot refuse last two if all conf high
        # with L large enough to need 5-run — actually all 6 is one run
        # make last two a short isolated high-conf after a dip
        conf = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.95])
        res = np.array([10.0, 10.0, 10.0, 10.0, 20.0, 50.0])  # spike residual on isolated conf
        prior = np.array([10.0, 10.0, 10.0, 10.0, 20.0, 20.0])
        y = prior.copy()
        sel = select_adaptive_conf_threshold(
            y,
            res,
            prior,
            conf,
            candidates=(0.35,),
            lifetime_candidates=(0, 1, 3),
            default=0.35,
            default_lifetime=0,
        )
        # L>=3 refuses isolated last station → adaptive perfect
        self.assertLess(float(sel["adaptive_oof_rmse"]), 1e-9)
        self.assertGreaterEqual(int(sel["min_lifetime"]), 3)

    def test_serve_blob_lifetime(self) -> None:
        from rwt.submit import predict_stations

        class _M:
            def predict(self, X):
                return np.full(len(X), 5.0)

        st = pd.DataFrame(
            {
                "prior_stack": np.full(5, 100.0),
                "pf_confidence": np.array([0.9, 0.9, 0.9, 0.1, 0.95]),
                "relative_md": np.arange(5, dtype=float),
            }
        )
        blob = {
            "models": [_M()],
            "feature_cols": ["relative_md"],
            "prior_column": "prior_stack",
            "use_residual": True,
            "residual_max_dev": 18.0,
            "feature_medians": {"relative_md": 0.0},
            "adaptive_serve": True,
            "adaptive_conf_threshold": 0.35,
            "adaptive_min_lifetime": 3,
        }
        y = predict_stations(blob, st)
        # first three: conf run of 3 → fire residual 105; low conf / short spike → prior 100
        self.assertAlmostEqual(float(y[0]), 105.0)
        self.assertAlmostEqual(float(y[2]), 105.0)
        self.assertAlmostEqual(float(y[3]), 100.0)
        self.assertAlmostEqual(float(y[4]), 100.0)  # lifetime 1 < 3


if __name__ == "__main__":
    unittest.main()
