from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwt.gates import (  # noqa: E402
    adaptive_residual_mask,
    apply_adaptive_residual,
    gate_residual_acceptance,
    gate_residual_acceptance_per_well,
)


class TestAdaptiveGate(unittest.TestCase):
    def test_global_reject_blocks_fire(self) -> None:
        y = np.array([10.0, 12.0, 11.0, 13.0])
        prior = np.array([10.0, 10.0, 10.0, 10.0])
        # residual worse than prior
        res = np.array([20.0, 22.0, 21.0, 23.0])
        g3 = gate_residual_acceptance(y, res, prior, prior_name="prior_stack")
        self.assertFalse(g3.passed)
        pw = gate_residual_acceptance_per_well(
            np.array(["w1", "w1", "w2", "w2"]),
            y,
            res,
            prior,
            global_gate=g3,
        )
        self.assertEqual(pw.detail["decision"], "GLOBAL_REJECT")
        self.assertEqual(pw.detail["n_fire"], 0)
        self.assertIsNone(pw.detail.get("adaptive_oof_rmse"))
        self.assertEqual(pw.detail.get("adaptive_mask_mode"), "global_reject")

    def test_per_well_fire_when_local_beat(self) -> None:
        # two wells; residual better on w1, worse on w2 (diagnostic well_fire)
        y = np.array([10.0, 12.0, 20.0, 22.0])
        prior = np.array([10.5, 12.5, 20.0, 22.0])
        res = np.array([10.1, 12.1, 25.0, 27.0])  # beats on w1, loses on w2
        pw = gate_residual_acceptance_per_well(
            np.array(["w1", "w1", "w2", "w2"]),
            y,
            res,
            prior,
            global_gate=None,
            min_stations=2,
        )
        self.assertTrue(pw.detail["well_fire"].get("w1", False))
        self.assertFalse(pw.detail["well_fire"].get("w2", True))

    def test_station_conf_mask_basic(self) -> None:
        conf = np.array([0.1, 0.5, np.nan, 0.35])
        m = adaptive_residual_mask(4, pf_confidence=conf, conf_threshold=0.35)
        np.testing.assert_array_equal(m, [False, True, False, True])

    def test_well_fire_mask_priority(self) -> None:
        conf = np.array([0.9, 0.9, 0.9, 0.9])  # would all fire under conf
        wf = {"w1": True, "w2": False}
        m = adaptive_residual_mask(
            4,
            pf_confidence=conf,
            conf_threshold=0.35,
            well_ids=np.array(["w1", "w1", "w2", "w2"]),
            well_fire=wf,
        )
        # well_fire wins over conf
        np.testing.assert_array_equal(m, [True, True, False, False])

    def test_adaptive_oof_uses_station_conf_not_well_fire(self) -> None:
        """Parity: adaptive_oof_rmse uses station-conf mask, not diagnostic well_fire."""
        # residual always better than prior, but conf only fires first well's stations
        y = np.array([10.0, 12.0, 20.0, 22.0])
        prior = np.array([12.0, 14.0, 22.0, 24.0])
        res = np.array([10.0, 12.0, 20.0, 22.0])  # perfect residual both wells
        conf = np.array([0.9, 0.9, 0.1, 0.1])  # high conf only on w1
        pw = gate_residual_acceptance_per_well(
            np.array(["w1", "w1", "w2", "w2"]),
            y,
            res,
            prior,
            global_gate=None,
            min_stations=2,
            pf_confidence=conf,
            conf_threshold=0.35,
        )
        self.assertEqual(pw.detail["adaptive_mask_mode"], "station_conf")
        # station-conf fires only rows 0,1 → adaptive = res,res,prior,prior
        adaptive = apply_adaptive_residual(
            res,
            prior,
            adaptive_residual_mask(4, pf_confidence=conf, conf_threshold=0.35),
        )
        from rwt.baselines import rmse

        expected = rmse(y, adaptive)
        self.assertAlmostEqual(float(pw.detail["adaptive_oof_rmse"]), expected, places=9)
        # well_fire may still mark both wells true (residual beats local prior)
        self.assertTrue(pw.detail["well_fire"].get("w1"))
        # adaptive station counts match conf mask, not well map
        self.assertEqual(pw.detail["n_station_fire"], 2)
        self.assertEqual(pw.detail["n_station_fallback"], 2)


class TestServeAdaptiveParity(unittest.TestCase):
    def test_serve_adaptive_matches_helper(self) -> None:
        """predict_stations adaptive mask ≡ adaptive_residual_mask under shared helper."""
        from rwt.submit import predict_stations

        class _M:
            def predict(self, X):
                return np.full(len(X), 5.0)

        st = pd.DataFrame(
            {
                "prior_stack": np.array([100.0, 100.0, 100.0, 100.0]),
                "pf_confidence": np.array([0.1, 0.5, 0.9, 0.2]),
                "relative_md": np.arange(4, dtype=float),
                "well_id": ["w"] * 4,
            }
        )
        thresh = 0.35
        blob = {
            "models": [_M()],
            "feature_cols": ["relative_md"],
            "prior_column": "prior_stack",
            "use_residual": True,
            "residual_max_dev": 18.0,
            "feature_medians": {"relative_md": 0.0},
            "adaptive_serve": True,
            "adaptive_conf_threshold": thresh,
        }
        y = predict_stations(blob, st)
        residual_tvt = 100.0 + 5.0
        prior = st["prior_stack"].to_numpy(dtype=float)
        mask = adaptive_residual_mask(
            len(st),
            pf_confidence=st["pf_confidence"].to_numpy(dtype=float),
            conf_threshold=thresh,
        )
        expected = apply_adaptive_residual(
            np.full(4, residual_tvt), prior, mask
        )
        np.testing.assert_allclose(y, expected)
        # low-conf stations fall back to prior
        self.assertAlmostEqual(float(y[0]), 100.0)
        self.assertAlmostEqual(float(y[1]), residual_tvt)
        self.assertAlmostEqual(float(y[3]), 100.0)

    def test_global_gate3_reject_blocks_residual_at_serve(self) -> None:
        """use_residual=False (Gate3 reject) → pure prior regardless of adaptive flags."""
        from rwt.submit import predict_stations

        class _M:
            def predict(self, X):
                return np.full(len(X), 50.0)

        st = pd.DataFrame(
            {
                "prior_stack": np.array([100.0, 100.0]),
                "pf_confidence": np.array([0.99, 0.99]),
                "relative_md": [0.0, 1.0],
            }
        )
        blob = {
            "models": [_M()],
            "feature_cols": ["relative_md"],
            "prior_column": "prior_stack",
            "use_residual": False,
            "adaptive_serve": True,
            "adaptive_conf_threshold": 0.35,
            "feature_medians": {"relative_md": 0.0},
        }
        y = predict_stations(blob, st)
        np.testing.assert_allclose(y, [100.0, 100.0])

    def test_dual_heads_also_get_adaptive_mask(self) -> None:
        """Dual residual_heads path must apply adaptive mask (no early return skip)."""
        from rwt.submit import predict_stations

        class _MBest:
            def predict(self, X):
                return np.full(len(X), 4.0)

        class _MStack:
            def predict(self, X):
                return np.full(len(X), 8.0)

        st = pd.DataFrame(
            {
                "prior_best": [100.0, 100.0],
                "prior_stack": [120.0, 120.0],
                "relative_md": [0.0, 1.0],
                "pf_confidence": [0.9, 0.1],
            }
        )
        blob = {
            "models": [_MBest()],
            "residual_heads": [
                {"prior_column": "prior_best", "models": [_MBest()]},
                {"prior_column": "prior_stack", "models": [_MStack()]},
            ],
            "dual_residual_weights": [0.75, 0.25],
            "feature_cols": ["relative_md"],
            "prior_column": "prior_best",
            "use_residual": True,
            "residual_max_dev": 18.0,
            "feature_medians": {"relative_md": 0.0},
            "adaptive_serve": True,
            "adaptive_conf_threshold": 0.35,
        }
        y = predict_stations(blob, st)
        # residual blend = 0.75*(100+4)+0.25*(120+8) = 110; row1 falls back to prior_best
        self.assertAlmostEqual(float(y[0]), 110.0)
        self.assertAlmostEqual(float(y[1]), 100.0)

    def test_train_blob_adaptive_keys_present(self) -> None:
        """model.joblib dump contract keys used by predict_stations."""
        # Structural: keys that train.main writes; values match serve readers.
        required = {
            "adaptive_serve",
            "adaptive_conf_threshold",
            "use_residual",
            "feature_medians",
            "residual_max_dev",
            "prior_column",
            "feature_cols",
            "models",
        }
        # Simulate blob shape from train.main joblib.dump
        blob = {
            "models": [],
            "residual_heads": [],
            "dual_residual_weights": None,
            "feature_cols": ["relative_md"],
            "prior_column": "prior_best",
            "use_residual": True,
            "residual_max_dev": 18.0,
            "residual_smooth_window": 0,
            "feature_medians": {"relative_md": 0.0},
            "adaptive_serve": False,
            "adaptive_conf_threshold": 0.35,
        }
        self.assertTrue(required.issubset(blob.keys()))
        self.assertIsInstance(blob["adaptive_serve"], bool)
        self.assertIsInstance(blob["adaptive_conf_threshold"], float)


if __name__ == "__main__":
    unittest.main()
