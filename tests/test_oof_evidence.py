"""Exact retained OOF rows, selected paths, and coverage are independently replayable."""
from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwt.train import train_lgbm_residual  # noqa: E402


def stations(n_wells: int = 3, per_well: int = 24) -> pd.DataFrame:
    n = n_wells * per_well
    return pd.DataFrame({
        "well_id": np.repeat([f"W{i}" for i in range(n_wells)], per_well),
        "station": np.tile(np.arange(per_well), n_wells),
        "MD": np.tile(np.arange(per_well, dtype=float), n_wells),
        "relative_md": np.tile(np.arange(per_well, dtype=float), n_wells),
        "is_predict": np.ones(n, dtype=bool), "TVT": np.full(n, 102.),
        "prior_stack": np.full(n, 100.), "prior_last_tvt": np.full(n, 100.),
    })


def run_with_constant_residual(frame: pd.DataFrame, prediction: float, splits: int = 3) -> dict:
    fake = ModuleType("lightgbm")
    fake.Dataset = lambda *args, **kwargs: SimpleNamespace()
    fake.early_stopping = lambda *args, **kwargs: None
    fake.train = lambda *args, **kwargs: SimpleNamespace(
        best_iteration=1, predict=lambda features: np.full(len(features), prediction),
    )
    with patch.dict(sys.modules, {"lightgbm": fake}):
        return train_lgbm_residual(frame, n_splits=splits, prior_col="prior_stack")


class TestOofEvidence(unittest.TestCase):
    def assert_selected_score(self, result: dict) -> None:
        evidence = result["oof_evidence"]
        finite = np.isfinite(evidence.pred_selected.to_numpy())
        expected = float(np.sqrt(np.mean((evidence.y_true.to_numpy()[finite] - evidence.pred_selected.to_numpy()[finite]) ** 2)))
        self.assertAlmostEqual(expected, result["report"]["oof_rmse_selected"])
        self.assertEqual(result["report"]["oof_evidence"]["selected_finite_rows"], int(finite.sum()))
        self.assertFalse(any("beam" in name or "lattice" in name for name in evidence.columns))

    def test_group_folds_record_one_validation_fold_per_well(self) -> None:
        result = run_with_constant_residual(stations(), 1.)
        evidence = result["oof_evidence"]
        self.assertEqual(len(evidence), 72)
        self.assertEqual(evidence.fold_id.nunique(), 3)
        self.assertTrue(evidence.groupby("well_id").fold_id.nunique().eq(1).all())
        self.assertEqual(set(evidence.selected_mode), {"residual"})
        np.testing.assert_array_equal(evidence.pred_selected, evidence.pred_residual)
        self.assert_selected_score(result)

    def test_one_well_excludes_rows_never_validated(self) -> None:
        result = run_with_constant_residual(stations(n_wells=1, per_well=80), 1.)
        evidence = result["oof_evidence"]
        self.assertEqual(len(evidence), 20)
        self.assertEqual(set(evidence.station), set(range(60, 80)))
        self.assertEqual(set(evidence.fold_id), {0})
        self.assertEqual(result["report"]["oof_evidence"]["cv_strategy"], "single_well_tail_holdout")
        self.assert_selected_score(result)

    def test_adaptive_vector_matches_selected_threshold_and_scalar(self) -> None:
        frame = stations()
        high = np.arange(len(frame)) % 2 == 0
        frame["TVT"] = np.where(high, 102., 100.)
        frame["pf_confidence"] = high.astype(float)
        result = run_with_constant_residual(frame, 2.)
        evidence = result["oof_evidence"]
        self.assertTrue(result["adaptive_serve"])
        self.assertEqual(set(evidence.selected_mode), {"adaptive_residual"})
        np.testing.assert_array_equal(evidence.pred_selected, evidence.y_true)
        np.testing.assert_array_equal(evidence.adaptive_fire, high)
        self.assert_selected_score(result)

    def test_sparse_pure_floor_keeps_missing_selected_rows_and_reports_coverage(self) -> None:
        frame = stations()
        frame["prior_ncc_tvt"] = np.where(np.arange(len(frame)) % 2 == 0, 102., np.nan)
        result = run_with_constant_residual(frame, 5.)
        evidence = result["oof_evidence"]
        self.assertFalse(result["use_residual"])
        self.assertEqual(set(evidence.selected_mode), {"pure_prior"})
        self.assertEqual(len(evidence), 72)
        self.assertEqual(int(evidence.selected_finite.sum()), 36)
        self.assertFalse(result["report"]["oof_evidence"]["selected_shared_coverage_complete"])
        self.assertEqual(result["report"]["oof_evidence"]["selected_shared_coverage_fraction"], 0.5)
        self.assert_selected_score(result)


if __name__ == "__main__":
    unittest.main()
