"""Field audit rejects score changes and confirmation contamination."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit_master_baseline", ROOT / "scripts/audit_master_baseline.py")
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


class TestMasterBaselineAudit(unittest.TestCase):
    def _fixture(self, root: Path) -> None:
        cohort = {"pilot_well_ids": ["dev_a", "dev_b"], "development_well_ids": ["dev_a", "dev_b", "spare"],
                  "confirmation_well_ids": ["held_out"]}
        cohort_path = root / "cohort.json"
        cohort_path.write_text(json.dumps(cohort))
        (root / "model").mkdir()
        (root / "ast.json").write_text(json.dumps({"manifest_sha256": "source"}))
        contract = {"source_manifest_sha256": "source", "cohort_path": str(cohort_path),
                    "cohort_sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
                    "pilot_well_ids": ["dev_a", "dev_b"], "folds": 2}
        (root / "contract.json").write_text(json.dumps({"payload": contract}))
        stations = pd.DataFrame({"well_id": ["dev_a", "dev_a", "dev_a", "dev_b", "dev_b"],
                                 "station": [0, 1, 2, 0, 1], "TVT": [1., 2., 3., 4., 5.],
                                 "is_predict": [True] * 5, "prior_stack": [0., 0., np.nan, 0., 0.]})
        stations.to_csv(root / "model/stations.csv", index=False)
        eligible = stations[stations.prior_stack.notna()].copy()
        oof = eligible[["well_id", "station"]].copy()
        oof["y_true"] = eligible.TVT.to_numpy()
        oof["pred_selected"] = eligible.TVT.to_numpy() + .5
        oof["fold_id"] = -1
        for fold, (_fit, val) in enumerate(GroupKFold(n_splits=2).split(np.zeros((4, 1)), groups=eligible.well_id)):
            oof.iloc[val, oof.columns.get_loc("fold_id")] = fold
        oof.to_csv(root / "model/oof_predictions.csv", index=False)
        report = {"oof_rmse_selected": .5, "n_stations": 4, "n_wells": 2, "residual_bases": ["prior_stack"],
                  "fold_rmse": [.5, .5], "oof_evidence": {"cv_strategy": "GroupKFold_by_well", "eligible_rows": 4,
                  "validation_rows": 4, "shared_rows": 4, "selected_finite_rows": 4,
                  "selected_shared_coverage_complete": True, "selected_shared_coverage_fraction": 1.0}}
        (root / "model/cv_report.json").write_text(json.dumps(report))

    @staticmethod
    def _report(root: Path, **updates) -> None:
        path = root / "model/cv_report.json"
        report = json.loads(path.read_text())
        report.update(updates)
        path.write_text(json.dumps(report))

    def test_exact_metric_and_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(auditor, "verify_source_manifest", return_value={"passed": True}):
            root = Path(tmp)
            self._fixture(root)
            result = auditor.audit(root)
            self.assertEqual(result["selected_oof_rmse"], 0.5)
            self.assertFalse(result["predictive_admitted"])
            self.assertEqual(result["input_labeled_prediction_rows"], 5)
            self.assertEqual(result["eligible_rows"], 4)
            self.assertEqual(result["excluded_nonfinite_truth_or_base_rows"], 1)
            self.assertTrue(result["fold_assignment_replayed"])
            self._report(root, oof_rmse_selected=.1)
            with self.assertRaisesRegex(ValueError, "does not match"):
                auditor.audit(root)

    def test_confirmation_and_missing_predictions_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(auditor, "verify_source_manifest", return_value={"passed": True}):
            root = Path(tmp)
            self._fixture(root)
            path = root / "model/oof_predictions.csv"
            original = pd.read_csv(path)
            changed = original.copy()
            changed.loc[0, "well_id"] = "held_out"
            changed.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "development pilot"):
                auditor.audit(root)
            changed = original.copy()
            changed.loc[0, "pred_selected"] = np.nan
            changed.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "coverage is incomplete"):
                auditor.audit(root)

    def test_duplicate_and_dropped_rows_cannot_keep_a_matching_scalar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(auditor, "verify_source_manifest", return_value={"passed": True}):
            root = Path(tmp)
            self._fixture(root)
            path = root / "model/oof_predictions.csv"
            original = pd.read_csv(path)
            pd.concat([original, original.iloc[:1]], ignore_index=True).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                auditor.audit(root)
            original.iloc[1:].to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "row coverage"):
                auditor.audit(root)

    def test_exact_station_keys_and_labels_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(auditor, "verify_source_manifest", return_value={"passed": True}):
            root = Path(tmp)
            self._fixture(root)
            path = root / "model/oof_predictions.csv"
            original = pd.read_csv(path)
            changed = original.copy()
            changed.loc[0, "station"] = 99
            changed.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "station keys"):
                auditor.audit(root)
            changed = original.copy()
            changed.loc[0, ["y_true", "pred_selected"]] += 10
            changed.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "truth labels"):
                auditor.audit(root)

    def test_fold_tampering_and_non_grouped_cv_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(auditor, "verify_source_manifest", return_value={"passed": True}):
            root = Path(tmp)
            self._fixture(root)
            path = root / "model/oof_predictions.csv"
            original = pd.read_csv(path)
            changed = original.copy()
            changed["fold_id"] = 1 - changed["fold_id"]
            changed.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "fold assignments"):
                auditor.audit(root)
            changed["fold_id"] = changed["fold_id"].astype(float)
            changed.loc[0, "fold_id"] = .5
            changed.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "nonnegative integers"):
                auditor.audit(root)
            original.to_csv(path, index=False)
            report = json.loads((root / "model/cv_report.json").read_text())
            report["oof_evidence"]["cv_strategy"] = "single_well_tail_holdout"
            self._report(root, **report)
            with self.assertRaisesRegex(ValueError, "grouped OOF"):
                auditor.audit(root)

    def test_reported_counts_and_input_duplicates_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(auditor, "verify_source_manifest", return_value={"passed": True}):
            root = Path(tmp)
            self._fixture(root)
            self._report(root, n_stations=3)
            with self.assertRaisesRegex(ValueError, "count mismatch"):
                auditor.audit(root)
            self._report(root, n_stations=4)
            path = root / "model/stations.csv"
            original = pd.read_csv(path)
            pd.concat([original, original.iloc[-1:]], ignore_index=True).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                auditor.audit(root)

    def test_reordered_oof_rows_preserve_keyed_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(auditor, "verify_source_manifest", return_value={"passed": True}):
            root = Path(tmp)
            self._fixture(root)
            path = root / "model/oof_predictions.csv"
            pd.read_csv(path).iloc[::-1].to_csv(path, index=False)
            result = auditor.audit(root)
            self.assertEqual(result["selected_oof_rmse"], .5)
            self.assertTrue(result["exact_station_keys_and_truth_match"])


if __name__ == "__main__":
    unittest.main()
