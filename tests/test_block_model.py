"""Experimental fixed block route: leakage, grouped evidence and serving parity."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rwt.block_model import (
    BlockPredictor, FEATURE_NAMES, LGBM_PARAMETERS, _metadata, apply_medians,
    build_block_features, evaluate_block_oof, fit_block_model, fit_train_medians,
    load_block_predictor, main, save_block_predictor, select_development_wells,
)


def well(index=0):
    n, heel = 45, 20
    md = np.arange(n, dtype=float) * 2
    visible = 100 + .02 * md
    last = visible[heel - 1]
    truth = np.where(np.arange(n) < heel, visible, last + 25 + index * 10)
    t = np.arange(0., 260., .5)
    curve = 70 + 20 * np.sin(t / 5)
    lateral = pd.DataFrame({"MD": md, "X": index * 1000 + md, "Y": .5 * md,
        "Z": -9999.25 - .1 * md, "GR": np.interp(truth, t, curve),
        "TVT_input": np.where(np.arange(n) < heel, visible, np.nan), "TVT": truth})
    return lateral, pd.DataFrame({"TVT": t, "GR": curve})


class BlockFeatureTests(unittest.TestCase):
    def test_target_top_mutation_and_batch_independence(self):
        lateral, book = well()
        expected, meta = build_block_features(lateral, book)
        build_block_features(*well(9))
        for name in ("TVT", "TVT_true", "ANCC", "BUDA", "private_secret"):
            lateral[name] = np.arange(len(lateral)) * 1e20
        actual, same = build_block_features(lateral, book)
        np.testing.assert_array_equal(expected.view(np.uint32), actual.view(np.uint32))
        np.testing.assert_array_equal(meta["center_station"], same["center_station"])
        self.assertEqual(expected.shape, (4, 92))

    def test_short_blocks_null_policy_and_anchors_without_clamp(self):
        lateral, book = well()
        lateral.loc[20:, "GR"] = -9999.25
        lateral["Z"] = -9999.25
        matrix, meta = build_block_features(lateral, book)
        np.testing.assert_array_equal(meta["row_count"], [8, 8, 8, 1])
        np.testing.assert_array_equal(meta["center_station"], [23, 31, 39, 44])
        self.assertTrue(np.isnan(matrix[:, -41:]).all())
        self.assertEqual(matrix[0, FEATURE_NAMES.index("last_z")], np.float32(-9999.25))

        class Model:
            def predict(self, features, **kwargs):
                return np.array([30., -30., np.nan, 0.])

        predictor = BlockPredictor(Model(), np.zeros(92, np.float32), _metadata())
        result = predictor.predict_well(lateral, book, well_id="W")
        np.testing.assert_array_equal(result.tvt[:20], lateral.TVT_input[:20])
        self.assertEqual(result.tvt.iloc[20], lateral.TVT_input.iloc[19] + 30)
        self.assertEqual(result.tvt.iloc[28], lateral.TVT_input.iloc[19] - 30)
        self.assertEqual(result.fallback_used.sum(), 8)
        self.assertFalse(result.gr_observed.iloc[20:].any())
        self.assertTrue(np.isfinite(result.tvt).all())

    def test_missing_observation_does_not_use_target_fallback(self):
        lateral, book = well()
        lateral["TVT_input"] = np.nan
        with self.assertRaisesRegex(ValueError, "visible heel"):
            build_block_features(lateral, book)

    def test_manifest_reservation_checked_before_csv_access(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = {"development_well_ids": ["A", "B", "C"], "confirmation_well_ids": ["HOLD"]}
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            with patch("rwt.block_model.pd.read_csv", side_effect=AssertionError("CSV must remain unopened")):
                with self.assertRaisesRegex(ValueError, "confirmation"):
                    main(["fit", "--data-root", str(root), "--manifest", str(path), "--well-id", "HOLD", "--out", str(root / "out")])
            self.assertFalse((root / "out").exists())
            self.assertEqual(select_development_wells(manifest), ["A", "B", "C"])
            with self.assertRaises(ValueError):
                select_development_wells({"development_well_ids": ["../A"], "confirmation_well_ids": []})


class BlockFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pairs = {f"W{i}": well(i) for i in range(3)}
        # A lazy label loader proves feature extraction precedes target access.
        cls.label_reads = []

        def labels(wid):
            cls.label_reads.append(wid)
            return cls.pairs[wid][0].TVT.to_numpy(float)

        cls.result = fit_block_model(cls.pairs, labels=labels)

    def test_complete_grouped_oof_exact_score_and_train_only_medians(self):
        result = self.result
        evidence = result.oof_evidence
        self.assertEqual(len(evidence), 75)
        self.assertFalse(evidence.duplicated(["well_id", "station"]).any())
        self.assertTrue(evidence.groupby("well_id").fold.nunique().eq(1).all())
        self.assertEqual(set(evidence.fold), {0, 1, 2})
        self.assertEqual(self.label_reads, list(self.pairs))
        sse = float(np.square(evidence.pred_selected - evidence.y_true).sum())
        self.assertEqual(sse, result.report["methods"]["candidate"]["sse"])
        self.assertTrue(result.report["complete_coverage_asserted"])
        self.assertFalse(result.report["predictive_admitted"])
        for fold, medians in enumerate(result.fold_medians):
            training = [build_block_features(*pair)[0] for wid, pair in self.pairs.items() if int(evidence.loc[evidence.well_id.eq(wid), "fold"].iloc[0]) != fold]
            np.testing.assert_array_equal(medians, fit_train_medians(np.concatenate(training)))
        self.assertEqual(result.predictor.metadata["lightgbm_parameters"], LGBM_PARAMETERS)
        self.assertEqual(result.predictor.metadata["residual_clamp"], None)

    def test_saved_plain_bundle_parity_and_schema_tamper(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bundle"
            save_block_predictor(self.result.predictor, path)
            loaded = load_block_predictor(path)
            expected = self.result.predictor.predict_well(*self.pairs["W0"], well_id="W0")
            actual = loaded.predict_well(*self.pairs["W0"], well_id="W0")
            pd.testing.assert_frame_equal(expected, actual)
            self.assertFalse(any(path.glob("*.pkl")))
            metadata = json.loads((path / "metadata.json").read_text())
            metadata["feature_names"][0] = "wrong"
            (path / "metadata.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "schema"):
                load_block_predictor(path)

    def test_model_artifact_tamper_and_fresh_output(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bundle"
            save_block_predictor(self.result.predictor, path)
            with self.assertRaises(FileExistsError):
                save_block_predictor(self.result.predictor, path)
            with (path / "model.txt").open("a") as stream:
                stream.write("tampered")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                load_block_predictor(path)

    def test_oof_auditor_rejects_dropped_rows_nonfinite_and_fold_mix(self):
        evidence = self.result.oof_evidence
        expected = evidence[["well_id", "station"]]
        with self.assertRaisesRegex(ValueError, "coverage"):
            evaluate_block_oof(evidence.iloc[:-1], expected_keys=expected)
        bad = evidence.copy()
        bad.loc[0, "pred_selected"] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            evaluate_block_oof(bad, expected_keys=expected)
        bad = evidence.copy()
        bad.loc[0, "fold"] = (int(bad.loc[0, "fold"]) + 1) % 3
        with self.assertRaisesRegex(ValueError, "one fold"):
            evaluate_block_oof(bad, expected_keys=expected)

    def test_oof_flag_and_station_schema_cannot_coerce_invalid_values(self):
        for field, values, message in (
            ("fallback_used", ["false"] * 75, "boolean"),
            ("fallback_used", pd.array([False] * 74 + [None], dtype="boolean"), "boolean"),
            ("station", [1.5] * 75, "integers"),
            ("station", np.arange(75) - 1, "nonnegative"),
        ):
            bad = self.result.oof_evidence.copy()
            bad[field] = values
            with self.assertRaisesRegex(ValueError, message):
                evaluate_block_oof(bad)
        metadata = _metadata()
        metadata["model_kind"] = "legacy_model"
        with self.assertRaisesRegex(ValueError, "schema"):
            BlockPredictor(object(), np.zeros(92, np.float32), metadata).predict_well(*well(), well_id="W")

    def test_cli_three_artificial_wells_fit_predict_and_saved_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            train = root / "data/train"
            train.mkdir(parents=True)
            for wid, (lateral, book) in self.pairs.items():
                lateral.to_csv(train / f"{wid}__horizontal_well.csv", index=False)
                book.to_csv(train / f"{wid}__typewell.csv", index=False)
            # A reserved file cannot be parsed as a CSV; successful fit must
            # keep it outside the selected input boundary.
            (train / "HOLD__horizontal_well.csv").write_bytes(b"\xffreserved")
            manifest = root / "cohort.json"
            manifest.write_text(json.dumps({"development_well_ids": list(self.pairs), "confirmation_well_ids": ["HOLD"]}))
            out = root / "fit"
            self.assertEqual(main(["fit", "--data-root", str(root / "data"), "--manifest", str(manifest), "--out", str(out)]), 0)
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(report["n_stations"], 75)
            self.assertTrue(report["exact_saved_vector_replay_passed"])
            self.assertFalse(report["confirmation_labels_parsed"])
            contract = json.loads((out / "fit_contract.json").read_text())
            self.assertIn("execution_graph.py", contract["source_sha256"])
            self.assertFalse(any("HOLD" in path for path in contract["input_sha256"]))
            with np.load(out / "oof_predictions.npz", allow_pickle=False) as evidence:
                self.assertEqual(len(evidence["station"]), 75)
                self.assertEqual(evidence["fallback_used"].dtype, np.dtype(bool))
            prediction_path = root / "predictions.csv"
            self.assertEqual(main(["predict", "--model", str(out / "model"), "--lateral", str(train / "W0__horizontal_well.csv"),
                "--typewell", str(train / "W0__typewell.csv"), "--well-id", "W0", "--out", str(prediction_path)]), 0)
            prediction = pd.read_csv(prediction_path)
            self.assertEqual(len(prediction), 45)
            self.assertTrue(np.isfinite(prediction.tvt).all())
            np.testing.assert_allclose(prediction.tvt[:20], self.pairs["W0"][0].TVT_input[:20])


if __name__ == "__main__":
    unittest.main()
