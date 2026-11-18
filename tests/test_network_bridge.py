"""Boundary falsifiers: labels, missing support, axis identity, optional native API."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwt.network_bridge import build_network_input, run_local_engine  # noqa: E402


def lateral() -> pd.DataFrame:
    i = np.arange(64, dtype=float)
    return pd.DataFrame({"MD": i, "Z": -2500 + np.sin(i / 4), "GR": 80 + np.cos(i / 6)})


class TestNetworkBoundary(unittest.TestCase):
    def test_target_and_private_columns_cannot_change_export(self) -> None:
        frame = lateral()
        baseline = build_network_input(frame, well_id="W", depth_unit="m")
        for name in ("TVT", "TVT_true", "TVT_input", "ANCC", "private_secret", "prior_beam_tvt"):
            frame[name] = np.arange(len(frame)) * 1e9
        changed = build_network_input(frame, well_id="W", depth_unit="m")
        self.assertEqual(baseline.manifest, changed.manifest)
        self.assertEqual(baseline.spec, changed.spec)
        self.assertEqual(set(changed.data["well"]), {"t", "MD", "Z", "GR", "MD__present", "Z__present", "GR__present"})

    def test_missing_support_survives_disk_roundtrip(self) -> None:
        frame = lateral()
        frame.loc[20:25, "GR"] = np.nan
        frame.loc[30, "GR"] = np.inf
        bundle = build_network_input(frame, well_id="W", depth_unit="m")
        self.assertEqual(bundle.manifest["support"]["GR"], {"observed": 57, "missing": 7, "contiguous_spans": 3})
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "input"
            bundle.write(out)
            with np.load(out / "data.npz", allow_pickle=False) as arrays:
                self.assertTrue(np.isnan(arrays["well/GR"][20:26]).all())
                self.assertFalse(arrays["well/GR__present"][30])
                np.testing.assert_array_equal(arrays["well/Z__present"], True)
            before = (out / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                bundle.write(out)
            self.assertEqual(before, (out / "manifest.json").read_bytes())

    def test_acquisition_order_pauses_and_reversals_are_preserved(self) -> None:
        frame = lateral()
        frame.loc[:6, "MD"] = [0, 1, 2, 2, 1, 0, 3]
        bundle = build_network_input(frame, well_id="W", depth_unit="m")
        np.testing.assert_array_equal(bundle.data["well"]["MD"], frame.MD)
        np.testing.assert_array_equal(bundle.data["well"]["t"], np.arange(len(frame)))
        self.assertEqual(bundle.mode["detector"], "slice")
        self.assertNotIn("windows", bundle.mode)
        self.assertFalse(bundle.mode["levels"]["dynamics"])

    def test_gr_null_codes_do_not_mask_finite_coordinate_values(self) -> None:
        frame = lateral()
        frame.loc[3:5, "GR"] = [-999.25, -999.0, -100.0]
        frame.loc[8, "Z"] = -9999.0
        bundle = build_network_input(frame, well_id="W", depth_unit="m")
        self.assertFalse(bundle.data["well"]["GR__present"][3:6].any())
        self.assertTrue(bundle.data["well"]["Z__present"][8])
        self.assertEqual(bundle.data["well"]["Z"][8], -9999.0)
        self.assertEqual(bundle.manifest["preprocessing"]["columns"]["Z"]["null_policy"], "nonfinite_only")
        self.assertTrue(np.isnan(bundle.data["well"]["GR"][3:6]).all())

    def test_declared_units_convert_and_missing_channels_stay_absent(self) -> None:
        metres = lateral()
        feet = metres.copy()
        feet[["MD", "Z"]] /= 0.3048
        a = build_network_input(metres, well_id="W", depth_unit="m")
        b = build_network_input(feet, well_id="W", depth_unit="ft")
        for name in ("MD", "Z", "GR"):
            np.testing.assert_allclose(a.data["well"][name], b.data["well"][name])
        missing = build_network_input(metres[["MD"]], well_id="W", depth_unit="m")
        self.assertFalse(missing.data["well"]["GR__present"].any())
        self.assertTrue(np.isnan(missing.data["well"]["GR"]).all())
        for unit in ("unknown", "seconds"):
            with self.assertRaises(ValueError):
                build_network_input(metres, well_id="W", depth_unit=unit)
        metres.loc[3, "MD"] = np.nan
        with self.assertRaisesRegex(ValueError, "MD must be finite"):
            build_network_input(metres, well_id="W", depth_unit="m")

    def test_reproducible_identity_and_observed_data_sensitivity(self) -> None:
        frame = lateral()
        a = build_network_input(frame, well_id="W", depth_unit="m")
        b = build_network_input(frame.copy(), well_id="W", depth_unit="m")
        self.assertEqual(a.manifest["input_sha256"], b.manifest["input_sha256"])
        frame.loc[3, "GR"] += 0.1
        c = build_network_input(frame, well_id="W", depth_unit="m")
        self.assertNotEqual(a.manifest["input_sha256"], c.manifest["input_sha256"])
        frame["well_id"] = ["a"] * 32 + ["b"] * 32
        with self.assertRaisesRegex(ValueError, "one well"):
            build_network_input(frame, well_id="a", depth_unit="m")

    def test_unsupported_or_failed_native_engine_never_saves_success(self) -> None:
        bundle = build_network_input(lateral(), well_id="W", depth_unit="m")
        for summary in (
            {"levels": {}, "all_property_checks_hold": True},
            {"levels": {"slices": {"detector": "slice"}}, "all_property_checks_hold": False},
            {"levels": {"slices": {"detector": "slice"}, "dynamics": {}}, "all_property_checks_hold": True},
        ):
            summary["operators_applied"] = ["S_slice"]
            kernel = ModuleType("network_engine.kernel")
            kernel.run = lambda *_, summary=summary: SimpleNamespace(summary_json=lambda: json.dumps(summary))
            with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"network_engine": ModuleType("network_engine"), "network_engine.kernel": kernel}):
                out = Path(tmp) / "native"
                with self.assertRaises(RuntimeError):
                    run_local_engine(bundle, out)
                self.assertFalse(out.exists())


@unittest.skipUnless(importlib.util.find_spec("network_engine"), "optional local Network Engine is not installed")
class TestNativeSlices(unittest.TestCase):
    def test_no_observed_signals_cannot_claim_relational_support(self) -> None:
        frame = lateral()
        frame[["Z", "GR"]] = np.nan
        bundle = build_network_input(frame, well_id="W", depth_unit="m")
        with tempfile.TemporaryDirectory() as tmp:
            receipt = run_local_engine(bundle, Path(tmp) / "native")
            self.assertEqual(receipt["supported_node_records"], 0)
            self.assertEqual(receipt["analyzed_relation_records"], 0)
            self.assertEqual(receipt["status"], "insufficient_relation_support")

    def test_native_support_and_repeat_hashes(self) -> None:
        frame = lateral()
        frame.loc[20:25, "GR"] = np.nan
        bundle = build_network_input(frame, well_id="W", depth_unit="m")
        from network_engine.kernel import run

        first = run(bundle.data, bundle.spec, bundle.mode)
        second = run(bundle.data, bundle.spec, bundle.mode)
        a, b = json.loads(first.summary_json()), json.loads(second.summary_json())
        self.assertEqual(a, b)
        self.assertTrue(a["all_property_checks_hold"])
        self.assertIsNone(a["levels"]["dynamics"])
        slices = a["levels"]["slices"]
        self.assertEqual(slices["detector"], "slice")
        self.assertEqual(sorted(r["n"] for r in slices["nodes"] if r["channel"] == "GR"), [20, 38])
        self.assertEqual(sorted(r["n"] for r in slices["relations"]), [20, 38])
        with tempfile.TemporaryDirectory() as tmp:
            receipt = run_local_engine(bundle, Path(tmp) / "native")
            self.assertEqual(receipt["analyzed_relation_records"], 2)
            self.assertTrue((Path(tmp) / "native" / "bridge_receipt.json").is_file())


if __name__ == "__main__":
    unittest.main()
