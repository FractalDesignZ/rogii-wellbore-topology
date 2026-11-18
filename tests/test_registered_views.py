"""Projection, calibration and image-region isolation falsifiers."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rwt.registered_views import (  # noqa: E402
    allowed_measurements, endpoint_azimuth, extract_color_trace, fit_affine,
    inverse_projection, project_trajectory, region_metadata,
)


class TestProjectionRegistration(unittest.TestCase):
    def test_cli_requires_explicit_depth_unit(self) -> None:
        from inspect_registered_views import main
        errors = io.StringIO()
        with redirect_stderr(errors), self.assertRaises(SystemExit) as failure:
            main(["--image", "unused.png", "--lateral", "unused.csv", "--path-roi", "0", "0", "1", "1", "--out", "unused"])
        self.assertEqual(failure.exception.code, 2)
        self.assertIn("--depth-unit", errors.getvalue())

    def test_rotation_inverse_preserves_station_identity_and_missing_support(self) -> None:
        x = np.array([100.0, 103.0, np.nan, 109.0])
        y = np.array([200.0, 204.0, 207.0, 212.0])
        z = np.array([-1000.0, -1001.0, -1002.0, np.nan])
        ids = np.array([10, 20, 30, 40])
        view = project_trajectory(x, y, z, 31.0, station_ids=ids)
        np.testing.assert_array_equal(view["station_ids"], ids)
        np.testing.assert_array_equal(view["valid_path"], [True, True, False, False])
        self.assertTrue(np.isnan(view["u"][2]))
        rebuilt = inverse_projection(view["u"], view["cross"], view["z"], view["origin_xy"], 31.0)
        np.testing.assert_allclose(rebuilt[[0, 1]], np.column_stack([x, y, z])[[0, 1]], atol=1e-12)
        np.testing.assert_allclose(rebuilt[3, :2], [x[3], y[3]], atol=1e-12)
        self.assertTrue(np.isnan(rebuilt[3, 2]))
        self.assertAlmostEqual(endpoint_azimuth(np.array([0, 0, np.nan]), np.array([0, 2, np.nan])), 90.0)

    def test_affine_camera_roundtrip_and_control_residuals(self) -> None:
        world = np.array([[0, 0], [2, 0], [0, 4], [2, 4], [1, 1]], float)
        coefficients = np.array([[2.0, 0.3], [-0.2, -3.0], [17.0, 29.0]])
        pixels = np.column_stack([world, np.ones(len(world))]) @ coefficients
        camera = fit_affine(world, pixels, max_residual_px=1e-8)
        np.testing.assert_allclose(camera.to_pixels(world), pixels, atol=1e-10)
        np.testing.assert_allclose(camera.to_world(pixels), world, atol=1e-10)
        self.assertFalse(camera.metadata()["independently_validated"])
        self.assertEqual(camera.rank, 3)
        self.assertTrue(np.isnan(camera.to_pixels(np.array([[np.nan, 1.0]]))).all())

    def test_degenerate_and_inconsistent_controls_are_refused(self) -> None:
        line = np.array([[0, 0], [1, 1], [2, 2]], float)
        square = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
        for world, pixels in ((line, line), (square[:3], line), (square[:2], square[:2]), (square, np.full_like(square, np.nan))):
            with self.subTest(world=world.shape), self.assertRaises(ValueError):
                fit_affine(world, pixels)
        shifted = square.copy()
        shifted[-1] += 10
        with self.assertRaisesRegex(ValueError, "residual"):
            fit_affine(square, shifted, max_residual_px=0.1)

    def test_target_and_surface_values_cannot_change_measurement_ingress(self) -> None:
        frame = pd.DataFrame({"MD": [1, 2, 3], "X": [10, 11, 12], "Y": [20, 21, 22], "Z": [-100, -101, -102], "GR": [80, -999.25, 90]})
        baseline = allowed_measurements(frame)
        for name in ("TVT", "TVT_true", "TVT_input", "ANCC", "ASTNU", "BUDA", "private_secret"):
            frame[name] = [1e20, -1e20, 0.0]
        changed = allowed_measurements(frame)
        self.assertEqual(set(baseline), {"MD", "X", "Y", "Z", "GR", "MD__present", "X__present", "Y__present", "Z__present", "GR__present"})
        for key in baseline:
            np.testing.assert_array_equal(baseline[key], changed[key])
        self.assertFalse(changed["GR__present"][1])
        self.assertEqual(region_metadata("full_tvt_alignment", (0, 0, 2, 2))["role"], "evaluator_only")
        self.assertEqual(region_metadata("formation_tops", (0, 0, 2, 2))["role"], "evaluator_only")


@unittest.skipUnless(importlib.util.find_spec("PIL"), "optional Pillow image extra is unavailable")
class TestDeclaredImageRegion(unittest.TestCase):
    @staticmethod
    def fixture(path: Path, outside_blue: bool = False) -> None:
        from PIL import Image
        rgb = np.full((8, 10, 3), 255, dtype=np.uint8)
        rgb[3, 3] = [0, 0, 255]
        rgb[4, 4] = [12, 12, 245]
        rgb[4, 3] = [255, 0, 0]
        rgb[0, 0] = [0, 0, 255] if outside_blue else [255, 0, 0]
        Image.fromarray(rgb).save(path)

    def test_color_candidates_are_confined_to_declared_roi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.png", Path(tmp) / "b.png"
            self.fixture(a)
            self.fixture(b, outside_blue=True)
            first = extract_color_trace(a, (2, 2, 6, 6), tolerance=20)
            second = extract_color_trace(b, (2, 2, 6, 6), tolerance=20)
            np.testing.assert_array_equal(first["candidate_pixels"], [[3, 3], [4, 4]])
            np.testing.assert_array_equal(first["candidate_pixels"], second["candidate_pixels"])
            self.assertNotEqual(first["image_sha256"], second["image_sha256"])
            exact = extract_color_trace(a, (2, 2, 6, 6), tolerance=0)
            np.testing.assert_array_equal(exact["candidate_pixels"], [[3, 3]])

    def test_invalid_roi_and_color_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.png"
            self.fixture(path)
            for roi in ((-1, 0, 2, 2), (0, 0, 11, 2), (2, 2, 2, 4), (0, 0, 2.1, 2), (False, 0, 2, 2)):
                with self.subTest(roi=roi), self.assertRaises(ValueError):
                    extract_color_trace(path, roi)
            for color in ((0, 0, 256), (0, -1, 255), (0, 255)):
                with self.subTest(color=color), self.assertRaises(ValueError):
                    extract_color_trace(path, (0, 0, 2, 2), color=color)

    def test_cli_keeps_supplied_image_paired_without_unearned_registration(self) -> None:
        from inspect_registered_views import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "fixture.png"
            self.fixture(image)
            lateral = root / "well.csv"
            pd.DataFrame({"MD": [0, 1, 2], "X": [0, 1, 2], "Y": [0, 0, 0], "Z": [-100, -101, -102], "GR": [80, 90, 100], "TVT": [1e8, 2e8, 3e8], "ANCC": [1, 2, 3]}).to_csv(lateral, index=False)
            out = root / "evidence"
            args = ["--image", str(image), "--lateral", str(lateral), "--depth-unit", "ft", "--path-roi", "2", "2", "6", "6", "--out", str(out)]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            evidence = json.loads((out / "registration_evidence.json").read_text())["payload"]
            self.assertEqual(evidence["calibration"]["status"], "unregistered_candidates")
            self.assertEqual(evidence["source_image_availability"], "training_only")
            self.assertFalse(evidence["new_serve_time_measurement_claim"])
            with np.load(out / "registered_view_arrays.npz", allow_pickle=False) as arrays:
                self.assertNotIn("TVT", arrays.files)
                self.assertNotIn("ANCC", arrays.files)
                self.assertIn("candidate_pixels", arrays.files)
                self.assertIn("station_ids", arrays.files)
            with self.assertRaises(FileExistsError):
                main(args)


if __name__ == "__main__":
    unittest.main()
