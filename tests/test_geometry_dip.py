from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwt.geometry import (  # noqa: E402
    apply_orthogonal_dip_rotation,
    estimate_lateral_azimuth_dip_deg,
    trajectory_in_dip_frame,
)
from rwt.simd import as_f64_contig, cpu_feature_report, vectorized_ncc_template  # noqa: E402


class TestDipRotation(unittest.TestCase):
    def test_identity_zero_angles(self) -> None:
        xyz = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        out = apply_orthogonal_dip_rotation(xyz, 0.0, 0.0)
        self.assertTrue(np.allclose(out, xyz, atol=1e-9))

    def test_orthogonal_preserves_lengths(self) -> None:
        rng = np.random.default_rng(0)
        xyz = rng.normal(size=(50, 3))
        out = apply_orthogonal_dip_rotation(xyz, 37.0, 12.0)
        d0 = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        d1 = np.linalg.norm(np.diff(out, axis=0), axis=1)
        self.assertTrue(np.allclose(d0, d1, atol=1e-9))

    def test_estimate_and_frame(self) -> None:
        t = np.linspace(0, 100, 40)
        x, y, z = t, 0.1 * t, -0.05 * t
        az, dip = estimate_lateral_azimuth_dip_deg(x, y, z)
        fr = trajectory_in_dip_frame(x, y, z, azimuth_deg=az, dip_deg=dip)
        self.assertEqual(len(fr["x_dip"]), 40)


class TestSimd(unittest.TestCase):
    def test_cpu_report(self) -> None:
        r = cpu_feature_report()
        self.assertIn("numpy_version", r)
        self.assertIn("simd", r)

    def test_ncc_vectorized(self) -> None:
        t = np.sin(np.linspace(0, 20, 200))
        q = t[40:72].copy()
        ncc, lag = vectorized_ncc_template(q, t)
        self.assertGreater(ncc, 0.5)
        self.assertIsInstance(lag, int)
        # lag should be near the planted offset (periodic signals may multi-peak)
        self.assertLess(abs(lag - 40) % 31, 8)  # period-ish tolerance

    def test_contig(self) -> None:
        a = np.asfortranarray(np.arange(10, dtype=float))
        b = as_f64_contig(a)
        self.assertTrue(b.flags["C_CONTIGUOUS"])


if __name__ == "__main__":
    unittest.main()
