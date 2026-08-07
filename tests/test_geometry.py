from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwt.geometry import (  # noqa: E402
    discrete_curvature,
    e_ref_joint_strain,
    open_bishop_transport,
    rot_between,
    se3_screw_norm,
    tangents_from_xyz,
    trajectory_geometry_features,
)


class TestRotBetween(unittest.TestCase):
    def test_identity(self) -> None:
        u = np.array([1.0, 0.0, 0.0])
        R = rot_between(u, u)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=6)
        self.assertTrue(np.allclose(R @ u, u, atol=1e-9))

    def test_orthogonal(self) -> None:
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        R = rot_between(u, v)
        self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=6)
        self.assertTrue(np.allclose(R @ u, v, atol=1e-9))


class TestOpenCurve(unittest.TestCase):
    def test_straight_line_low_twist(self) -> None:
        n = 50
        x = np.linspace(0, 100, n)
        y = np.zeros(n)
        z = np.zeros(n)
        T = tangents_from_xyz(x, y, z)
        b = open_bishop_transport(T)
        self.assertLess(abs(b["cum_twist"]), 1e-6)
        self.assertLess(b["mean_curvature"], 1e-6)

    def test_helix_has_curvature(self) -> None:
        t = np.linspace(0, 4 * np.pi, 80)
        x, y, z = np.cos(t), np.sin(t), t
        T = tangents_from_xyz(x, y, z)
        curv = discrete_curvature(T)
        self.assertGreater(float(np.mean(curv)), 0.01)
        g = trajectory_geometry_features(t, x, y, z)
        self.assertGreater(g["mean_curv"], 0.01)
        self.assertEqual(g["n_stations"], 80)

    def test_helix_has_nonzero_torsion(self) -> None:
        """Discrete binormal torsion must be non-null on a helix (not pure-Bishop zero)."""
        t = np.linspace(0, 6 * np.pi, 120)
        x, y, z = np.cos(t), np.sin(t), 0.4 * t
        g = trajectory_geometry_features(t, x, y, z)
        self.assertGreater(abs(g["cum_twist"]), 0.05)
        self.assertGreater(g["mean_abs_torsion"], 1e-4)

    def test_e_ref_positive_on_bent_path(self) -> None:
        t = np.linspace(0, 1, 40)
        x = t * 100
        y = np.sin(t * 3) * 20
        z = -t * 50
        e = e_ref_joint_strain(x, y, z)
        self.assertGreater(float(e["mean_e_ref"]), 0.0)
        self.assertEqual(len(e["e_ref"]), 40)
        # identity rotation + unit translation → nonzero screw
        R = np.eye(3)
        self.assertGreater(se3_screw_norm(R, np.array([100.0, 0, 0])), 0.5)


if __name__ == "__main__":
    unittest.main()
