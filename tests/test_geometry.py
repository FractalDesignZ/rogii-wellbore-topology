from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwt.geometry import (  # noqa: E402
    discrete_curvature,
    open_bishop_transport,
    rot_between,
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


if __name__ == "__main__":
    unittest.main()
