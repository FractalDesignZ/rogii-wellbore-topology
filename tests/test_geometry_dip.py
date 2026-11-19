from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
    def tearDown(self) -> None:
        cpu_feature_report.cache_clear()

    def test_cpu_report(self) -> None:
        cpu_feature_report.cache_clear()
        r = cpu_feature_report()
        self.assertIn("numpy_version", r)
        self.assertIn("simd", r)
        self.assertIn("openblas", r)
        self.assertEqual(r["openblas"], "openblas" in r["blas_backend"].lower())

    def test_cpu_report_numpy_version_fallbacks(self) -> None:
        features = {"SSE2": True, "AVX2": True, "FMA3": True, "AVX512F": False}
        core = SimpleNamespace(
            __cpu_features__=features,
            __cpu_baseline__=["SSE2"],
            __cpu_dispatch__=["AVX2", "AVX512F"],
        )
        cases = (
            (
                "2.5.5", "numpy._core._multiarray_umath",
                SimpleNamespace(CONFIG={"Build Dependencies": {
                    "blas": {"name": "scipy-openblas", "found": True},
                }}), "scipy-openblas", True,
            ),
            (
                "1.24.4", "numpy.core._multiarray_umath",
                SimpleNamespace(get_info=lambda key: {
                    "libraries": ["mkl_rt"],
                } if key == "blas_opt_info" else {}), "mkl", False,
            ),
            (
                "1.24.4", "numpy.core._multiarray_umath",
                SimpleNamespace(get_info=lambda key: {
                    "libraries": ["openblas64_"],
                } if key == "blas_ilp64_opt_info" else {}), "openblas", True,
            ),
        )
        for version, module_name, config, backend, openblas in cases:
            with self.subTest(numpy=version, backend=backend):
                def import_module(name: str) -> object:
                    if name == module_name:
                        return core
                    if name == "numpy.__config__":
                        return config
                    raise ImportError(name)

                cpu_feature_report.cache_clear()
                with (
                    patch("rwt.simd.import_module", side_effect=import_module),
                    patch("rwt.simd.np", SimpleNamespace(__version__=version)),
                    patch("rwt.simd.platform.machine", return_value="AMD64"),
                    patch.dict(sys.modules, {"numba": None}),
                ):
                    r = cpu_feature_report()
                self.assertEqual(r["simd_family"], "x86")
                self.assertEqual(r["simd"], features)
                self.assertEqual(r["numpy_cpu_baseline"], ["SSE2"])
                # A compiled dispatch target must not turn a False runtime flag True.
                self.assertEqual(r["numpy_cpu_dispatch"], ["AVX2", "AVX512F"])
                self.assertEqual(r["blas_backend"], backend)
                self.assertIs(r["openblas"], openblas)

    def test_cpu_report_unknown_hardware_and_backend(self) -> None:
        # Importing config alone proves neither OpenBLAS nor an ISA capability.
        for machine, family in (("mips64", "unknown"), ("armv7l", "ARM")):
            with self.subTest(machine=machine):
                def import_module(name: str) -> object:
                    if name == "numpy.__config__":
                        return SimpleNamespace()
                    raise ImportError(name)

                cpu_feature_report.cache_clear()
                with (
                    patch("rwt.simd.import_module", side_effect=import_module),
                    patch("rwt.simd.np", SimpleNamespace(__version__="unknown")),
                    patch("rwt.simd.platform.machine", return_value=machine),
                    patch.dict(sys.modules, {"numba": None}),
                ):
                    r = cpu_feature_report()
                self.assertEqual(r["simd_family"], family)
                self.assertEqual(r["simd"], {})
                self.assertEqual(r["blas_backend"], "unknown")
                self.assertIs(r["openblas"], False)

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
