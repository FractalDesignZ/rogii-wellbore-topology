from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwt.kernel_api import resolve_kernel  # noqa: E402
from rwt.provenance import public_code_digest, seal_artifact  # noqa: E402
from rwt.watermark import embed_column, verify_column, watermark_token  # noqa: E402


class TestKernelPublicDefault(unittest.TestCase):
    def test_default_public(self) -> None:
        k = resolve_kernel()
        self.assertFalse(k.is_private())
        self.assertTrue(k.kernel_id().startswith("public"))
        g = k.trajectory_features(
            np.linspace(0, 10, 20),
            np.linspace(0, 100, 20),
            np.zeros(20),
            np.linspace(0, -50, 20),
        )
        self.assertIn("mean_curv", g)


class TestWatermark(unittest.TestCase):
    def test_roundtrip(self) -> None:
        df = pd.DataFrame({"well_id": ["W1", "W1"], "MD": [1.0, 2.0]})
        w = embed_column(df)
        self.assertIn("rwt_wm", w.columns)
        v = verify_column(w)
        self.assertTrue(v["ok"])
        self.assertEqual(watermark_token("W1", "1.0000", "rwt.v1"), w.loc[0, "rwt_wm"])


class TestProvenance(unittest.TestCase):
    def test_seal_stable_fields(self) -> None:
        d = public_code_digest()
        self.assertGreater(d["n_files"], 5)
        self.assertEqual(len(d["tree_sha256"]), 64)
        s = seal_artifact({"hello": 1}, kind="test")
        self.assertEqual(s["artifact"], "RWT_PROVENANCE_SEAL")
        self.assertEqual(len(s["payload_sha256"]), 64)
        self.assertEqual(len(s["seal_sha256"]), 64)


class TestPrivateKernelNotInTreeDigest(unittest.TestCase):
    def test_no_private_paths_in_digest(self) -> None:
        d = public_code_digest()
        for path in d["files"]:
            self.assertNotIn("private_kernel", path)


if __name__ == "__main__":
    unittest.main()
