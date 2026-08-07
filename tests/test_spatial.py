from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwt.features import build_station_frame  # noqa: E402
from rwt.middle_out import resolve_middle_out  # noqa: E402
from rwt.spatial import (  # noqa: E402
    attach_spatial_features,
    build_spatial_index,
    neighbor_tvt_transfer,
    well_spatial_node_from_lateral,
)


class TestSpatial(unittest.TestCase):
    def test_neighbor_graph_and_transfer(self) -> None:
        def make_well(wid: str, x0: float, y0: float, tvt0: float) -> pd.DataFrame:
            n = 30
            return pd.DataFrame(
                {
                    "MD": np.arange(n, dtype=float),
                    "X": x0 + np.linspace(0, 500, n),
                    "Y": y0 + np.zeros(n),
                    "Z": -9000 - np.linspace(0, 20, n),
                    "TVT": tvt0 + np.linspace(0, 10, n),
                    "TVT_input": [tvt0 + i if i < 10 else np.nan for i in range(n)],
                }
            )

        a = make_well("a", 0, 0, 100)
        b = make_well("b", 50, 10, 110)  # close neighbor
        c = make_well("c", 5000, 5000, 200)  # far
        idx = build_spatial_index([("a", a), ("b", b), ("c", c)])
        # adaptive radii from field quantiles (close pair a–b must link)
        neigh = idx.neighbors("a", k=5)
        ids = [w for w, _d in neigh]
        self.assertIn("b", ids)
        self.assertGreater(idx.max_dist, 0.0)
        node_b = well_spatial_node_from_lateral("b", b)
        tr = neighbor_tvt_transfer(a, node_b, close_dist=150.0)
        self.assertEqual(len(tr["neighbor_tvt"]), len(a))
        self.assertTrue(np.isfinite(tr["neighbor_tvt"]).all())

    def test_attach_defaults_when_well_absent(self) -> None:
        """Test wells not in train index get default neighbor columns (submit path)."""
        a = pd.DataFrame(
            {
                "MD": np.arange(10, dtype=float),
                "X": np.linspace(0, 100, 10),
                "Y": np.zeros(10),
                "Z": -np.arange(10, dtype=float),
                "TVT": 100 + np.arange(10, dtype=float),
                "TVT_input": [100.0 + i if i < 4 else np.nan for i in range(10)],
            }
        )
        idx = build_spatial_index([("a", a)])
        sp = resolve_middle_out(a, well_id="test_only")
        st = build_station_frame(a, None, sp, station_idx=sp.predict_idx)
        st2 = attach_spatial_features(st, a, idx, "test_only")
        self.assertEqual(float(st2["neighbor_count"].iloc[0]), 0.0)
        self.assertTrue(st2["neighbor_tvt"].isna().all())


if __name__ == "__main__":
    unittest.main()
