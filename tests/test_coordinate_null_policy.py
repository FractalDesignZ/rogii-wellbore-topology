"""Finite coordinate crossings cannot inherit the GR null convention."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwt.load import NULLABLE_CURVES, apply_curve_nulls_frame, read_well_csv  # noqa: E402
from rwt.network_bridge import build_network_input  # noqa: E402
from rwt.registered_views import allowed_measurements, project_trajectory  # noqa: E402


def smooth_fixture() -> pd.DataFrame:
    n = 7
    return pd.DataFrame({
        "MD": np.arange(n, dtype=float), "X": np.arange(n, dtype=float) * 2, "Y": np.ones(n),
        "Z": [-9999.14, -9999.17, -9999.21, -9999.25, -9999.28, -9999.32, -9999.36],
        "GR": [80., 81., -999.25, 82., 83., 84., 85.],
        "TVT": [100., 101., 102., 103., 104., 105., 106.],
        "TVT_input": [100., 101., 102., np.nan, np.nan, np.nan, np.nan],
    })


class TestCoordinateNullPolicy(unittest.TestCase):
    def test_smooth_crossing_survives_default_curve_cleaning(self) -> None:
        frame = smooth_fixture()
        original = frame.copy(deep=True)
        cleaned = apply_curve_nulls_frame(frame)
        self.assertNotIn("Z", NULLABLE_CURVES)
        np.testing.assert_array_equal(cleaned.Z, frame.Z)
        self.assertEqual(cleaned.Z.iloc[3], -9999.25)
        self.assertTrue(np.isnan(cleaned.GR.iloc[2]))
        pd.testing.assert_frame_equal(frame, original)

    def test_csv_bridge_and_view_preserve_coordinate_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "smooth__horizontal_well.csv"
            smooth_fixture().rename(columns={"Z": "TVD"}).to_csv(path, index=False)
            loaded = read_well_csv(path)
            raw = pd.read_csv(path).rename(columns={"TVD": "Z"})
            np.testing.assert_array_equal(loaded.Z, raw.Z)
            a = build_network_input(raw, well_id="smooth", depth_unit="ft")
            b = build_network_input(loaded, well_id="smooth", depth_unit="ft")
            self.assertEqual(a.manifest["input_sha256"], b.manifest["input_sha256"])
            self.assertTrue(b.data["well"]["Z__present"].all())
            np.testing.assert_array_equal(b.data["well"]["Z"], raw.Z.to_numpy() * 0.3048)
            view = allowed_measurements(loaded)
            np.testing.assert_array_equal(view["Z"], raw.Z)
            self.assertTrue(view["Z__present"].all())
            projected = project_trajectory(view["X"], view["Y"], view["Z"], 0.)
            self.assertTrue(projected["valid_path"].all())

    def test_explicit_coordinate_null_declaration_is_opt_in(self) -> None:
        frame = smooth_fixture()
        preserved = apply_curve_nulls_frame(frame)
        declared = apply_curve_nulls_frame(frame, columns=("Z",))
        self.assertTrue(np.isfinite(preserved.Z.iloc[3]))
        self.assertTrue(np.isnan(declared.Z.iloc[3]))
        self.assertEqual(declared.Z.iloc[2], frame.Z.iloc[2])
        # Explicitly cleaning only Z does not silently clean another column.
        self.assertEqual(declared.GR.iloc[2], -999.25)

    def test_nonfinite_coordinates_stay_missing_without_finite_sentinel_rules(self) -> None:
        frame = smooth_fixture()
        frame.loc[0, "Z"] = np.inf
        frame.loc[1, "Z"] = -np.inf
        frame.loc[2, "Z"] = np.nan
        cleaned = apply_curve_nulls_frame(frame)
        self.assertTrue(np.isnan(cleaned.Z.to_numpy()[:3]).all())
        bundle = build_network_input(frame, well_id="smooth", depth_unit="m")
        view = allowed_measurements(frame)
        for present in (bundle.data["well"]["Z__present"], view["Z__present"]):
            np.testing.assert_array_equal(present, [False, False, False, True, True, True, True])
        self.assertEqual(view["Z"][3], -9999.25)

    def test_target_null_conventions_remain_separate(self) -> None:
        frame = smooth_fixture()
        frame.loc[0, "TVT"] = -999.25
        frame.loc[1, "TVT_input"] = -9999.25
        cleaned = apply_curve_nulls_frame(frame)
        self.assertTrue(np.isnan(cleaned.TVT.iloc[0]))
        self.assertTrue(np.isnan(cleaned.TVT_input.iloc[1]))
        self.assertEqual(cleaned.Z.iloc[3], -9999.25)


if __name__ == "__main__":
    unittest.main()
