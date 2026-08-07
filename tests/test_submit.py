from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwt.submit import format_submission, submission_id  # noqa: E402


class TestSubmissionSchema(unittest.TestCase):
    def test_id_format(self) -> None:
        self.assertEqual(submission_id("015fe0d2", 1654), "015fe0d2_1654")

    def test_format_submission_columns(self) -> None:
        st = pd.DataFrame(
            {
                "well_id": ["000d7d20", "000d7d20"],
                "station": [1442, 1443],
                "is_predict": [True, True],
                "prior_flat_tvt": [100.0, 100.0],
            }
        )
        sub = format_submission(st, np.array([111.0, 112.0]))
        self.assertEqual(list(sub.columns), ["id", "tvt"])
        self.assertEqual(sub["id"].tolist(), ["000d7d20_1442", "000d7d20_1443"])
        self.assertEqual(sub["tvt"].tolist(), [111.0, 112.0])


if __name__ == "__main__":
    unittest.main()
