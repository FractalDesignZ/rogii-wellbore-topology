from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rwt.ensemble import blend_inner_predictions


def test_keyed_blend_preserves_fixed_float64_arithmetic():
    keys = np.array([5, 6, 20])
    full = np.array([10693.03, 10693.60, 10690.87])
    models = np.array([full - 3.1, full + 4.77, full + 3.55])
    expected = .5 * full + .5 * models.mean(axis=0)
    actual = blend_inner_predictions(keys, full, ((keys, model) for model in models))
    np.testing.assert_array_equal(expected.view(np.uint64), actual.view(np.uint64))


@pytest.mark.parametrize("keys,tvt,inner", [
    ([0, 1], [10., 11.], []),
    ([0, 1], [10., 11.], [([0, 2], [10., 11.])]),
    ([0, 1], [10., 11.], [([1, 0], [10., 11.])]),
    ([0, 1], [10., 11.], [([0, 1], [10., np.nan])]),
    ([0, 1], [10., 11.], [([0., 1.], [10., 11.])]),
    ([1, 1], [10., 11.], [([1, 1], [10., 11.])]),
    ([0, 1], [10., np.inf], [([0, 1], [10., 11.])]),
])
def test_incomplete_or_mismatched_predictions_are_refused(keys,tvt,inner):
    with pytest.raises(ValueError):
        blend_inner_predictions(keys,tvt,inner)
