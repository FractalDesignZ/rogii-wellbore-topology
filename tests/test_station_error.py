from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rwt.station_error import decompose_station_error


def test_station_error_separates_affine_and_finer_error_on_irregular_depth():
    md = np.array([1010., 1011., 1013., 1016.])
    truth = np.array([101., 102., 101., 99.])
    affine = 3. + 2. * (md - md.mean())
    higher = np.array([2., -3., 1., 0.])
    result = decompose_station_error(md, truth + affine + higher, truth)
    assert result.rows == 4
    assert result.mean_error_ft == pytest.approx(3.)
    assert result.drift_ft_per_md_ft == pytest.approx(2.)
    expected = [36., np.dot(affine - 3., affine - 3.), 14.]
    np.testing.assert_allclose(result[3:], expected)
    assert sum(result[3:]) == pytest.approx(np.dot(affine + higher, affine + higher))
    shifted = decompose_station_error(md + 123456, truth + affine + higher, truth)
    assert shifted == result


def test_one_station_only_has_offset_and_perfect_matches_are_zero():
    assert decompose_station_error([42.], [33.], [36.]) == (1, -3., 0., 9., 0., 0.)
    assert decompose_station_error([42., 100.], [33., 38.], [33., 38.]) == (2, 0., 0., 0., 0., 0.)


@pytest.mark.parametrize("md,prediction,truth", [
    ([], [], []), ([0., 1.], [0.], [0., 1.]),
    ([0., 0.], [0., 0.], [0., 0.]), ([1., 0.], [0., 0.], [0., 0.]),
    ([0., 1.], [0., np.nan], [0., 1.]), ([0., 1.], [0., 1.], [0., np.inf]),
])
def test_ambiguous_or_incomplete_station_error_is_rejected(md, prediction, truth):
    with pytest.raises(ValueError):
        decompose_station_error(md, prediction, truth)
