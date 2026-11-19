import numpy as np
import pytest

from rwt.member_corrections import blend_corrected_inner_predictions


def test_member_corrections_precede_the_keyed_blend():
    keys = np.array([1, 8, 12])
    full = np.array([31., 17., 22.])
    first = np.array([18., 43., 36.])
    second = np.array([32., 7., 14.])
    a = np.array([-3., 0., 1.])
    b = np.array([0., -5., 5.])
    expected = 0.5 * full + 0.5 * (np.stack((first+a, second+b)).mean(axis=0))
    actual = blend_corrected_inner_predictions(keys, full, [(keys, first, keys, a), (keys, second, keys, b)])
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


@pytest.mark.parametrize("ids, adjustment", [([2, 1], [1., 2.]), ([1., 2.], [1., 2.]),
                                               ([1, 2], [1.]), ([1, 2], [0., np.nan]),
                                               ([1, 2], [1e308, 0.])])
def test_invalid_correction_is_rejected(ids, adjustment):
    with pytest.raises(ValueError):
        blend_corrected_inner_predictions([1, 2], [20., 30.], [([1, 2], [1e308, 15.], ids, adjustment)])
