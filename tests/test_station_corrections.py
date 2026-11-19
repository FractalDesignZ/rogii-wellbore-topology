import numpy as np
import pytest

from rwt.station_corrections import lift_increment_corrections


def test_lift_preserves_heel_and_interpolates_cumulative_correction_in_md():
    md = [0, 3, 4, 7, 9, 13, 15, 20]
    visible = np.array([100, 101, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    toe = np.array([90, 91, 92, 93, 94, 95])
    actual = lift_increment_corrections(md, visible, toe, [3, 4, 6], [8, 0, -6])
    np.testing.assert_array_equal(actual, [100, 101, 92, 99, 100, 97, 96, 97])
    np.testing.assert_array_equal(visible, [100, 101, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    np.testing.assert_array_equal(toe, [90, 91, 92, 93, 94, 95])


def test_empty_correction_and_fully_observed_well():
    np.testing.assert_array_equal(lift_increment_corrections([0, 2], [10, np.nan], [11], [], []), [10, 11])
    np.testing.assert_array_equal(lift_increment_corrections([0, 2], [10, 11], [], [], []), [10, 11])


@pytest.mark.parametrize("endpoints, increments", [([1.0], [2]), ([0], [2]), ([4], [2]),
                                                  ([2, 2], [2, 3]), (np.array([2, 1], dtype="uint64"), [2, 3]),
                                                  ([1, 2], [2]), ([1], [np.nan]), ([1, 2], [1e308, 1e308])])
def test_rejects_invalid_or_unrepresentable_corrections(endpoints, increments):
    with pytest.raises(ValueError):
        lift_increment_corrections([0, 1, 2], [10, np.nan, np.nan], [11, 12], endpoints, increments)
