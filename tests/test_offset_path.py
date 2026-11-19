from itertools import product

import numpy as np
import pytest

from rwt.offset_path import select_offset_path


def test_path_matches_exhaustive_search_with_moving_center_and_weights():
    offsets = np.array([-5., 0., 7., 10.])
    center = np.array([99., 103., 98., 102.])
    desired = np.array([95., 112., 91., 114.])
    weights = np.array([1., 8., 1., 4.])
    allowed = [p for p in product(range(4), repeat=4) if np.max(np.abs(np.diff(p))) <= 1]
    path = select_offset_path(center, offsets, desired, row_weights=weights)

    def cost(indices):
        return np.sum(weights * (center + offsets[list(indices)] - desired)**2)

    assert tuple(path) in allowed
    assert cost(path) == min(map(cost, allowed))


def test_path_ties_edges_and_singletons():
    assert select_offset_path([10., 10.], [-1., 1.], [10., 10.]).tolist() == [0, 0]
    assert select_offset_path([0., 0.], [0., 1., 2., 3.], [3., 0.]).tolist() == [2, 1]
    assert select_offset_path([3., 4.], [1.], [9., 2.]).tolist() == [0, 0]
    assert select_offset_path([], [0.], []).size == 0


def test_rejects_invalid_grid_and_measurements():
    for offsets in ([], [0., 0.], [1., 0.], [np.nan]):
        with pytest.raises(ValueError):
            select_offset_path([0.], offsets, [1.])
    with pytest.raises(ValueError):
        select_offset_path([0.], [0.], [np.nan])
    with pytest.raises(ValueError):
        select_offset_path([0.], [0.], [0.], row_weights=[-1.])
    with pytest.raises(ValueError):
        select_offset_path([0., 1.], [0.], [1.])
