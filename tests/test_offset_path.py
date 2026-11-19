from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rwt.offset_path import minimum_cost_offset_path, relation_cost, select_offset_path


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


def test_context_weighted_transition_costs_match_full_path_enumeration():
    rng = np.random.default_rng(17)
    nodes = rng.normal(size=(5, 4))
    geo = rng.uniform(size=(4, 3, 4))
    gamma = rng.uniform(size=geo.shape)
    gamma[2] = np.nan
    agreement = rng.uniform(size=(4, 1, 4))
    edges = relation_cost({"geo":geo, "gamma":gamma}, {"geo":1+agreement, "gamma":2.})
    expected = geo*(1+agreement) + 2*np.nan_to_num(gamma)
    np.testing.assert_allclose(edges, expected)
    path = minimum_cost_offset_path(nodes, edges)

    def cost(route):
        vertex_cost = sum(nodes[t,j] for t,j in enumerate(route))
        return vertex_cost + sum(edges[t,j-i+1,j] for t,(i,j) in enumerate(zip(route, route[1:])))

    allowed = [p for p in product(range(4), repeat=5) if np.max(np.abs(np.diff(p))) <= 1]
    assert tuple(path) == min(allowed, key=cost)


def test_disabled_edges_and_missing_evidence():
    nodes = np.array([[0., np.inf], [2., 0.]])
    edges = np.zeros((1, 3, 2))
    edges[0, 2, 1] = np.inf  # Disable the move from 0 to 1.
    assert minimum_cost_offset_path(nodes, edges).tolist() == [0, 0]
    edges[0, 1, 0] = np.inf
    with pytest.raises(ValueError, match="no reachable"):
        minimum_cost_offset_path(nodes, edges)
    with pytest.raises(ValueError, match="weight"):
        relation_cost({"x":[1.]}, {"z":2.})
    with pytest.raises(ValueError, match="same edges"):
        relation_cost({"x":[1.], "y":[2.,3.]}, {"x":1., "y":1.})
    with pytest.raises(ValueError):
        minimum_cost_offset_path(nodes, np.zeros((1, 2, 3)))
    np.testing.assert_array_equal(relation_cost({"x":[np.nan, 2.]}, {"x":[3., np.nan]}), [0., 0.])
