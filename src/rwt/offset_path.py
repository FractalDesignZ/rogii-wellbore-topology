"""Path selection on ordered candidate depths using observed relational evidence."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def relation_cost(evidence: Mapping, weights: Mapping) -> np.ndarray:
    """Sum named measurements times their weights on the same candidate edges.

    Evidence arrays share a shape. A weight can be a scalar or an array that
    broadcasts to that shape, e.g. a local reliability computed from another
    measurement. A missing reading or weight (NaN) contributes zero. Return
    signed costs, with smaller values preferred by minimum_cost_offset_path.
    """
    if not evidence or evidence.keys() != weights.keys():
        raise ValueError("supply a weight for every named evidence channel")
    total = None
    with np.errstate(over="raise", invalid="raise"):
        for name, reading in evidence.items():
            values = np.asarray(reading, dtype=np.float64)
            if total is None:
                total = np.zeros_like(values)
            elif values.shape != total.shape:
                raise ValueError("evidence channels must describe the same edges")
            weight = np.broadcast_to(np.asarray(weights[name], dtype=np.float64), values.shape)
            if np.isinf(values).any() or np.isinf(weight).any():
                raise ValueError("use NaN for missing evidence or weight")
            present = np.isfinite(values) & np.isfinite(weight)
            total += np.multiply(values, weight, out=np.zeros_like(values), where=present)
    return total


def minimum_cost_offset_path(station_costs, transition_costs=None) -> np.ndarray:
    """Minimize summed station and edge costs through one well's depth bins.

    station_costs[t, j] scores bin j. For q in (-1, 0, 1), the edge from bin
    j-q at station t to bin j at t+1 costs transition_costs[t, q+1, j].
    Edges beyond the grid are ignored. Omitted edge costs are zero. Positive
    infinity disables a bin or transition. Equal costs favor the lowest final
    bin and then the lowest preceding bin. Return one bin index per station.
    """
    costs = np.asarray(station_costs, dtype=np.float64)
    if costs.ndim != 2 or not costs.shape[1]:
        raise ValueError("station_costs must have shape (stations, positive bin count)")
    n, k = costs.shape
    edges = None
    if np.isnan(costs).any() or np.isneginf(costs).any():
        raise ValueError("station costs must be finite or positive infinity")
    if transition_costs is not None:
        edges = np.asarray(transition_costs, dtype=np.float64)
        if edges.shape != (max(n-1, 0), 3, k):
            raise ValueError("transition_costs must have shape (stations minus one, 3, bins)")
        if np.isnan(edges).any() or np.isneginf(edges).any():
            raise ValueError("transition costs must be finite or positive infinity")
    return _minimum_path(costs, n, k, edges)


def _minimum_path(station_costs, n, k, edges=None):
    path = np.empty(n, dtype=np.intp)
    if not n:
        return path
    parent_step = np.zeros((n, k), dtype=np.int8)
    incoming = np.empty((3, k))
    bins = np.arange(k)
    with np.errstate(over="raise", invalid="raise"):
        for i, local in enumerate(station_costs):
            if i:
                # Incoming rows visit the lower predecessor first to settle ties.
                incoming[0, 0] = incoming[2, -1] = np.inf
                incoming[0, 1:] = costs[:-1]
                incoming[1] = costs
                incoming[2, :-1] = costs[1:]
                if edges is not None:
                    incoming += edges[i-1, ::-1]
                choice = np.argmin(incoming, axis=0)
                parent_step[i] = choice - 1
                local = local + incoming[choice, bins]
            floor = local.min()
            if not np.isfinite(floor):
                raise ValueError(f"no reachable finite-cost path at station {i}")
            costs = local - floor
    path[-1] = np.argmin(costs)
    for i in range(n - 1, 0, -1):
        path[i-1] = path[i] + parent_step[i, path[i]]
    return path


def select_offset_path(center_ft, offsets_ft, desired_ft, *, row_weights=None) -> np.ndarray:
    """Return offset indices for the least-error reachable path through one well.

    At station i, bin j represents center_ft[i] + offsets_ft[j]. All first bins
    are allowed. Each following bin can stay or move by one index. The path
    minimizes the weighted sum of squared differences from desired_ft, which
    can come from an observed-data predictor. Call once per well in station
    order. Ties prefer the lower terminal bin, then the lower predecessor.
    """
    center = np.asarray(center_ft, dtype=np.float64)
    offsets = np.asarray(offsets_ft, dtype=np.float64)
    desired = np.asarray(desired_ft, dtype=np.float64)
    weights = np.ones_like(center) if row_weights is None else np.asarray(row_weights, dtype=np.float64)
    if center.ndim != 1 or center.shape != desired.shape or center.shape != weights.shape:
        raise ValueError("center_ft, desired_ft and row_weights must be matching 1-D vectors")
    if offsets.ndim != 1 or not offsets.size or not np.all(offsets[1:] > offsets[:-1]):
        raise ValueError("offsets_ft must be a nonempty strictly increasing vector")
    if not all(np.isfinite(a).all() for a in (center, offsets, desired, weights)) or (weights <= 0).any():
        raise ValueError("depths must be finite and weights must be positive and finite")
    with np.errstate(over="raise", invalid="raise"):
        relative = desired - center
        local = (weight * np.square(value - offsets) for value, weight in zip(relative, weights))
        return _minimum_path(local, len(center), len(offsets))
