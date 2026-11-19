"""Least-squares path selection on one well's ordered candidate offsets."""
from __future__ import annotations

import numpy as np


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
    path = np.empty(len(center), dtype=np.intp)
    if not len(path):
        return path

    # Only the three incoming edges per bin can compete. Each backpointer
    # therefore fits in a signed byte, irrespective of the total bin count.
    parent_step = np.zeros((len(center), len(offsets)), dtype=np.int8)
    incoming = np.empty((3, len(offsets)))
    bins = np.arange(len(offsets))
    with np.errstate(over="raise", invalid="raise"):
        relative = desired - center
        for i in range(len(center)):
            loss = weights[i] * np.square(relative[i] - offsets)
            if i:
                incoming[0, 0] = incoming[2, -1] = np.inf
                incoming[0, 1:] = costs[:-1]
                incoming[1] = costs
                incoming[2, :-1] = costs[1:]
                choice = np.argmin(incoming, axis=0)
                parent_step[i] = choice - 1
                loss += incoming[choice, bins]
            costs = loss - loss.min()

    path[-1] = np.argmin(costs)
    for i in range(len(path) - 1, 0, -1):
        path[i-1] = path[i] + parent_step[i, path[i]]
    return path
