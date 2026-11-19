"""Combine nested-model predictions before lifting them to original stations."""
from __future__ import annotations

import numpy as np


def blend_inner_predictions(block_index, full_tvt, inner_predictions) -> np.ndarray:
    """Return equal parts full-model TVT and mean inner-model TVT.

    Each inner prediction is a pair of strictly increasing integer block IDs
    and its TVT vector. The IDs must match the full model exactly. Apply the
    original station lift to the returned centers. Callers authenticate model
    ownership and data roles before passing their arrays.
    """
    keys = np.asarray(block_index)
    full = np.asarray(full_tvt, dtype=np.float64)
    if keys.ndim != 1 or not len(keys) or keys.dtype.kind not in "iu":
        raise ValueError("block IDs must be a nonempty one-dimensional integer array")
    if keys[0] < 0 or (keys[1:] <= keys[:-1]).any():
        raise ValueError("block IDs must be strictly increasing and nonnegative")
    if full.shape != keys.shape or not np.isfinite(full).all():
        raise ValueError("full-model TVT must be finite and aligned to the block IDs")
    inner = []
    for candidate_ids, candidate_tvt in inner_predictions:
        ids = np.asarray(candidate_ids)
        tvt = np.asarray(candidate_tvt, dtype=np.float64)
        if ids.dtype.kind not in "iu" or not np.array_equal(ids, keys):
            raise ValueError("inner-model block IDs differ from the reference")
        if tvt.shape != full.shape or not np.isfinite(tvt).all():
            raise ValueError("inner-model TVT must be finite and aligned to the block IDs")
        inner.append(tvt)
    if not inner:
        raise ValueError("the blend requires at least one inner model")
    return 0.5 * full + 0.5 * np.mean(np.stack(inner), axis=0)
