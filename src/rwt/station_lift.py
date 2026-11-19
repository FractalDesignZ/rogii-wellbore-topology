"""Return block-center TVT predictions to the original survey rows of one well."""
from __future__ import annotations

import numpy as np

from rwt.load import normalize_curve_nulls


def lift_block_predictions(md, z, tvt_input, center_station, block_tvt, *,
                           method: str = "md_height") -> np.ndarray:
    """Return absolute TVT on every original row, preserving the observed heel.

    MD, survey Z, visible TVT and predictions use feet. The visible TVT prefix
    ends immediately before the toe. Each absolute block prediction must have
    its original zero-based lower-middle center row in the eight-row toe grid.
    The final block may be partial. A fully observed well has zero blocks.

    'constant' broadcasts the prediction over its block. 'md_tvt' linearly
    interpolates TVT in MD using the last heel row and all block centers.
    'md_height' interpolates TVT + Z on the same knots, then subtracts the
    original row's Z. A missing row or bracketing-knot Z uses ordinary TVT
    interpolation for that row. Both interpolators hold TVT constant beyond
    the final center. The returned float64 vector follows input station order.
    """
    if method not in ("constant", "md_tvt", "md_height"):
        raise ValueError("method must be constant, md_tvt or md_height")
    md = np.asarray(md, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    visible = normalize_curve_nulls(tvt_input)
    centers = np.asarray(center_station)
    predictions = np.asarray(block_tvt, dtype=np.float64)
    if md.ndim != 1 or md.shape != z.shape or md.shape != visible.shape:
        raise ValueError("MD, Z and visible TVT must be aligned station vectors")
    if not np.isfinite(md).all() or np.any(np.diff(md) <= 0):
        raise ValueError("MD must be finite and strictly increasing")
    if centers.ndim != 1 or (centers.size and centers.dtype.kind not in "iu"):
        raise ValueError("center_station must contain integer station indices")
    if predictions.shape != centers.shape or not np.isfinite(predictions).all():
        raise ValueError("each center requires one finite absolute TVT prediction")

    known = np.flatnonzero(np.isfinite(visible))
    if not len(known) or not np.array_equal(known, np.arange(len(known))):
        raise ValueError("a nonempty contiguous visible heel is required")
    heel_end = len(known) - 1
    toe_start = heel_end + 1
    starts = np.arange(toe_start, len(md), 8)
    counts = np.minimum(8, len(md) - starts)
    expected = starts + (counts - 1) // 2
    if not np.array_equal(centers, expected):
        raise ValueError("center indices differ from the original eight-row block centers")
    if not len(centers):
        return visible
    if method == "constant":
        visible[toe_start:] = np.repeat(predictions, counts)
        return visible

    knots = np.r_[heel_end, expected]
    knot_md = md[knots]
    knot_tvt = np.r_[visible[heel_end], predictions]
    toe_md = md[toe_start:]
    toe_tvt = np.interp(toe_md, knot_md, knot_tvt)
    if method == "md_height":
        right = np.searchsorted(knot_md, toe_md, side="left").clip(1, len(knots) - 1)
        left = right - 1
        knot_z = z[knots]
        toe_z = z[toe_start:]
        supported = ((toe_md <= knot_md[-1]) & np.isfinite(toe_z)
                     & np.isfinite(knot_z[left]) & np.isfinite(knot_z[right]))
        low, high = left[supported], right[supported]
        weight = (toe_md[supported] - knot_md[low]) / (knot_md[high] - knot_md[low])
        lower_height = knot_tvt[low] + knot_z[low]
        upper_height = knot_tvt[high] + knot_z[high]
        toe_tvt[supported] = (1 - weight) * lower_height + weight * upper_height - toe_z[supported]
        toe_tvt[expected - toe_start] = predictions
    visible[toe_start:] = toe_tvt
    return visible
