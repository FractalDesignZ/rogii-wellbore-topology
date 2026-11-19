"""Apply interval corrections on the original station grid of one well."""

from __future__ import annotations

import numpy as np

from rwt.load import normalize_curve_nulls


def lift_increment_corrections(md, tvt_input, toe_tvt, endpoint_station, increment_correction) -> np.ndarray:
    """Return full-station TVT with an anchored cumulative correction.

    MD, TVT and signed correction increments use feet. ``toe_tvt`` covers each
    row after the contiguous visible heel. Each integer station in
    ``endpoint_station`` is the right edge of an interval starting at the
    previous edge or at the final heel row. Use a zero increment when an
    interval lacks additional evidence and should carry forward its correction.
    The correction starts at zero, interpolates in original MD and stays
    constant after the final endpoint. Visible TVT is preserved.
    """
    md = np.asarray(md, dtype=np.float64)
    visible = normalize_curve_nulls(tvt_input)
    toe = np.asarray(toe_tvt, dtype=np.float64)
    endpoints = np.asarray(endpoint_station)
    increments = np.asarray(increment_correction, dtype=np.float64)

    if md.ndim != 1 or md.shape != visible.shape or not np.isfinite(md).all() or np.any(md[1:] <= md[:-1]):
        raise ValueError("MD and visible TVT must be aligned and MD must be finite and strictly increasing")
    known = np.flatnonzero(np.isfinite(visible))
    if not known.size or not np.array_equal(known, np.arange(known.size)):
        raise ValueError("a nonempty contiguous visible heel is required")
    if toe.shape != (len(md) - known.size,) or not np.isfinite(toe).all():
        raise ValueError("a finite toe prediction is required for every original toe row")
    if endpoints.ndim != 1 or (endpoints.size and endpoints.dtype.kind not in "iu"):
        raise ValueError("endpoint_station must contain integer station indices")
    if endpoints.size and (endpoints[0] < known.size or endpoints[-1] >= len(md)
                           or np.any(endpoints[1:] <= endpoints[:-1])):
        raise ValueError("station endpoints must be strictly increasing and inside the toe")
    if increments.shape != endpoints.shape or not np.isfinite(increments).all():
        raise ValueError("each endpoint requires one finite correction increment")

    if not endpoints.size:
        visible[known.size:] = toe
        return visible

    knots = np.r_[known[-1], endpoints.astype(np.intp, copy=False)]
    with np.errstate(over="ignore", invalid="ignore"):
        correction = np.cumsum(np.r_[0., increments])
        corrected = toe + np.interp(md[known.size:], md[knots], correction)
    if not np.isfinite(correction).all() or not np.isfinite(corrected).all():
        raise ValueError("cumulative correction or corrected TVT is not finite")
    visible[known.size:] = corrected
    return visible
