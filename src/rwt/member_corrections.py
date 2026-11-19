"""Apply keyed signed errors to ensemble members before station interpolation."""

from __future__ import annotations

import numpy as np

from rwt.ensemble import blend_inner_predictions


def blend_corrected_inner_predictions(block_index, full_tvt, members) -> np.ndarray:
    """Blend a full forecast with the mean of its individually corrected members.

    Each member supplies ``(prediction_ids, tvt, correction_ids, delta_tvt)``.
    The correction is a signed TVT adjustment in feet with any ramp or gain
    already applied by the caller. Both key sequences must match the full model
    exactly. Keep the returned block keys when lifting to original stations.
    The caller supplies the correct model and correction pairing and verifies
    their training roles.
    """
    def corrected():
        for prediction_ids, tvt, correction_ids, delta_tvt in members:
            ids = np.asarray(prediction_ids)
            correction_keys = np.asarray(correction_ids)
            values = np.asarray(tvt, dtype=np.float64)
            delta = np.asarray(delta_tvt, dtype=np.float64)
            if (correction_keys.dtype.kind not in "iu" or not np.array_equal(ids, correction_keys)
                    or delta.shape != ids.shape or values.shape != ids.shape):
                raise ValueError("each correction must align to its member's original block IDs")
            if not np.isfinite(delta).all():
                raise ValueError("member corrections must be finite")
            with np.errstate(over="ignore", invalid="ignore"):
                corrected_tvt = values + delta
            yield ids, corrected_tvt

    return blend_inner_predictions(block_index, full_tvt, corrected())
