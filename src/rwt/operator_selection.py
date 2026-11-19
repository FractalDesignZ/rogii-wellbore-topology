"""Choose an operator on an explicitly excluded, whole-well training bank."""
from __future__ import annotations

import math
import numpy as np


def _well_set(values):
    items = tuple(values)
    if any(not isinstance(item, str) or not item or item != item.strip() for item in items):
        raise ValueError("well IDs must be nonempty strings without surrounding whitespace")
    ids = set(items)
    if len({item.casefold() for item in items}) != len(items):
        raise ValueError("duplicate or case-aliased well ID in selection roles")
    return ids


def select_supported_operator(outer_train, outer_held, bank_wells, well_ids,
                              squared_errors, row_counts, regimes, query_regime,
                              *, min_wells=12):
    """Return 0 (OFF), 1 or 2 from paired per-well sums of squared native error.

    Each observed bank well must have one row with three SSEs on its same
    positive number of stations. Regime -1 denotes no evidence and always
    returns OFF. In a supported regime select the unique lowest pooled SSE
    only if its improvement exceeds one whole-well standard error of the sum.
    The minimum bank size and one-standard-error rule are study choices.
    The caller authenticates station keys and the model's exclusion of the bank.
    """
    ids = tuple(well_ids)
    train, held = _well_set(outer_train), _well_set(outer_held)
    bank = _well_set(bank_wells)
    if ({w.casefold() for w in train} & {w.casefold() for w in held}
            or not train or not held or not bank <= train or _well_set(ids) != bank):
        raise ValueError("bank membership or outer roles disagree")
    errors = np.asarray(squared_errors, dtype=np.float64)
    count = np.asarray(row_counts)
    region = np.asarray(regimes)
    n = len(ids)
    if (errors.shape != (n, 3) or not np.isfinite(errors).all() or (errors < 0).any()
            or count.shape != (n,) or count.dtype.kind not in "iu" or (count == 0).any()
            or (count < 0).any() or region.shape != (n,) or region.dtype.kind not in "iu"
            or (region < -1).any()):
        raise ValueError("invalid per-well error, count or support")
    if (not isinstance(query_regime, (int, np.integer)) or isinstance(query_regime, (bool, np.bool_))
            or query_regime < -1 or type(min_wells) is not int or min_wells < 2):
        raise ValueError("invalid query support or minimum bank size")
    group = errors[region == query_regime]
    if query_regime == -1 or len(group) < min_wells:
        return 0
    largest_error = group.max()
    if largest_error > np.finfo(np.float64).max / len(group):
        pooled = (group / largest_error).sum(axis=0)
    else:
        pooled = group.sum(axis=0)
    winners = np.flatnonzero(pooled == pooled.min())
    if len(winners) != 1 or winners[0] == 0:
        return 0
    best = int(winners[0])
    gains = group[:, 0] - group[:, best]
    scale = np.max(np.abs(gains))
    if scale == 0:
        return 0
    # For a positive sum, exceeding one SE of that sum is equivalent to
    # sum(gains) > ||gains||₂. Scaling keeps both sides finite.
    scaled = gains / scale
    return best if math.fsum(scaled) > math.hypot(*scaled) else 0
