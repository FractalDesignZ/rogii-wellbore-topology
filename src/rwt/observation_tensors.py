"""Pack inbound well observations and gather measured-depth relationships."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from rwt.block_model import INPUT_COLUMNS
from rwt.load import GR_ABSURD_LE, normalize_curve_nulls


class ObservationTensor(NamedTuple):
    """Parsed values and validity indexed by original station and channel."""

    well_id: str
    table: str
    channels: tuple[str, ...]
    units: tuple[str, ...]
    values: np.ndarray
    valid: np.ndarray

    def to_structured(self) -> np.ndarray:
        """Return one typed tuple per source row with named value and presence fields."""
        dtype = [("source_station", np.int64)]
        dtype.extend((name, np.float64) for name in self.channels)
        dtype.extend((f"{name}__present", np.bool_) for name in self.channels)
        rows = np.empty(len(self.values), dtype=dtype)
        rows["source_station"] = np.arange(len(rows), dtype=np.int64)
        for i, name in enumerate(self.channels):
            rows[name] = self.values[:, i]
            rows[f"{name}__present"] = self.valid[:, i]
        return rows


class WellObservations(NamedTuple):
    lateral: ObservationTensor
    typewell: ObservationTensor


class NeighborTensor(NamedTuple):
    """Values indexed by receiver row, requested MD offset and channel."""

    well_id: str
    channels: tuple[str, ...]
    units: tuple[str, ...]
    center_station: np.ndarray
    requested_offset_ft: np.ndarray
    source_station: np.ndarray
    actual_offset_ft: np.ndarray
    values: np.ndarray
    valid: np.ndarray


def _pack(well_id, table, frame, channels, units):
    if not frame.columns.is_unique or not set(channels).issubset(frame.columns):
        raise ValueError(f"{table} requires one column for each inbound channel")
    values = frame.loc[:, list(channels)].to_numpy(dtype=np.float64, copy=True)
    valid = np.isfinite(values)
    for name in ("GR", "TVT_input", "TVT"):
        if name in channels:
            idx = channels.index(name)
            valid[:, idx] = np.isfinite(normalize_curve_nulls(
                values[:, idx], absurd_le=GR_ABSURD_LE if name == "GR" else None))
    values.flags.writeable = valid.flags.writeable = False
    return ObservationTensor(well_id, table, channels, units, values, valid)


def pack_well_observations(well_id: str, lateral: pd.DataFrame, typewell: pd.DataFrame) -> WellObservations:
    """Copy permitted parsed observations in each full input table's original row order.

    A value is addressed by (well ID, table, zero-based source row, channel).
    Preserve the parsed number even when a curve null makes its mask false.
    The two tables use separate row axes. Callers authenticate source identities
    and read roles before packing. The lateral includes only visible TVT_input.
    """
    if not isinstance(well_id, str) or not well_id:
        raise ValueError("a nonempty source well ID is required")
    return WellObservations(
        _pack(well_id, "lateral", lateral, INPUT_COLUMNS, ("ft", "ft", "ft", "ft", "raw GR", "ft")),
        _pack(well_id, "typewell", typewell, ("TVT", "GR"), ("ft", "raw GR")),
    )


def gather_md_neighbors(lateral: ObservationTensor, center_station, offsets_ft, *, tolerance_ft: float) -> NeighborTensor:
    """Gather one nearest original row per receiver and signed MD offset.

    Return the sender index and its actual MD offset along with source values.
    At an exact tie choose the earlier row. Unavailable relationships use
    sender -1, NaN offsets and an all-false validity mask. A found row can still
    contain a missing individual channel. No interpolation is performed.
    """
    if lateral.table != "lateral" or lateral.channels != INPUT_COLUMNS:
        raise ValueError("MD neighbors require a packed lateral")
    md = lateral.values[:, 0]
    if not len(md) or not np.isfinite(md).all() or np.any(md[1:] <= md[:-1]):
        raise ValueError("measured depth must be finite and strictly increasing")
    center = np.asarray(center_station)
    offsets = np.asarray(offsets_ft, dtype=np.float64)
    if (center.ndim != 1 or (center.size and center.dtype.kind not in "iu")
            or np.any(center < 0) or np.any(center >= len(md))):
        raise ValueError("center_station must contain original lateral row IDs")
    if offsets.ndim != 1 or not np.isfinite(offsets).all() or len(np.unique(offsets)) != len(offsets):
        raise ValueError("offsets must be a finite vector with distinct values")
    if isinstance(tolerance_ft, bool) or not np.isfinite(tolerance_ft) or tolerance_ft < 0:
        raise ValueError("tolerance_ft must be finite and nonnegative")
    center = center.astype(np.intp, copy=True)
    with np.errstate(over="ignore", invalid="ignore"):
        wanted = md[center, None] + offsets[None, :]
        right = np.searchsorted(md, wanted).clip(0, len(md) - 1)
        left = np.maximum(0, right - 1)
        source = np.where(np.abs(md[left] - wanted) <= np.abs(md[right] - wanted), left, right)
        supported = (wanted >= md[0]) & (wanted <= md[-1]) & (np.abs(md[source] - wanted) <= tolerance_ft)
        actual = np.where(supported, md[source] - md[center, None], np.nan)
    return NeighborTensor(
        lateral.well_id, lateral.channels, lateral.units, center, offsets.copy(), np.where(supported, source, -1), actual,
        np.where(supported[..., None], lateral.values[source], np.nan), supported[..., None] & lateral.valid[source],
    )
