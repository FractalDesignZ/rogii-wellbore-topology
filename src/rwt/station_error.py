"""Evaluation-only error decomposition at the original stations of one well."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class StationError(NamedTuple):
    """Additive squared errors in ft², with signed offset and drift."""

    rows: int
    mean_error_ft: float
    drift_ft_per_md_ft: float
    offset_sse_ft2: float
    linear_sse_ft2: float
    residual_sse_ft2: float


def decompose_station_error(md, prediction, truth) -> StationError:
    """Project prediction-minus-truth onto a constant and centered measured depth.

    Use once per well and only after its predictions have been frozen. Each
    original row receives equal weight. Summing the three squared terms
    recovers station SSE. Sum energies across wells before dividing by the
    total row count or taking a square root.
    """
    md, prediction, truth = (np.asarray(value, dtype=float) for value in (md, prediction, truth))
    if md.ndim != 1 or not len(md) or md.shape != prediction.shape or md.shape != truth.shape:
        raise ValueError("require aligned, nonempty measured-depth, prediction and truth vectors")
    if not all(np.isfinite(value).all() for value in (md, prediction, truth)):
        raise ValueError("station error requires complete finite support")
    if np.any(np.diff(md) <= 0):
        raise ValueError("measured depth must increase within one well")

    error = prediction - truth
    offset = error.mean()
    centered = error - offset
    trend = np.zeros_like(centered)
    slope = 0.
    if len(md) > 1:
        distance = md[-1] - md[0]
        phase = (md - md[0]) / distance
        phase -= phase.mean()
        phase_coefficient = np.dot(phase, centered) / np.dot(phase, phase)
        trend = phase * phase_coefficient
        slope = float(phase_coefficient / distance)
    remaining = centered - trend
    return StationError(len(md), float(offset), slope, float(len(md) * offset**2),
                        float(np.dot(trend, trend)), float(np.dot(remaining, remaining)))
