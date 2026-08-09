"""
Curve hygiene: LAS nulls (see rwt.load), short-gap fill, gap-run features.

Policy (score-drop Wave A):
  - Sentinel nulls → NaN at load (never treat -999.25 as GR API).
  - Short gaps may be linearly filled for *shape* ops only, and tagged imputed.
  - Long gaps stay missing; adaptive residual refuses low conf / short lifetime.
"""
from __future__ import annotations

import numpy as np


def gap_run_lengths(valid: np.ndarray) -> np.ndarray:
    """
    Per-station length of the contiguous False-run (missing run) containing it.

    Valid stations get 0. Used as ``gr_gap_run`` feature / adaptive demotion.
    """
    ok = np.asarray(valid, dtype=bool)
    n = len(ok)
    out = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        if ok[i]:
            out[i] = 0.0
            i += 1
            continue
        j = i
        while j < n and not ok[j]:
            j += 1
        run = float(j - i)
        out[i:j] = run
        i = j
    return out


def conf_run_lifetime(conf: np.ndarray, threshold: float) -> np.ndarray:
    """
    Per-station length of the contiguous high-conf run containing it.

    Station fires under dual adaptive only if conf ≥ thr **and** lifetime ≥ L*
    (zigzag-inspired: long-lived structure may fire residual; short spikes refuse).
    """
    c = np.asarray(conf, dtype=float)
    thr = float(threshold)
    ok = np.isfinite(c) & (c >= thr)
    n = len(ok)
    out = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        if not ok[i]:
            out[i] = 0.0
            i += 1
            continue
        j = i
        while j < n and ok[j]:
            j += 1
        run = float(j - i)
        out[i:j] = run
        i = j
    return out


def fill_short_gaps(
    values: np.ndarray,
    *,
    max_gap: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linear-interp NaN runs of length ≤ ``max_gap`` only; longer runs stay NaN.

    Returns (filled, imputed_mask). Does not invent edges beyond finite endpoints
    (no extrapolation past first/last finite).
    """
    x = np.asarray(values, dtype=float).copy()
    n = len(x)
    imputed = np.zeros(n, dtype=bool)
    if n == 0:
        return x, imputed
    finite = np.isfinite(x)
    if finite.sum() < 2:
        return x, imputed
    max_gap = max(0, int(max_gap))
    if max_gap == 0:
        return x, imputed

    i = 0
    while i < n:
        if finite[i]:
            i += 1
            continue
        j = i
        while j < n and not finite[j]:
            j += 1
        gap = j - i
        # interior gap only, with finite neighbors, short enough
        left = i - 1
        right = j
        if (
            gap <= max_gap
            and left >= 0
            and right < n
            and finite[left]
            and finite[right]
        ):
            t = np.linspace(0.0, 1.0, gap + 2)[1:-1]
            x[i:j] = x[left] + t * (x[right] - x[left])
            imputed[i:j] = True
        i = j
    return x, imputed


def gr_gap_features(
    gr: np.ndarray,
    *,
    short_gap: int = 5,
    window: int = 21,
) -> dict[str, np.ndarray]:
    """
    GR missingness abstraction for residual X + diagnostics.

    - gr_is_null: 1 if raw GR non-finite
    - gr_gap_run: length of current missing run (0 if valid)
    - gr_valid_frac_win: fraction finite in centered window
    - gr_filled_short: short-gap linear fill (long gaps stay NaN)
    - gr_was_imputed: 1 where short-gap fill wrote a value
    """
    raw = np.asarray(gr, dtype=float)
    n = len(raw)
    valid = np.isfinite(raw)
    is_null = (~valid).astype(float)
    gap_run = gap_run_lengths(valid)
    filled, imputed = fill_short_gaps(raw, max_gap=short_gap)

    w = max(1, int(window))
    half = w // 2
    frac = np.zeros(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        frac[i] = float(np.mean(valid[lo:hi])) if hi > lo else 0.0

    return {
        "gr_is_null": is_null,
        "gr_gap_run": gap_run,
        "gr_valid_frac_win": frac,
        "gr_filled_short": filled,
        "gr_was_imputed": imputed.astype(float),
    }
