"""
GR shape matching to typewell — geosteering core (not TSS peptide math).

Multi-scale normalized cross-correlation of lateral GR windows vs typewell GR.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _zscore(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    m = np.nanmean(a)
    s = np.nanstd(a)
    if not np.isfinite(s) or s < 1e-9:
        return np.zeros_like(a)
    return (a - m) / s


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation at lag 0 (equal length)."""
    a = _zscore(a)
    b = _zscore(b)
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    return float(np.dot(a[:n], b[:n]) / max(n, 1))


def best_ncc_lag(query: np.ndarray, template: np.ndarray, max_lag: int | None = None) -> dict[str, Any]:
    """Slide query along template; return best NCC and lag."""
    q = _zscore(np.asarray(query, float))
    t = _zscore(np.asarray(template, float))
    nq, nt = len(q), len(t)
    if nq < 3 or nt < nq:
        return {"ncc": 0.0, "lag": 0, "tvt_index": 0}
    max_lag = max_lag if max_lag is not None else nt - nq
    max_lag = max(0, min(max_lag, nt - nq))
    best = -1e9
    best_lag = 0
    for lag in range(0, max_lag + 1):
        window = t[lag : lag + nq]
        score = float(np.dot(q, window) / max(nq, 1))
        if score > best:
            best = score
            best_lag = lag
    return {"ncc": float(best), "lag": int(best_lag), "tvt_index": int(best_lag)}


def multi_scale_ncc(
    lateral_gr: np.ndarray,
    typewell_gr: np.ndarray,
    typewell_tvt: np.ndarray | None = None,
    scales: tuple[int, ...] = (16, 32, 64),
    center: int | None = None,
) -> dict[str, float]:
    """
    At station `center`, extract lateral windows of each scale and match typewell.
    Returns features + optional TVT hint from best lag.
    """
    gr = np.asarray(lateral_gr, float)
    tw = np.asarray(typewell_gr, float)
    center = int(center if center is not None else len(gr) // 2)
    out: dict[str, float] = {}
    tvt_hints = []
    for s in scales:
        half = s // 2
        lo = max(0, center - half)
        hi = min(len(gr), center + half)
        window = gr[lo:hi]
        if len(window) < max(4, s // 4):
            out[f"ncc_{s}"] = 0.0
            out[f"ncc_{s}_lag"] = 0.0
            continue
        # resample window to fixed s for stable NCC
        if len(window) != s:
            xp = np.linspace(0, 1, len(window))
            xq = np.linspace(0, 1, s)
            window = np.interp(xq, xp, window)
        r = best_ncc_lag(window, tw)
        out[f"ncc_{s}"] = r["ncc"]
        out[f"ncc_{s}_lag"] = float(r["lag"])
        if typewell_tvt is not None and len(typewell_tvt) > r["lag"]:
            tvt_hints.append(float(typewell_tvt[min(r["lag"] + s // 2, len(typewell_tvt) - 1)]))
    if tvt_hints:
        out["tvt_hint_median"] = float(np.median(tvt_hints))
        out["tvt_hint_std"] = float(np.std(tvt_hints))
    else:
        out["tvt_hint_median"] = 0.0
        out["tvt_hint_std"] = 0.0
    return out
