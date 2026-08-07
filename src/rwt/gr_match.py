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
    """
    Slide query along template; return best NCC and lag.

    Vectorized via np.correlate on float64 contiguous buffers (OpenBLAS/SIMD).
    """
    from rwt.simd import as_f64_contig, vectorized_ncc_template

    q = as_f64_contig(np.asarray(query, float))
    t = as_f64_contig(np.asarray(template, float))
    nq, nt = len(q), len(t)
    if nq < 3 or nt < nq:
        return {"ncc": 0.0, "lag": 0, "tvt_index": 0}
    max_lag = max_lag if max_lag is not None else nt - nq
    max_lag = max(0, min(max_lag, nt - nq))
    # restrict template to searchable region
    t_use = t[: nq + max_lag] if max_lag + nq <= nt else t
    ncc_v, lag = vectorized_ncc_template(q, t_use)
    lag = int(min(lag, max_lag))
    return {"ncc": float(ncc_v), "lag": lag, "tvt_index": lag}


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
    tw_finite = tw[np.isfinite(tw)]
    center = int(center if center is not None else len(gr) // 2)
    out: dict[str, float] = {}
    tvt_hints = []
    for s in scales:
        half = s // 2
        lo = max(0, center - half)
        hi = min(len(gr), center + half)
        window = gr[lo:hi]
        finite = np.isfinite(window)
        # Missing GR is not measured zero: require enough finite samples
        if finite.sum() < max(4, s // 4) or len(tw_finite) < 3:
            out[f"ncc_{s}"] = 0.0
            out[f"ncc_{s}_lag"] = 0.0
            continue
        if finite.all():
            win = window.astype(float)
        else:
            idx = np.arange(len(window), dtype=float)
            win = np.interp(idx, idx[finite], window[finite])
        if len(win) != s:
            xp = np.linspace(0, 1, len(win))
            xq = np.linspace(0, 1, s)
            win = np.interp(xq, xp, win)
        # Template: median-fill internal NaNs so lag search stays defined without zero-impute of lateral
        med = float(np.median(tw_finite))
        tw_f = np.where(np.isfinite(tw), tw, med)
        r = best_ncc_lag(win, tw_f)
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
