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
        out["ncc_peak"] = float(
            np.nanmax([out.get(f"ncc_{s}", 0.0) for s in scales])
        )
    else:
        out["tvt_hint_median"] = 0.0
        out["tvt_hint_std"] = 0.0
        out["ncc_peak"] = 0.0
    return out


def ncc_tvt_track(
    lateral_gr: np.ndarray,
    typewell_gr: np.ndarray,
    typewell_tvt: np.ndarray,
    *,
    known_tvt: np.ndarray | None = None,
    seed_tvt: float | None = None,
    stride: int = 8,
    scales: tuple[int, ...] = (32, 64),
    max_dev: float = 18.0,
    min_peak: float = 0.15,
) -> dict[str, np.ndarray]:
    """
    Deterministic typewell stalk: sparse multi-scale NCC TVT hints → filled track.

    Complements PF (stochastic). High ncc_peak → trust hint; else hold last-known.
    Always hard-clamped to seed/last ± max_dev (clamp law).
    """
    gr = np.asarray(lateral_gr, dtype=float)
    tw_gr = np.asarray(typewell_gr, dtype=float)
    tw_tvt = np.asarray(typewell_tvt, dtype=float)
    n = len(gr)
    if known_tvt is None:
        known_tvt = np.full(n, np.nan)
    else:
        known_tvt = np.asarray(known_tvt, dtype=float)

    last = float(seed_tvt) if seed_tvt is not None and np.isfinite(seed_tvt) else float("nan")
    if not np.isfinite(last) and np.isfinite(known_tvt).any():
        last = float(known_tvt[np.isfinite(known_tvt)][-1])
    if not np.isfinite(last) and np.isfinite(tw_tvt).any():
        last = float(np.nanmedian(tw_tvt))

    track = np.full(n, np.nan, dtype=float)
    conf = np.zeros(n, dtype=float)
    if n == 0 or not np.isfinite(last):
        return {
            "prior_ncc_tvt": np.full(n, np.nan, dtype=float),
            "ncc_track_conf": conf,
        }

    sample = list(range(0, n, max(1, int(stride))))
    if sample[-1] != n - 1:
        sample.append(n - 1)

    m_tw = np.isfinite(tw_gr) & np.isfinite(tw_tvt)
    if m_tw.sum() < 8:
        arr = np.full(n, last, dtype=float)
        arr = np.where(np.isfinite(known_tvt), known_tvt, arr)
        return {
            "prior_ncc_tvt": arr,
            "ncc_track_conf": np.where(np.isfinite(known_tvt), 1.0, conf).astype(float),
        }

    for i in sample:
        if np.isfinite(known_tvt[i]):
            track[i] = float(known_tvt[i])
            conf[i] = 1.0
            continue
        m = multi_scale_ncc(gr, tw_gr, tw_tvt, scales=scales, center=i)
        peak = float(m.get("ncc_peak") or 0.0)
        hint = m.get("tvt_hint_median")
        if hint is None or not np.isfinite(hint) or peak < float(min_peak):
            continue
        # clamp each sparse hint to seed ± max_dev
        track[i] = last + float(np.clip(float(hint) - last, -max_dev, max_dev))
        conf[i] = float(np.clip(peak, 0.0, 1.0))

    # dense track: linear interp of finite sparse samples; else hold seed
    idx = np.flatnonzero(np.isfinite(track))
    if len(idx) >= 2:
        filled = np.interp(np.arange(n, dtype=float), idx.astype(float), track[idx])
    elif len(idx) == 1:
        filled = np.full(n, float(track[idx[0]]), dtype=float)
    else:
        filled = np.full(n, last, dtype=float)

    # conf: nearest finite-sample confidence, attenuated off-sample
    conf_f = np.zeros(n, dtype=float)
    if len(idx) >= 1:
        # vectorized nearest sample index into idx
        # for each station i, j* = argmin |idx - i|
        pos = np.searchsorted(idx, np.arange(n))
        pos = np.clip(pos, 0, len(idx) - 1)
        pos_lo = np.clip(pos - 1, 0, len(idx) - 1)
        d_hi = np.abs(idx[pos] - np.arange(n))
        d_lo = np.abs(idx[pos_lo] - np.arange(n))
        nearest = np.where(d_lo <= d_hi, pos_lo, pos)
        conf_f = conf[idx[nearest]] * 0.85
        conf_f[idx] = conf[idx]  # exact samples keep full conf

    # clamp law + heel lock
    filled = last + np.clip(filled - last, -max_dev, max_dev)
    known_m = np.isfinite(known_tvt)
    filled = np.where(known_m, known_tvt, filled)
    conf_f = np.where(known_m, 1.0, conf_f)
    return {
        "prior_ncc_tvt": filled.astype(float),
        "ncc_track_conf": conf_f.astype(float),
    }
