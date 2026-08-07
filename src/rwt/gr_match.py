"""
GR shape matching to typewell — geosteering core (not TSS peptide math).

Multi-scale normalized cross-correlation of lateral GR windows vs typewell GR.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from rwt.priors import MAX_TVT_DEV


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
    *,
    seed_tvt: float | None = None,
    tvt_search_radius: float | None = None,
) -> dict[str, float]:
    """
    At station `center`, extract lateral windows of each scale and match typewell.
    Returns features + optional TVT hint from best lag.

    When ``seed_tvt`` + ``tvt_search_radius`` are set, lag search is restricted to
    typewell indices whose TVT lies in [seed−r, seed+r] (plus a half-window
    pad). Full-well lag search aliases on cyclic GR and produced pure NCC RMSE
    ~22 on 12w; seed-local search keeps the stalk on-formation.

    Seed-local band uses ``in_band[0]``/``in_band[-1]`` as a contiguous index
    range, so ``typewell_tvt`` must be monotonically increasing by index.
    Callers must pass TVT-sorted typewell rows (same order as ``typewell_gr``).
    """
    gr = np.asarray(lateral_gr, float)
    tw = np.asarray(typewell_gr, float)
    tw_tvt_arr = np.asarray(typewell_tvt, float) if typewell_tvt is not None else None
    tw_finite = tw[np.isfinite(tw)]
    center = int(center if center is not None else len(gr) // 2)
    out: dict[str, float] = {}
    tvt_hints = []
    peaks: list[float] = []
    med = float(np.median(tw_finite)) if len(tw_finite) else 0.0
    tw_f = np.where(np.isfinite(tw), tw, med)

    # Optional seed-local template band (indices into full typewell).
    # Expand radius if seed sits off-grid / sparse so we never silently fall
    # back to full-well lag (cyclic GR alias).
    band_lo, band_hi = 0, len(tw_f)
    if (
        tw_tvt_arr is not None
        and seed_tvt is not None
        and np.isfinite(seed_tvt)
        and tvt_search_radius is not None
        and float(tvt_search_radius) > 0
        and len(tw_tvt_arr) == len(tw_f)
    ):
        r_tvt = float(tvt_search_radius)
        in_band = np.array([], dtype=int)
        for _ in range(6):
            in_band = np.flatnonzero(
                np.isfinite(tw_tvt_arr)
                & (tw_tvt_arr >= float(seed_tvt) - r_tvt)
                & (tw_tvt_arr <= float(seed_tvt) + r_tvt)
            )
            if in_band.size >= 4:
                band_lo = int(in_band[0])
                band_hi = int(in_band[-1]) + 1
                break
            r_tvt *= 1.5
        # last resort: nearest-index window (~2·original radius in samples)
        if in_band.size < 4 and np.isfinite(tw_tvt_arr).any():
            i0 = int(np.nanargmin(np.abs(tw_tvt_arr - float(seed_tvt))))
            half_n = max(4, int(len(tw_f) * 0.08))
            band_lo = max(0, i0 - half_n)
            band_hi = min(len(tw_f), i0 + half_n + 1)

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
        # pad band so lag search has room for window length s
        t_lo = max(0, band_lo - half)
        t_hi = min(len(tw_f), max(band_hi + half, t_lo + s))
        tw_slice = tw_f[t_lo:t_hi]
        if len(tw_slice) < s:
            tw_slice = tw_f
            t_lo = 0
        r = best_ncc_lag(win, tw_slice)
        lag_global = int(t_lo + r["lag"])
        out[f"ncc_{s}"] = r["ncc"]
        out[f"ncc_{s}_lag"] = float(lag_global)
        peaks.append(float(r["ncc"]))
        if tw_tvt_arr is not None and len(tw_tvt_arr) > lag_global:
            tip = min(lag_global + half, len(tw_tvt_arr) - 1)
            h = float(tw_tvt_arr[tip])
            if np.isfinite(h):
                tvt_hints.append(h)
    if tvt_hints:
        out["tvt_hint_median"] = float(np.median(tvt_hints))
        out["tvt_hint_std"] = float(np.std(tvt_hints))
        out["ncc_peak"] = float(np.nanmax(peaks)) if peaks else 0.0
    else:
        out["tvt_hint_median"] = 0.0
        out["tvt_hint_std"] = 0.0
        out["ncc_peak"] = float(np.nanmax(peaks)) if peaks else 0.0
    return out


def typewell_gr_at_tvt(
    typewell_gr: np.ndarray,
    typewell_tvt: np.ndarray,
    query_tvt: np.ndarray,
    *,
    lateral_gr: np.ndarray | None = None,
    known_tvt: np.ndarray | None = None,
) -> np.ndarray:
    """
    Typewell GR interpolated at hypothesized lateral TVT (optionally heel-calibrated).

    When known heel TVT + lateral GR are provided (≥8 points), fits affine
    map typewell→lateral GR (same spirit as PF stalk) so mismatch is in
    lateral GR units.
    """
    tw_gr = np.asarray(typewell_gr, dtype=float)
    tw_tvt = np.asarray(typewell_tvt, dtype=float)
    q = np.asarray(query_tvt, dtype=float)
    m = np.isfinite(tw_gr) & np.isfinite(tw_tvt)
    if m.sum() < 4:
        return np.full(len(q), np.nan, dtype=float)
    order = np.argsort(tw_tvt[m])
    tw_t = tw_tvt[m][order]
    tw_g = tw_gr[m][order]
    # unique TVT for interp
    if len(tw_t) >= 2:
        uniq = np.concatenate([[True], np.diff(tw_t) > 1e-12])
        tw_t, tw_g = tw_t[uniq], tw_g[uniq]
    a_cal, b_cal = 1.0, 0.0
    if (
        lateral_gr is not None
        and known_tvt is not None
        and len(lateral_gr) == len(known_tvt)
    ):
        lat = np.asarray(lateral_gr, dtype=float)
        kn = np.asarray(known_tvt, dtype=float)
        km = np.isfinite(lat) & np.isfinite(kn)
        if km.sum() >= 8:
            tw_k = np.interp(kn[km], tw_t, tw_g)
            A = np.column_stack([tw_k, np.ones(int(km.sum()))])
            try:
                coef, _, _, _ = np.linalg.lstsq(A, lat[km], rcond=None)
                a0, b0 = float(coef[0]), float(coef[1])
                if np.isfinite(a0) and a0 > 1e-3 and np.isfinite(b0):
                    a_cal, b_cal = a0, b0
            except np.linalg.LinAlgError:
                pass
    tw_g = a_cal * tw_g + b_cal
    out = np.full(len(q), np.nan, dtype=float)
    qm = np.isfinite(q)
    if qm.any() and len(tw_t) >= 2:
        out[qm] = np.interp(q[qm], tw_t, tw_g, left=tw_g[0], right=tw_g[-1])
    elif qm.any() and len(tw_t) == 1:
        out[qm] = tw_g[0]
    return out


def gr_mismatch_at_tvt(
    lateral_gr: np.ndarray,
    typewell_gr: np.ndarray,
    typewell_tvt: np.ndarray,
    query_tvt: np.ndarray,
    *,
    known_tvt: np.ndarray | None = None,
) -> np.ndarray:
    """Lateral GR − calibrated typewell GR at hypothesized TVT (residual signal)."""
    lat = np.asarray(lateral_gr, dtype=float)
    hyp = typewell_gr_at_tvt(
        typewell_gr,
        typewell_tvt,
        query_tvt,
        lateral_gr=lat,
        known_tvt=known_tvt,
    )
    out = lat - hyp
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
    max_dev: float = MAX_TVT_DEV,
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

    # Clamp law is vs original seed (heel last-known), not progressive walk.
    clamp_origin = float(last)

    # Typewell must be TVT-sorted for lag index ↔ TVT and seed-local band.
    m_tw = np.isfinite(tw_gr) & np.isfinite(tw_tvt)
    if m_tw.sum() < 8:
        arr = np.full(n, clamp_origin, dtype=float)
        arr = np.where(np.isfinite(known_tvt), known_tvt, arr)
        return {
            "prior_ncc_tvt": arr,
            "ncc_track_conf": np.where(np.isfinite(known_tvt), 1.0, conf).astype(float),
        }
    order = np.argsort(tw_tvt[m_tw])
    tw_gr = tw_gr[m_tw][order]
    tw_tvt = tw_tvt[m_tw][order]

    sample = list(range(0, n, max(1, int(stride))))
    if sample[-1] != n - 1:
        sample.append(n - 1)

    for i in sample:
        if np.isfinite(known_tvt[i]):
            track[i] = float(known_tvt[i])
            # walk progressive last for next sparse clamp; do NOT set conf=1
            # here — heel conf must not bleed into toe via nearest-fill
            last = float(known_tvt[i])
            continue
        # seed-local lag search (± max_dev·1.25) — full-well lag aliases on cyclic GR
        m = multi_scale_ncc(
            gr,
            tw_gr,
            tw_tvt,
            scales=scales,
            center=i,
            seed_tvt=last,
            tvt_search_radius=float(max_dev) * 1.25,
        )
        peak = float(m.get("ncc_peak") or 0.0)
        hint = m.get("tvt_hint_median")
        if hint is None or not np.isfinite(hint) or peak < float(min_peak):
            continue
        # clamp each sparse hint to progressive last ± max_dev
        track[i] = last + float(np.clip(float(hint) - last, -max_dev, max_dev))
        conf[i] = float(np.clip(peak, 0.0, 1.0))

    # dense track: linear interp of finite sparse samples; else hold clamp origin
    idx = np.flatnonzero(np.isfinite(track))
    if len(idx) >= 2:
        filled = np.interp(np.arange(n, dtype=float), idx.astype(float), track[idx])
    elif len(idx) == 1:
        filled = np.full(n, float(track[idx[0]]), dtype=float)
    else:
        filled = np.full(n, clamp_origin, dtype=float)

    # conf: nearest *NCC-peak* sample only (never heel-known conf → cold toe)
    conf_f = np.zeros(n, dtype=float)
    peak_idx = np.flatnonzero(conf > 0.0)
    if len(peak_idx) >= 1:
        pos = np.searchsorted(peak_idx, np.arange(n))
        pos = np.clip(pos, 0, len(peak_idx) - 1)
        pos_lo = np.clip(pos - 1, 0, len(peak_idx) - 1)
        d_hi = np.abs(peak_idx[pos] - np.arange(n))
        d_lo = np.abs(peak_idx[pos_lo] - np.arange(n))
        nearest = np.where(d_lo <= d_hi, pos_lo, pos)
        conf_f = conf[peak_idx[nearest]] * 0.85
        conf_f[peak_idx] = conf[peak_idx]

    # clamp law vs seed + heel lock (known conf = 1 only on known stations)
    filled = clamp_origin + np.clip(filled - clamp_origin, -max_dev, max_dev)
    known_m = np.isfinite(known_tvt)
    filled = np.where(known_m, known_tvt, filled)
    conf_f = np.where(known_m, 1.0, conf_f)
    return {
        "prior_ncc_tvt": filled.astype(float),
        "ncc_track_conf": conf_f.astype(float),
    }
