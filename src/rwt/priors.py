"""
Shared residual prior contract + three-layer structural priors.

Layer 1 — Flat: C = mean(TVT_heel + Z_heel); prior = C − Z
Layer 2 — Dip: Ridge(X,Y) on heel residuals of Layer 1; project to toe
Layer 3 — Residual: external (LGBM) only if Gate 3 accepts vs last_known

Also: PF bagging blend columns produced in particle_track / features.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rwt.middle_out import MiddleOutSplit

# Residual base: clamped stack (last_known + soft PF/dip) is best pure floor
# after hard-clamp math (12w: stack ~11.8 < last ~13.4). Residual must beat it.
DEFAULT_PRIOR_COL = "prior_stack"
PRIOR_FALLBACKS = (
    "prior_stack",
    "prior_pf_blend",
    "prior_last_tvt",
    "prior_local_dip",
    "prior_pf_bag",
    "prior_dip_stack",
    "prior_mid",
    "prior_flat_tvt",
)

# Train-only geological surfaces — never features (test has no ANCC..BUDA).
FORM_SURFACE_LEAK_COLS = (
    "ANCC",
    "ASTNU",
    "ASTNL",
    "EGFDU",
    "EGFDL",
    "BUDA",
)


def resolve_prior_column(
    df: pd.DataFrame,
    preferred: str = DEFAULT_PRIOR_COL,
    *,
    strict: bool = False,
) -> str:
    """
    Resolve residual prior column name.

    strict=True: preferred must exist (train/serve contract — no silent swap).
    strict=False: fall back through PRIOR_FALLBACKS for exploratory paths.
    """
    if preferred in df.columns:
        return preferred
    if strict:
        raise KeyError(
            f"strict prior column '{preferred}' missing; have {list(df.columns)[:24]}"
        )
    for c in PRIOR_FALLBACKS:
        if c in df.columns:
            return c
    raise KeyError(f"No residual prior column in {list(df.columns)[:24]}")


def _heel_tvt_values(
    df: pd.DataFrame,
    known_idx: np.ndarray,
    *,
    tvt_col: str = "TVT",
    tvt_input_col: str = "TVT_input",
) -> np.ndarray:
    """TVT on known heel: prefer TVT_input, else TVT."""
    k = np.asarray(known_idx, dtype=int)
    out = np.full(len(k), np.nan, dtype=float)
    if tvt_input_col in df.columns:
        out = np.asarray(df.iloc[k][tvt_input_col].to_numpy(dtype=float), dtype=float).copy()
    if tvt_col in df.columns:
        t = np.asarray(df.iloc[k][tvt_col].to_numpy(dtype=float), dtype=float)
        miss = ~np.isfinite(out)
        out[miss] = t[miss]
    return out


def compute_local_dip_prior(
    df: pd.DataFrame,
    split: MiddleOutSplit,
    *,
    heel_window_ft: float = 300.0,
    md_col: str = "MD",
    z_col: str = "Z",
    tvt_col: str = "TVT",
    tvt_input_col: str = "TVT_input",
    max_md_slope: float = 0.008,
) -> dict[str, Any]:
    """
    M2 PRIOR / M4 MIDDLE_OUT: last-known PS anchor + *tiny* MD-rate only.

    Horizontal laterals: raw ΔTVT/ΔZ is poison (diag RMSE 50–150). Heel MD-slope
    also over-extrapolates. Math:
      prior = ps_TVT + clip(ΔTVT/ΔMD, ±max_md_slope) · (MD − ps_MD)
    with max_md_slope ≈ 0.008 (≤ 8 ft TVT per 1000 ft MD). That is a soft
    trend around the last-known floor — residual + PF own the rest.

    Heel locks to observed TVT. Empty heel → relative anchors on first finite MD.
    """
    n = len(df)
    md = (
        df[md_col].to_numpy(dtype=float)
        if md_col in df.columns
        else np.arange(n, dtype=float)
    )
    z = df[z_col].to_numpy(dtype=float) if z_col in df.columns else np.zeros(n)
    k = np.asarray(split.known_idx, dtype=int)

    def _relative_from_anchor(ps_md: float, ps_z: float) -> tuple[np.ndarray, np.ndarray]:
        return (md - ps_md).astype(float), (z - ps_z).astype(float)

    # Label-free anchor when no known heel: first finite MD station
    fin_md = np.isfinite(md)
    if fin_md.any():
        a0 = int(np.flatnonzero(fin_md)[0])
        rel_md0, rel_z0 = _relative_from_anchor(float(md[a0]), float(z[a0]) if np.isfinite(z[a0]) else 0.0)
    else:
        rel_md0 = np.full(n, np.nan, dtype=float)
        rel_z0 = np.full(n, np.nan, dtype=float)

    empty = {
        "prior_local_dip": np.full(n, np.nan, dtype=float),
        "local_dip": float("nan"),
        "md_slope": float("nan"),
        "ps_md": float("nan"),
        "ps_z": float("nan"),
        "ps_tvt": float("nan"),
        "relative_md": rel_md0,
        "relative_z": rel_z0,
    }
    if len(k) == 0:
        return empty

    order = np.argsort(md[k])
    k_ord = k[order]
    heel_tvt = _heel_tvt_values(
        df, k_ord, tvt_col=tvt_col, tvt_input_col=tvt_input_col
    )
    fin = np.isfinite(heel_tvt) & np.isfinite(z[k_ord]) & np.isfinite(md[k_ord])
    if not fin.any():
        return empty

    k_ord = k_ord[fin]
    heel_tvt = heel_tvt[fin]
    heel_z = z[k_ord]
    heel_md = md[k_ord]

    ps_tvt = float(heel_tvt[-1])
    ps_z = float(heel_z[-1])
    ps_md = float(heel_md[-1])

    md_lo = ps_md - float(heel_window_ft)
    hist = heel_md >= md_lo
    z_dip = 0.0  # never project ΔTVT/ΔZ on laterals
    if hist.sum() < 8:
        md_slope = 0.0
    else:
        md_h = heel_md[hist]
        tv_h = heel_tvt[hist]
        dmd = float(md_h[-1] - md_h[0])
        dt = float(tv_h[-1] - tv_h[0])
        if abs(dmd) > 1.0:
            md_slope = float(np.clip(dt / dmd, -max_md_slope, max_md_slope))
        else:
            md_slope = 0.0

    prior = ps_tvt + md_slope * (md - ps_md)
    # lock heel to observed
    prior[k_ord] = heel_tvt
    if len(split.known_idx):
        kk = np.asarray(split.known_idx, dtype=int)
        ht = _heel_tvt_values(df, kk, tvt_col=tvt_col, tvt_input_col=tvt_input_col)
        m = np.isfinite(ht)
        prior[kk[m]] = ht[m]

    rel_md, rel_z = _relative_from_anchor(ps_md, ps_z)
    return {
        "prior_local_dip": prior.astype(float),
        "local_dip": float(z_dip),
        "md_slope": float(md_slope),
        "ps_md": ps_md,
        "ps_z": ps_z,
        "ps_tvt": ps_tvt,
        "relative_md": rel_md,
        "relative_z": rel_z,
    }


def compute_heel_fitted_dip_prior(
    df: pd.DataFrame,
    split: MiddleOutSplit,
    *,
    alpha: float = 50.0,
    tvt_col: str = "TVT",
    tvt_input_col: str = "TVT_input",
    z_col: str = "Z",
    lateral_window: int = 400,
    md_col: str = "MD",
) -> dict[str, Any]:
    """
    Layers 1+2: flat C + XY dip on **near-PS lateral heel only** (not the build).

    Fitting on full heel mixes build dZ into the plane → toe RMSE 50–200.
    Use last ``lateral_window`` known stations (≈ last N ft of lateral at PS).
    Strong ridge (alpha) shrinks dip toward flat C−Z.
    """
    n = len(df)
    z = df[z_col].to_numpy(dtype=float) if z_col in df.columns else np.zeros(n)
    x = df["X"].to_numpy(dtype=float) if "X" in df.columns else np.arange(n, dtype=float)
    y = df["Y"].to_numpy(dtype=float) if "Y" in df.columns else np.zeros(n)
    md = df[md_col].to_numpy(dtype=float) if md_col in df.columns else np.arange(n, dtype=float)

    k = np.asarray(split.known_idx, dtype=int)

    def _const_fallback(last: float) -> dict[str, Any]:
        if not np.isfinite(last):
            last = float("nan")
        arr = np.full(n, last if np.isfinite(last) else np.nan, dtype=float)
        return {
            "prior_flat_tvt": arr.copy(),
            "prior_dip_trend": np.zeros(n),
            "prior_dip_stack": arr,
            "C_heel": float(last) if np.isfinite(last) else float("nan"),
            "dip_coef_x": 0.0,
            "dip_coef_y": 0.0,
            "dip_intercept": 0.0,
        }

    if len(k) < 3:
        ht = _heel_tvt_values(df, k, tvt_col=tvt_col, tvt_input_col=tvt_input_col)
        last = float(ht[np.isfinite(ht)][-1]) if np.isfinite(ht).any() else float("nan")
        return _const_fallback(last)

    # MD-sorted known; keep only near-PS lateral window
    order = np.argsort(md[k])
    k_ord = k[order]
    k_fit = k_ord[-min(int(lateral_window), len(k_ord)) :]

    heel_tvt = _heel_tvt_values(df, k_fit, tvt_col=tvt_col, tvt_input_col=tvt_input_col)
    heel_z = z[k_fit]
    m = np.isfinite(heel_tvt) & np.isfinite(heel_z)
    if m.sum() < 3:
        ht_all = _heel_tvt_values(df, k, tvt_col=tvt_col, tvt_input_col=tvt_input_col)
        last = float(ht_all[np.isfinite(ht_all)][-1]) if np.isfinite(ht_all).any() else float("nan")
        return _const_fallback(last)

    C_heel = float(np.mean(heel_tvt[m] + heel_z[m]))
    flat_full = C_heel - z
    res_heel = heel_tvt[m] - (C_heel - heel_z[m])
    Xh = np.column_stack([x[k_fit][m], y[k_fit][m]])

    x_mean = Xh.mean(axis=0)
    Xc = Xh - x_mean
    y_mean = float(np.mean(res_heel))
    yc = res_heel - y_mean
    xtx = Xc.T @ Xc + alpha * np.eye(2)
    try:
        coef = np.linalg.solve(xtx, Xc.T @ yc)
    except np.linalg.LinAlgError:
        coef = np.zeros(2)
    intercept = y_mean - float(coef @ x_mean)
    dip_trend = coef[0] * x + coef[1] * y + intercept
    # shrink dip trend toward 0 for far toe: blend by relative MD from PS
    ps_md = float(md[k_ord[-1]])
    rel = np.maximum(md - ps_md, 0.0)
    # after ~2000 ft MD from PS, trust flat more than XY plane
    w_dip = np.exp(-rel / 2500.0)
    stack = flat_full + w_dip * dip_trend

    is_known = np.zeros(n, dtype=bool)
    is_known[k] = True
    heel_full = np.full(n, np.nan)
    heel_full[k] = _heel_tvt_values(df, k, tvt_col=tvt_col, tvt_input_col=tvt_input_col)
    lock = is_known & np.isfinite(heel_full)
    stack = np.where(lock, heel_full, stack)
    flat_full = np.where(lock, heel_full, flat_full)

    return {
        "prior_flat_tvt": flat_full,
        "prior_dip_trend": dip_trend,
        "prior_dip_stack": stack,
        "C_heel": C_heel,
        "dip_coef_x": float(coef[0]),
        "dip_coef_y": float(coef[1]),
        "dip_intercept": float(intercept),
    }


def blend_stack_priors(
    last_known: float,
    prior_dip_stack: np.ndarray,
    prior_pf_bag: np.ndarray,
    pf_confidence: np.ndarray,
    *,
    conf_floor: float = 0.25,
    conf_ceil: float = 0.70,
    max_dev: float = 18.0,
) -> np.ndarray:
    """
    Conservative stack: last_known floor + small dip pull + PF only where confident.

    Hard-clamps deviation from last_known to ±max_dev so a runaway PF/dip cannot
    destroy the strong public floor (diag: unclamped blend lost 5–7 ft on some wells).
    """
    last_known = float(last_known)
    dip = np.asarray(prior_dip_stack, float)
    pf = np.asarray(prior_pf_bag, float)
    conf = np.asarray(pf_confidence, float)
    # structural: mostly last_known; dip only a soft nudge (already MD-decayed)
    dip_delta = np.clip(dip - last_known, -max_dev, max_dev)
    struct = last_known + 0.15 * dip_delta
    c = np.clip((conf - conf_floor) / max(conf_ceil - conf_floor, 1e-6), 0.0, 1.0)
    pf_delta = np.clip(pf - last_known, -max_dev, max_dev)
    out = (1.0 - c) * struct + c * (last_known + pf_delta)
    return last_known + np.clip(out - last_known, -max_dev, max_dev)
