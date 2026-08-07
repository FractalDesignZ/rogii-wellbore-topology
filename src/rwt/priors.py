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

# Residual base: pf_blend beat last_known on 40w OOF (~11.0 vs ~11.8);
# residual on this base GATE3-accepted (~10.3). Stack kept as fallback feature.
DEFAULT_PRIOR_COL = "prior_pf_blend"
PRIOR_FALLBACKS = (
    "prior_pf_blend",
    "prior_pf_bag",
    "prior_stack",
    "prior_dip_stack",
    "prior_last_tvt",
    "prior_mid",
    "prior_flat_tvt",
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


def compute_heel_fitted_dip_prior(
    df: pd.DataFrame,
    split: MiddleOutSplit,
    *,
    alpha: float = 10.0,
    tvt_col: str = "TVT",
    tvt_input_col: str = "TVT_input",
    z_col: str = "Z",
) -> dict[str, Any]:
    """
    Layers 1+2 only: fit flat C and XY dip plane on **known heel**, project full well.

    Never uses toe labels. Ridge is regularized to curb runaway toe extrapolation.
    """
    n = len(df)
    z = df[z_col].to_numpy(dtype=float) if z_col in df.columns else np.zeros(n)
    x = df["X"].to_numpy(dtype=float) if "X" in df.columns else np.arange(n, dtype=float)
    y = df["Y"].to_numpy(dtype=float) if "Y" in df.columns else np.zeros(n)

    k = np.asarray(split.known_idx, dtype=int)
    def _const_fallback(last: float) -> dict[str, Any]:
        # never invent TVT=0; prefer last heel TVT or nanmedian Z-independent constant
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

    heel_tvt = _heel_tvt_values(df, k, tvt_col=tvt_col, tvt_input_col=tvt_input_col)
    heel_z = z[k]
    m = np.isfinite(heel_tvt) & np.isfinite(heel_z)
    if m.sum() < 3:
        last = float(np.nanmedian(heel_tvt)) if np.isfinite(heel_tvt).any() else float("nan")
        return _const_fallback(last)

    C_heel = float(np.mean(heel_tvt[m] + heel_z[m]))
    flat_full = C_heel - z
    res_heel = heel_tvt[m] - (C_heel - heel_z[m])
    Xh = np.column_stack([x[k][m], y[k][m]])

    # Ridge closed form: (X'X + αI)^{-1} X'y  with intercept via centered features
    x_mean = Xh.mean(axis=0)
    Xc = Xh - x_mean
    y_mean = float(np.mean(res_heel))
    yc = res_heel - y_mean
    xtx = Xc.T @ Xc + alpha * np.eye(2)
    try:
        coef = np.linalg.solve(xtx, Xc.T @ yc)
    except np.linalg.LinAlgError:
        coef = np.zeros(2)
    # intercept so plane passes mean residual at mean XY
    intercept = y_mean - float(coef @ x_mean)
    dip_trend = coef[0] * x + coef[1] * y + intercept
    stack = flat_full + dip_trend

    # lock heel to observed TVT where known
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
    conf_floor: float = 0.2,
    conf_ceil: float = 0.65,
) -> np.ndarray:
    """
    Conservative stack: start from last_known, pull toward dip_stack, then PF bag
    only where PF confidence is high.
    """
    last_known = float(last_known)
    dip = np.asarray(prior_dip_stack, float)
    pf = np.asarray(prior_pf_bag, float)
    conf = np.asarray(pf_confidence, float)
    # structural blend: 70% last_known + 30% dip (dip alone can run away on toe)
    struct = 0.7 * last_known + 0.3 * dip
    c = np.clip((conf - conf_floor) / max(conf_ceil - conf_floor, 1e-6), 0.0, 1.0)
    return (1.0 - c) * struct + c * pf
