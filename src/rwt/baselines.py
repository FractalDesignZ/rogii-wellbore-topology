"""Structural baselines for TVT (must beat before claiming fancy wins)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rwt.middle_out import MiddleOutSplit, known_prior_tvt


def baseline_last_known(
    df: pd.DataFrame,
    split: MiddleOutSplit,
    *,
    tvt_col: str = "TVT",
    tvt_input_col: str = "TVT_input",
) -> np.ndarray:
    """Constant = last known TVT on known section."""
    prior = known_prior_tvt(df, split, tvt_col=tvt_col, tvt_input_col=tvt_input_col)
    pred = np.full(len(split.predict_idx), prior["prior_last_tvt"], dtype=float)
    return pred


def baseline_flat_formation(
    df: pd.DataFrame,
    split: MiddleOutSplit,
    *,
    z_col: str = "Z",
    tvt_col: str = "TVT",
    tvt_input_col: str = "TVT_input",
) -> np.ndarray:
    """TVT ≈ C − Z with C fit on known section (flat beds)."""
    prior = known_prior_tvt(df, split, tvt_col=tvt_col, tvt_input_col=tvt_input_col, z_col=z_col)
    c = prior["prior_tvt_plus_z"]
    z = df.iloc[split.predict_idx][z_col].astype(float).to_numpy()
    return c - z


def baseline_linear_md(
    df: pd.DataFrame,
    split: MiddleOutSplit,
    *,
    md_col: str = "MD",
    tvt_col: str = "TVT",
    tvt_input_col: str = "TVT_input",
) -> np.ndarray:
    """Linear extrapolation of TVT vs MD from known section."""
    prior = known_prior_tvt(df, split, tvt_col=tvt_col, tvt_input_col=tvt_input_col)
    k = split.known_idx
    if tvt_col in df.columns and df.iloc[k][tvt_col].notna().any():
        tvt = df.iloc[k][tvt_col].astype(float).to_numpy()
    else:
        tvt = df.iloc[k][tvt_input_col].astype(float).to_numpy()
    md_k = df.iloc[k][md_col].astype(float).to_numpy()
    if len(md_k) < 2:
        return np.full(len(split.predict_idx), prior["prior_last_tvt"])
    coeff = np.polyfit(md_k, np.nan_to_num(tvt, nan=float(np.nanmedian(tvt))), 1)
    md_p = df.iloc[split.predict_idx][md_col].astype(float).to_numpy()
    return coeff[0] * md_p + coeff[1]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))


def evaluate_baselines_on_known_holdout(
    df: pd.DataFrame,
    split: MiddleOutSplit,
    *,
    holdout_frac: float = 0.25,
    tvt_col: str = "TVT",
) -> dict[str, Any]:
    """
    Pseudo-eval: hold out last holdout_frac of *known* stations as fake toe.
    Only for synthetic / debug when true mask TVT unavailable.
    """
    k = split.known_idx
    if len(k) < 8 or tvt_col not in df.columns:
        return {"ok": False, "reason": "insufficient_known"}
    n_hold = max(2, int(len(k) * holdout_frac))
    known2 = k[:-n_hold]
    fake_pred = k[-n_hold:]
    fake = MiddleOutSplit(
        well_id=split.well_id + "_holdout",
        known_idx=known2,
        predict_idx=fake_pred,
        ps_md=None,
        n_known=len(known2),
        n_predict=len(fake_pred),
    )
    y = df.iloc[fake_pred][tvt_col].astype(float).to_numpy()
    preds = {
        "last_known": baseline_last_known(df, fake, tvt_col=tvt_col),
        "flat_formation": baseline_flat_formation(df, fake, tvt_col=tvt_col),
        "linear_md": baseline_linear_md(df, fake, tvt_col=tvt_col),
    }
    return {
        "ok": True,
        "n_hold": n_hold,
        "rmse": {name: rmse(y, p) for name, p in preds.items()},
    }
