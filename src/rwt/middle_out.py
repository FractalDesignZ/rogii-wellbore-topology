"""
Middle-out split for wellbore prediction zones.

Lineage: topology-sees-sequence middle_out_v2 — "slice 0 = prior / known".
Here: known = stations with TVT available (or pre-PS); residual = mask toe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MiddleOutSplit:
    well_id: str
    known_idx: np.ndarray
    predict_idx: np.ndarray
    ps_md: float | None
    n_known: int
    n_predict: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "well_id": self.well_id,
            "n_known": self.n_known,
            "n_predict": self.n_predict,
            "ps_md": self.ps_md,
        }


def split_by_mask(
    df: pd.DataFrame,
    *,
    well_id: str,
    mask_col: str = "is_predict",
    md_col: str = "MD",
) -> MiddleOutSplit:
    """Boolean mask: True → prediction zone (toe)."""
    m = df[mask_col].astype(bool).to_numpy()
    idx = np.arange(len(df))
    pred = idx[m]
    known = idx[~m]
    ps_md = None
    if len(pred) and md_col in df.columns:
        ps_md = float(df.iloc[pred[0]][md_col])
    return MiddleOutSplit(
        well_id=str(well_id),
        known_idx=known,
        predict_idx=pred,
        ps_md=ps_md,
        n_known=int(len(known)),
        n_predict=int(len(pred)),
    )


def split_by_prediction_start(
    df: pd.DataFrame,
    *,
    well_id: str,
    ps_md: float,
    md_col: str = "MD",
) -> MiddleOutSplit:
    """Stations with MD >= ps_md are prediction zone."""
    md = df[md_col].to_numpy(dtype=float)
    pred = np.where(md >= ps_md)[0]
    known = np.where(md < ps_md)[0]
    return MiddleOutSplit(
        well_id=str(well_id),
        known_idx=known,
        predict_idx=pred,
        ps_md=float(ps_md),
        n_known=int(len(known)),
        n_predict=int(len(pred)),
    )


def split_by_tvt_nan(
    df: pd.DataFrame,
    *,
    well_id: str,
    tvt_col: str = "TVT",
    md_col: str = "MD",
) -> MiddleOutSplit:
    """NaN TVT → predict zone (common Kaggle pattern for train masks)."""
    tvt = df[tvt_col]
    m = tvt.isna().to_numpy()
    if not m.any() and "TVT_input" in df.columns and df["TVT_input"].isna().any():
        # Train labels often fill full TVT; heel/toe mask is TVT_input
        m = df["TVT_input"].isna().to_numpy()
    idx = np.arange(len(df))
    pred = idx[m]
    known = idx[~m]
    ps_md = float(df.iloc[pred[0]][md_col]) if len(pred) and md_col in df.columns else None
    return MiddleOutSplit(
        well_id=str(well_id),
        known_idx=known,
        predict_idx=pred,
        ps_md=ps_md,
        n_known=int(len(known)),
        n_predict=int(len(pred)),
    )


def resolve_middle_out(
    df: pd.DataFrame,
    *,
    well_id: str,
) -> MiddleOutSplit:
    """
    Prefer explicit mask, then TVT_input heel mask (ROGII), then TVT NaNs.
    """
    if "is_predict" in df.columns:
        return split_by_mask(df, well_id=well_id)
    if "TVT_input" in df.columns and df["TVT_input"].isna().any():
        m = df["TVT_input"].isna().to_numpy()
        idx = np.arange(len(df))
        pred = idx[m]
        known = idx[~m]
        ps_md = float(df.iloc[pred[0]]["MD"]) if len(pred) and "MD" in df.columns else None
        return MiddleOutSplit(
            well_id=str(well_id),
            known_idx=known,
            predict_idx=pred,
            ps_md=ps_md,
            n_known=int(len(known)),
            n_predict=int(len(pred)),
        )
    if "TVT" in df.columns:
        return split_by_tvt_nan(df, well_id=well_id)
    # last resort: all known
    idx = np.arange(len(df))
    return MiddleOutSplit(
        well_id=str(well_id),
        known_idx=idx,
        predict_idx=np.array([], dtype=int),
        ps_md=None,
        n_known=int(len(idx)),
        n_predict=0,
    )


def known_prior_tvt(
    df: pd.DataFrame,
    split: MiddleOutSplit,
    *,
    tvt_col: str = "TVT",
    tvt_input_col: str = "TVT_input",
    z_col: str = "Z",
) -> dict[str, float]:
    """
    Slice-0 priors (E_ref analogue): statistics on known section only.

    Returns constants usable as residual baselines on the toe.
    """
    k = split.known_idx
    if len(k) == 0:
        return {
            "prior_last_tvt": 0.0,
            "prior_mean_tvt": 0.0,
            "prior_tvt_plus_z": 0.0,
            "prior_slope_tvt_md": 0.0,
        }
    # prefer true TVT on known; else TVT_input
    if tvt_col in df.columns and df.iloc[k][tvt_col].notna().any():
        tvt = df.iloc[k][tvt_col].astype(float).to_numpy()
    elif tvt_input_col in df.columns:
        tvt = df.iloc[k][tvt_input_col].astype(float).to_numpy()
    else:
        tvt = np.zeros(len(k))
    z = df.iloc[k][z_col].astype(float).to_numpy() if z_col in df.columns else np.zeros(len(k))
    md = (
        df.iloc[k]["MD"].astype(float).to_numpy()
        if "MD" in df.columns
        else np.arange(len(k), dtype=float)
    )
    # flat formation constant C in TVT + Z ≈ C
    c_flat = float(np.nanmedian(tvt + z)) if len(z) else float(np.nanmedian(tvt))
    # linear TVT vs MD on known
    slope = 0.0
    if len(md) >= 2 and np.nanstd(md) > 1e-9:
        coeff = np.polyfit(md, np.nan_to_num(tvt, nan=np.nanmedian(tvt)), 1)
        slope = float(coeff[0])
    return {
        "prior_last_tvt": float(tvt[-1]),
        "prior_mean_tvt": float(np.nanmean(tvt)),
        "prior_tvt_plus_z": c_flat,
        "prior_slope_tvt_md": slope,
    }
