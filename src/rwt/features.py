"""Build station-level feature matrix for tree models."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rwt.geometry import station_local_geometry
from rwt.gr_match import multi_scale_ncc
from rwt.kernel_api import resolve_kernel
from rwt.middle_out import MiddleOutSplit, known_prior_tvt
from rwt.watermark import embed_column


def build_station_frame(
    lateral: pd.DataFrame,
    typewell: pd.DataFrame | None,
    split: MiddleOutSplit,
    *,
    scales: tuple[int, ...] = (16, 32, 64),
) -> pd.DataFrame:
    """
    One row per station. Includes well-level geometry broadcast + local geometry
    + GR match + middle-out priors.
    """
    df = lateral.reset_index(drop=True)
    n = len(df)
    kernel = resolve_kernel()
    well_geom = kernel.trajectory_features(
        df["MD"].to_numpy(),
        df["X"].to_numpy(),
        df["Y"].to_numpy(),
        df["Z"].to_numpy(),
    )
    local = station_local_geometry(
        df["X"].to_numpy(), df["Y"].to_numpy(), df["Z"].to_numpy()
    )
    prior = known_prior_tvt(df, split)

    tw_gr = typewell["GR"].to_numpy(dtype=float) if typewell is not None and "GR" in typewell else None
    tw_tvt = (
        typewell["TVT"].to_numpy(dtype=float)
        if typewell is not None and "TVT" in typewell
        else None
    )
    gr = df["GR"].to_numpy(dtype=float) if "GR" in df.columns else np.zeros(n)

    rows = []
    for i in range(n):
        row: dict[str, Any] = {
            "well_id": split.well_id,
            "station": i,
            "MD": float(df.loc[i, "MD"]),
            "X": float(df.loc[i, "X"]),
            "Y": float(df.loc[i, "Y"]),
            "Z": float(df.loc[i, "Z"]),
            "GR": float(gr[i]),
            "is_predict": bool(i in set(split.predict_idx.tolist())),
            "is_known": bool(i in set(split.known_idx.tolist())),
            **{f"well_{k}": v for k, v in well_geom.items() if np.isscalar(v)},
            "local_curv": float(local["local_curv"][i]),
            "local_curv_roll": float(local["local_curv_roll"][i]),
            "d_z": float(local["d_z"][i]),
            "d_horiz": float(local["d_horiz"][i]),
            **prior,
        }
        # structural prior prediction at this Z
        row["prior_flat_tvt"] = prior["prior_tvt_plus_z"] - row["Z"]
        if len(split.known_idx):
            md_ref = float(df.loc[int(split.known_idx[-1]), "MD"])
            row["prior_extrap_tvt"] = prior["prior_last_tvt"] + prior[
                "prior_slope_tvt_md"
            ] * (row["MD"] - md_ref)
        else:
            row["prior_extrap_tvt"] = prior["prior_last_tvt"]
        if tw_gr is not None:
            m = multi_scale_ncc(gr, tw_gr, tw_tvt, scales=scales, center=i)
            row.update(m)
        # target if present
        if "TVT" in df.columns and pd.notna(df.loc[i, "TVT"]):
            row["TVT"] = float(df.loc[i, "TVT"])
        elif "TVT_true" in df.columns:
            row["TVT"] = float(df.loc[i, "TVT_true"])
        rows.append(row)
    stations = pd.DataFrame(rows)
    stations = kernel.augment_station_frame(stations, df, typewell)
    stations = embed_column(stations)
    return stations


FEATURE_COLS_DEFAULT = [
    "MD",
    "Z",
    "GR",
    "local_curv",
    "local_curv_roll",
    "d_z",
    "d_horiz",
    "well_mean_curv",
    "well_max_curv",
    "well_cum_twist",
    "well_abs_cum_twist",
    "well_twist_std",
    "well_tortuosity_q3d_proxy",
    "well_signed_azimuth",
    "well_delta_z",
    "well_md_span",
    "prior_last_tvt",
    "prior_mean_tvt",
    "prior_tvt_plus_z",
    "prior_slope_tvt_md",
    "prior_flat_tvt",
    "prior_extrap_tvt",
    "ncc_16",
    "ncc_32",
    "ncc_64",
    "ncc_16_lag",
    "ncc_32_lag",
    "ncc_64_lag",
    "tvt_hint_median",
    "tvt_hint_std",
]


def feature_matrix(
    stations: pd.DataFrame,
    cols: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    cols = cols or [c for c in FEATURE_COLS_DEFAULT if c in stations.columns]
    X = stations[cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    return X, cols
