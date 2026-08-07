"""Build station-level feature matrix for tree models."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rwt.geometry import station_local_geometry
from rwt.gr_match import multi_scale_ncc
from rwt.kernel_api import resolve_kernel
from rwt.middle_out import MiddleOutSplit, dual_transport_prior
from rwt.spatial import SpatialIndex, attach_spatial_features
from rwt.watermark import embed_column


def build_station_frame(
    lateral: pd.DataFrame,
    typewell: pd.DataFrame | None,
    split: MiddleOutSplit,
    *,
    scales: tuple[int, ...] = (16, 32, 64),
    station_idx: np.ndarray | None = None,
    spatial_index: SpatialIndex | None = None,
) -> pd.DataFrame:
    """
    One row per station: SE(3) local geom, dual-transport prior_mid, GR NCC,
    formation tops, optional spatial neighbor transfer.
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
    dual = dual_transport_prior(df, split)

    tw_gr = typewell["GR"].to_numpy(dtype=float) if typewell is not None and "GR" in typewell else None
    tw_tvt = (
        typewell["TVT"].to_numpy(dtype=float)
        if typewell is not None and "TVT" in typewell
        else None
    )
    if "GR" in df.columns:
        gr = df["GR"].to_numpy(dtype=float)
    else:
        gr = np.full(n, np.nan, dtype=float)
    if tw_gr is not None:
        tw_gr = np.asarray(tw_gr, dtype=float)

    rows = []
    indices = range(n) if station_idx is None else [int(i) for i in station_idx]
    is_predict_mask = np.zeros(n, dtype=bool)
    is_known_mask = np.zeros(n, dtype=bool)
    if len(split.predict_idx):
        is_predict_mask[np.asarray(split.predict_idx, dtype=int)] = True
    if len(split.known_idx):
        is_known_mask[np.asarray(split.known_idx, dtype=int)] = True

    for i in indices:
        gi = float(gr[i])
        row: dict[str, Any] = {
            "well_id": split.well_id,
            "station": i,
            "MD": float(df.loc[i, "MD"]),
            "X": float(df.loc[i, "X"]),
            "Y": float(df.loc[i, "Y"]),
            "Z": float(df.loc[i, "Z"]),
            "GR": gi if np.isfinite(gi) else np.nan,
            "is_predict": bool(is_predict_mask[i]),
            "is_known": bool(is_known_mask[i]),
            **{f"well_{k}": v for k, v in well_geom.items() if np.isscalar(v)},
            "local_curv": float(local["local_curv"][i]),
            "local_curv_roll": float(local["local_curv_roll"][i]),
            "local_torsion": float(local["local_torsion"][i]),
            "local_torsion_roll": float(local["local_torsion_roll"][i]),
            "e_ref": float(local["e_ref"][i]),
            "e_ref_roll": float(local["e_ref_roll"][i]),
            "d_z": float(local["d_z"][i]),
            "d_horiz": float(local["d_horiz"][i]),
            "prior_last_tvt": float(dual["prior_last_tvt"]),
            "prior_mean_tvt": float(dual["prior_mean_tvt"]),
            "prior_tvt_plus_z": float(dual["prior_tvt_plus_z"]),
            "prior_slope_tvt_md": float(dual["prior_slope_tvt_md"]),
            "prior_flat_tvt": float(dual["prior_flat"][i]),
            "prior_forward_tvt": float(dual["prior_forward"][i]),
            "prior_backward_tvt": float(dual["prior_backward"][i]),
            "prior_mid": float(dual["prior_mid"][i]),
            # Alias of prior_forward for baseline_linear_md beat-table (not a second feature)
            "prior_extrap_tvt": float(dual["prior_forward"][i]),
        }
        for form_col in ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"):
            if form_col in df.columns and pd.notna(df.loc[i, form_col]):
                row[f"form_{form_col}"] = float(df.loc[i, form_col])
                row[f"z_minus_{form_col}"] = row["Z"] - float(df.loc[i, form_col])
        if tw_gr is not None:
            m = multi_scale_ncc(gr, tw_gr, tw_tvt, scales=scales, center=i)
            row.update(m)
        if "TVT" in df.columns and pd.notna(df.loc[i, "TVT"]):
            row["TVT"] = float(df.loc[i, "TVT"])
        elif "TVT_true" in df.columns:
            row["TVT"] = float(df.loc[i, "TVT_true"])
        rows.append(row)
    stations = pd.DataFrame(rows)
    stations = attach_spatial_features(
        stations, df, spatial_index, split.well_id
    )
    stations = kernel.augment_station_frame(stations, df, typewell)
    stations = embed_column(stations)
    return stations


FEATURE_COLS_DEFAULT = [
    "MD",
    "Z",
    "GR",
    "local_curv",
    "local_curv_roll",
    "local_torsion",
    "local_torsion_roll",
    "e_ref",
    "e_ref_roll",
    "d_z",
    "d_horiz",
    "well_mean_curv",
    "well_max_curv",
    "well_cum_twist",
    "well_abs_cum_twist",
    "well_twist_std",
    "well_mean_abs_torsion",
    "well_mean_e_ref",
    "well_max_e_ref",
    "well_tortuosity_q3d_proxy",
    "well_signed_azimuth",
    "well_azimuth_x",
    "well_azimuth_y",
    "well_delta_z",
    "well_md_span",
    "prior_last_tvt",
    "prior_mean_tvt",
    "prior_tvt_plus_z",
    "prior_slope_tvt_md",
    "prior_flat_tvt",
    "prior_forward_tvt",
    "prior_backward_tvt",
    "prior_mid",
    # prior_extrap_tvt omitted from features (duplicate of prior_forward_tvt; kept on frame for baselines)
    "ncc_16",
    "ncc_32",
    "ncc_64",
    "ncc_16_lag",
    "ncc_32_lag",
    "ncc_64_lag",
    "tvt_hint_median",
    "tvt_hint_std",
    "neighbor_tvt",
    "neighbor_xy_dist",
    "neighbor_transfer_ok",
    "neighbor_count",
    "neighbor_min_dist",
    "form_ANCC",
    "form_ASTNU",
    "form_ASTNL",
    "form_EGFDU",
    "form_EGFDL",
    "form_BUDA",
    "z_minus_ANCC",
    "z_minus_ASTNU",
    "z_minus_ASTNL",
    "z_minus_EGFDU",
    "z_minus_EGFDL",
    "z_minus_BUDA",
]


def feature_matrix(
    stations: pd.DataFrame,
    cols: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    cols = cols or [c for c in FEATURE_COLS_DEFAULT if c in stations.columns]
    X = stations[cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    return X, cols
