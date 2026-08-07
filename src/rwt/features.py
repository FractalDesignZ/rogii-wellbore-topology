"""Build station-level feature matrix for tree models."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rwt.geometry import station_local_geometry, trajectory_in_dip_frame
from rwt.gr_match import multi_scale_ncc, ncc_tvt_track
from rwt.kernel_api import resolve_kernel
from rwt.middle_out import MiddleOutSplit, dual_transport_prior
from rwt.particle_track import particle_prior_for_well
from rwt.priors import (
    FORM_SURFACE_LEAK_COLS,
    _heel_tvt_values,
    blend_prior_best,
    blend_stack_priors,
    compute_heel_fitted_dip_prior,
    compute_local_dip_prior,
)
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
    One row per station: SE(3) local geom, local-dip / PF / dual priors,
    relative kinematics, multi-scale GR rolls + NCC, spatial neighbors.

    Never emits train-only form surfaces (ANCC..BUDA) — structural leak trap.
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
    # Phase 1: orthogonal dip-frame parse (formation-aligned coordinates)
    dipf = trajectory_in_dip_frame(
        df["X"].to_numpy(), df["Y"].to_numpy(), df["Z"].to_numpy()
    )
    dual = dual_transport_prior(df, split)
    # Blueprint M2: PS anchor + local ΔTVT/ΔZ over last 300 ft heel
    local_dip_p = compute_local_dip_prior(df, split)
    # Layer 1+2: heel-only flat + XY dip plane (competition-safe)
    dip_prior = compute_heel_fitted_dip_prior(df, split)
    # Layer multi-PF bag (public lite of private-6th multi-PF idea)
    pf = particle_prior_for_well(df, typewell, split, bag=True)
    last_f = float(pf.get("prior_last_tvt", dual["prior_last_tvt"]))
    # Deterministic NCC stalk (complements stochastic PF)
    ncc_track = {
        "prior_ncc_tvt": np.full(n, last_f if np.isfinite(last_f) else np.nan),
        "ncc_track_conf": np.zeros(n),
    }
    if typewell is not None and "GR" in typewell.columns and "TVT" in typewell.columns:
        # Same heel lock contract as PF: TVT_input, else TVT on known stations only
        known_full = np.full(n, np.nan, dtype=float)
        if "TVT_input" in df.columns:
            known_full = np.asarray(df["TVT_input"].to_numpy(dtype=float), dtype=float).copy()
        if len(split.known_idx) and "TVT" in df.columns:
            kidx = np.asarray(split.known_idx, dtype=int)
            tcol = np.asarray(df["TVT"].to_numpy(dtype=float), dtype=float)
            miss = ~np.isfinite(known_full[kidx]) & np.isfinite(tcol[kidx])
            known_full[kidx[miss]] = tcol[kidx[miss]]
        gr_full = (
            df["GR"].to_numpy(dtype=float)
            if "GR" in df.columns
            else np.full(n, np.nan, dtype=float)
        )
        ncc_track = ncc_tvt_track(
            gr_full,
            typewell["GR"].to_numpy(dtype=float),
            typewell["TVT"].to_numpy(dtype=float),
            known_tvt=known_full,
            seed_tvt=last_f,
            stride=8,
        )
    stack = blend_stack_priors(
        last_f,
        dip_prior["prior_dip_stack"],
        pf["prior_pf_bag"],
        pf["pf_confidence"],
    )
    # lock heel on stack via shared heel TVT helper
    if len(split.known_idx):
        k = np.asarray(split.known_idx, dtype=int)
        heel_tvt = _heel_tvt_values(df, k)
        full = np.full(n, np.nan, dtype=float)
        full[k] = heel_tvt
        lock = np.isfinite(full)
        stack = np.where(lock, full, stack)

    # M1 kinematics: real segments only (no synthetic prepend zero-step)
    md_arr = df["MD"].to_numpy(dtype=float) if "MD" in df.columns else np.arange(n, dtype=float)
    z_arr = df["Z"].to_numpy(dtype=float) if "Z" in df.columns else np.zeros(n)
    x_arr = df["X"].to_numpy(dtype=float) if "X" in df.columns else np.zeros(n)
    y_arr = df["Y"].to_numpy(dtype=float) if "Y" in df.columns else np.zeros(n)
    incl = np.zeros(n, dtype=float)
    dls = np.zeros(n, dtype=float)
    if n >= 2:
        d_md = np.diff(md_arr)
        d_z = np.diff(z_arr)
        d_x = np.diff(x_arr)
        d_y = np.diff(y_arr)
        d_md_safe = np.where(np.abs(d_md) < 1e-9, 1e-9, d_md)
        # inclination on real intervals → assign to trailing station
        incl_seg = np.degrees(np.arccos(np.clip(d_z / d_md_safe, -1.0, 1.0)))
        incl[1:] = incl_seg
        incl[0] = incl[1]
        # 3D unit tangents on real segments; DLS uses adjacent MD intervals
        dxyz = np.column_stack([d_x, d_y, d_z])
        seg_len = np.linalg.norm(dxyz, axis=1)
        seg_len = np.where(seg_len < 1e-12, 1e-12, seg_len)
        tang = dxyz / seg_len[:, None]
        if n >= 3:
            dots = np.sum(tang[1:] * tang[:-1], axis=1)
            dots = np.clip(dots, -1.0, 1.0)
            ang = np.degrees(np.arccos(dots))
            # MD interval spanning the two adjacent segments
            ds_md = 0.5 * (np.abs(d_md[1:]) + np.abs(d_md[:-1]))
            ds_md = np.where(ds_md < 1e-9, 1e-9, ds_md)
            dls_mid = ang / ds_md * 100.0
            dls[1:-1] = dls_mid
            dls[0] = dls[1]
            dls[-1] = dls[-2]
        else:
            dls[:] = 0.0

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

    # M3: multi-scale GR rolling stats (sensor-noise filter; trees handle NaN via fill)
    gr_s = pd.Series(gr, dtype=float)
    gr_rolls: dict[str, np.ndarray] = {}
    for w in (10, 20, 40):
        gr_rolls[f"GR_mean_{w}"] = (
            gr_s.rolling(window=w, min_periods=1, center=True).mean().to_numpy(dtype=float)
        )
        gr_rolls[f"GR_std_{w}"] = (
            gr_s.rolling(window=w, min_periods=1, center=True)
            .std()
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

    rows = []
    idx_list = list(range(n) if station_idx is None else [int(i) for i in station_idx])
    indices = idx_list
    is_predict_mask = np.zeros(n, dtype=bool)
    is_known_mask = np.zeros(n, dtype=bool)
    if len(split.predict_idx):
        is_predict_mask[np.asarray(split.predict_idx, dtype=int)] = True
    if len(split.known_idx):
        is_known_mask[np.asarray(split.known_idx, dtype=int)] = True

    # Sparse multi-scale NCC (every 8th station) — O(n) sample + O(log n) nearest fill
    ncc_cache: dict[int, dict[str, float]] = {}
    if tw_gr is not None and idx_list:
        first_i, last_i = idx_list[0], idx_list[-1]
        sample_idx = [i for i in idx_list if (i % 8 == 0 or i == first_i or i == last_i)]
        for i in sample_idx:
            ncc_cache[i] = multi_scale_ncc(gr, tw_gr, tw_tvt, scales=scales, center=i)
        if ncc_cache:
            keys = np.asarray(sorted(ncc_cache.keys()), dtype=int)
            for i in idx_list:
                if i not in ncc_cache:
                    j = int(np.searchsorted(keys, i))
                    if j <= 0:
                        nearest = int(keys[0])
                    elif j >= len(keys):
                        nearest = int(keys[-1])
                    else:
                        nearest = int(
                            keys[j - 1]
                            if abs(int(keys[j - 1]) - i) <= abs(int(keys[j]) - i)
                            else keys[j]
                        )
                    ncc_cache[i] = ncc_cache[nearest]

    rel_md = local_dip_p["relative_md"]
    rel_z = local_dip_p["relative_z"]
    prior_loc = local_dip_p["prior_local_dip"]
    loc_dip = float(local_dip_p["local_dip"]) if np.isfinite(local_dip_p["local_dip"]) else 0.0

    for i in indices:
        gi = float(gr[i])
        row: dict[str, Any] = {
            "well_id": split.well_id,
            "station": i,
            # absolute coords retained for spatial/submit id only — not in FEATURE_COLS
            "MD": float(df.loc[i, "MD"]),
            "X": float(df.loc[i, "X"]),
            "Y": float(df.loc[i, "Y"]),
            "Z": float(df.loc[i, "Z"]),
            "GR": gi if np.isfinite(gi) else np.nan,
            "is_predict": bool(is_predict_mask[i]),
            "is_known": bool(is_known_mask[i]),
            **{f"well_{k}": v for k, v in well_geom.items() if np.isscalar(v)},
            # relative kinematics (anti map-memorization)
            "relative_md": float(rel_md[i]),
            "relative_z": float(rel_z[i]),
            "inclination_deg": float(incl[i]),
            "curvature_dls": float(dls[i]),
            "local_dip": loc_dip,
            "local_curv": float(local["local_curv"][i]),
            "local_curv_roll": float(local["local_curv_roll"][i]),
            "local_torsion": float(local["local_torsion"][i]),
            "local_torsion_roll": float(local["local_torsion_roll"][i]),
            "e_ref": float(local["e_ref"][i]),
            "e_ref_roll": float(local["e_ref_roll"][i]),
            "d_z": float(local["d_z"][i]),
            "d_horiz": float(local["d_horiz"][i]),
            "x_dip": float(dipf["x_dip"][i]),
            "y_dip": float(dipf["y_dip"][i]),
            "z_dip": float(dipf["z_dip"][i]),
            "form_azimuth_deg": float(dipf["azimuth_deg"]),
            "form_dip_deg": float(dipf["dip_deg"]),
            "prior_last_tvt": float(dual["prior_last_tvt"]),
            "prior_mean_tvt": float(dual["prior_mean_tvt"]),
            "prior_tvt_plus_z": float(dual["prior_tvt_plus_z"]),
            "prior_slope_tvt_md": float(dual["prior_slope_tvt_md"]),
            "prior_flat_tvt": float(dip_prior["prior_flat_tvt"][i]),
            "prior_forward_tvt": float(dual["prior_forward"][i]),
            "prior_backward_tvt": float(dual["prior_backward"][i]),
            "prior_mid": float(dual["prior_mid"][i]),
            "prior_extrap_tvt": float(dual["prior_forward"][i]),
            "prior_local_dip": float(prior_loc[i]) if np.isfinite(prior_loc[i]) else float("nan"),
            "prior_dip_trend": float(dip_prior["prior_dip_trend"][i]),
            "prior_dip_stack": float(dip_prior["prior_dip_stack"][i]),
            "prior_pf_tvt": float(pf["prior_pf_tvt"][i]),
            "prior_pf_blend": float(pf["prior_pf_blend"][i]),
            "prior_pf_bag": float(pf["prior_pf_bag"][i]),
            "prior_stack": float(stack[i]),
            "prior_ncc_tvt": float(ncc_track["prior_ncc_tvt"][i])
            if np.isfinite(ncc_track["prior_ncc_tvt"][i])
            else float(last_f) if np.isfinite(last_f) else float("nan"),
            "ncc_track_conf": float(ncc_track["ncc_track_conf"][i]),
            "pf_confidence": float(pf["pf_confidence"][i]),
            "pf_tvt_std": float(pf["pf_tvt_std"][i])
            if np.isfinite(pf["pf_tvt_std"][i])
            else 0.0,
            "C_heel": float(dip_prior["C_heel"]),
            "dip_coef_x": float(dip_prior["dip_coef_x"]),
            "dip_coef_y": float(dip_prior["dip_coef_y"]),
        }
        for rk, arr in gr_rolls.items():
            v = float(arr[i])
            row[rk] = v if np.isfinite(v) else 0.0
        # INTENTIONAL: never add FORM_SURFACE_LEAK_COLS (ANCC..BUDA)
        if int(i) in ncc_cache:
            row.update(ncc_cache[int(i)])
        if "TVT" in df.columns and pd.notna(df.loc[i, "TVT"]):
            row["TVT"] = float(df.loc[i, "TVT"])
        elif "TVT_true" in df.columns:
            row["TVT"] = float(df.loc[i, "TVT_true"])
        rows.append(row)
    stations = pd.DataFrame(rows)
    stations = attach_spatial_features(
        stations, df, spatial_index, split.well_id
    )
    # Crown pure prior after neighbor attach (uses transfer_ok + heel offset)
    last_col = stations["prior_last_tvt"].to_numpy(dtype=float)
    last_f = float(last_col[0]) if len(last_col) and np.isfinite(last_col[0]) else float("nan")
    if not np.isfinite(last_f):
        last_f = float(np.nanmedian(last_col)) if np.isfinite(last_col).any() else 0.0
    nb = (
        stations["neighbor_tvt"].to_numpy(dtype=float)
        if "neighbor_tvt" in stations.columns
        else None
    )
    nb_ok = (
        stations["neighbor_transfer_ok"].to_numpy(dtype=float)
        if "neighbor_transfer_ok" in stations.columns
        else None
    )
    best = blend_prior_best(
        last_f,
        stations["prior_stack"].to_numpy(dtype=float),
        stations["prior_pf_blend"].to_numpy(dtype=float),
        stations["prior_local_dip"].to_numpy(dtype=float),
        stations["pf_confidence"].to_numpy(dtype=float),
        neighbor_tvt=nb,
        neighbor_ok=nb_ok,
        prior_ncc=stations["prior_ncc_tvt"].to_numpy(dtype=float)
        if "prior_ncc_tvt" in stations.columns
        else None,
        ncc_conf=stations["ncc_track_conf"].to_numpy(dtype=float)
        if "ncc_track_conf" in stations.columns
        else None,
    )
    # Heel lock: TVT_input-first (competition-safe), same helper as stack / dip.
    # Do not require form-label TVT column — test wells may only have TVT_input.
    if len(split.known_idx):
        k = np.asarray(split.known_idx, dtype=int)
        heel_vals = _heel_tvt_values(df, k)
        best = np.asarray(best, dtype=float).copy()
        if "station" in stations.columns:
            st = stations["station"].to_numpy(dtype=int)
            k_to_h = {int(ki): float(hv) for ki, hv in zip(k, heel_vals) if np.isfinite(hv)}
            for j, s in enumerate(st):
                if int(s) in k_to_h:
                    best[j] = k_to_h[int(s)]
        else:
            # stations rows align with full lateral 0..n-1
            for j, (ki, hv) in enumerate(zip(k, heel_vals)):
                if 0 <= int(ki) < len(best) and np.isfinite(hv):
                    best[int(ki)] = float(hv)
    stations["prior_best"] = best
    stations = kernel.augment_station_frame(stations, df, typewell)
    stations = embed_column(stations)
    return stations


# Tight residual X: relative kinematics + signal + strong priors only.
# Drop absolute map coords, well-azimuth lease encodings, dual-transport mid
# (diag: mid/flat/extrap RMSE 100–1000 on toe — poison for residual trees).
FEATURE_COLS_DEFAULT = [
    "relative_md",
    "relative_z",
    "GR",
    "GR_mean_10",
    "GR_std_10",
    "GR_mean_20",
    "GR_std_20",
    "GR_mean_40",
    "GR_std_40",
    "inclination_deg",
    "curvature_dls",
    "local_dip",
    "local_curv",
    "local_curv_roll",
    "local_torsion",
    "e_ref",
    "d_z",
    "d_horiz",
    "x_dip",
    "y_dip",
    "z_dip",
    "form_azimuth_deg",
    "form_dip_deg",
    "well_mean_curv",
    "well_mean_e_ref",
    "well_tortuosity_q3d_proxy",
    "prior_last_tvt",
    "prior_local_dip",
    "prior_dip_stack",
    "prior_pf_tvt",
    "prior_pf_blend",
    "prior_pf_bag",
    "prior_stack",
    "prior_best",
    "prior_ncc_tvt",
    "ncc_track_conf",
    "pf_confidence",
    "pf_tvt_std",
    "ncc_16",
    "ncc_32",
    "ncc_64",
    "ncc_16_lag",
    "ncc_32_lag",
    "ncc_64_lag",
    "tvt_hint_median",
    "ncc_peak",
    "neighbor_tvt",
    "neighbor_xy_dist",
    "neighbor_transfer_ok",
    "neighbor_count",
    "neighbor_min_dist",
    "neighbor_offset",
]


def feature_matrix(
    stations: pd.DataFrame,
    cols: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    cols = cols or [c for c in FEATURE_COLS_DEFAULT if c in stations.columns]
    # Trap 1: refuse train-only form surfaces if a caller tries to inject them
    leak = [
        c
        for c in cols
        if c in FORM_SURFACE_LEAK_COLS
        or c.startswith("form_ANCC")
        or c.startswith("form_AST")
        or c.startswith("form_EGF")
        or c.startswith("form_BUDA")
        or c.startswith("z_minus_")
    ]
    # form_azimuth/form_dip are SO(3) dip-frame angles — not geological tops
    leak = [c for c in leak if c not in ("form_azimuth_deg", "form_dip_deg")]
    if leak:
        raise ValueError(
            f"structural leak: refuse form-surface features {leak} "
            f"(train-only ANCC..BUDA; drop before residual train)"
        )
    X = stations[cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    return X, cols
