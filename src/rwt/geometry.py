"""
Open-curve geometry features for wellbore trajectories.

Lineage: adapted from topology-sees-sequence `holonomy_extract.py`
(_rot_between, Bishop-style parallel transport). Wells are OPEN curves —
we never force closed holonomy = 0.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.linalg import norm


def rotation_matrix_z(yaw_rad: float) -> np.ndarray:
    """Right-handed SO(3) rotation about +Z."""
    c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def rotation_matrix_y(pitch_rad: float) -> np.ndarray:
    """Right-handed SO(3) rotation about +Y."""
    c, s = float(np.cos(pitch_rad)), float(np.sin(pitch_rad))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def rotation_matrix_x(roll_rad: float) -> np.ndarray:
    """Right-handed SO(3) rotation about +X."""
    c, s = float(np.cos(roll_rad)), float(np.sin(roll_rad))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def apply_orthogonal_dip_rotation(
    xyz: np.ndarray,
    azimuth_deg: float,
    dip_deg: float,
) -> np.ndarray:
    """
    Orthogonally rotate wellbore coordinates into a local formation dip frame.

    R = R_y(dip) @ R_z(-azimuth)
    applied as  xyz_dip = xyz @ R.T   (row-vector convention)

    azimuth_deg: formation strike / lateral azimuth (deg from +X toward +Y)
    dip_deg: formation dip (deg, positive dips toward azimuth direction)
    """
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be (N, 3)")
    az = np.radians(float(azimuth_deg))
    dip = np.radians(float(dip_deg))
    R_z = rotation_matrix_z(-az)
    R_y = rotation_matrix_y(dip)
    R = R_y @ R_z
    # contiguous for SIMD matmul
    xyz_c = np.ascontiguousarray(xyz, dtype=float)
    return xyz_c @ R.T


def estimate_lateral_azimuth_dip_deg(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[float, float]:
    """
    Crude formation-frame estimate from trajectory: azimuth from horizontal
    mean tangent; dip from mean |dZ|/dS along lateral body.
    """
    x, y, z = map(lambda a: np.asarray(a, float), (x, y, z))
    if len(x) < 3:
        return 0.0, 0.0
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    ds = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-15
    # use last 60% of path (lateral)
    cut = max(1, int(0.4 * len(dx)))
    ax = float(np.mean(dx[cut:]))
    ay = float(np.mean(dy[cut:]))
    az_deg = float(np.degrees(np.arctan2(ay, ax + 1e-15)))
    horiz = np.sqrt(dx[cut:] ** 2 + dy[cut:] ** 2) + 1e-15
    dip_deg = float(np.degrees(np.arctan2(np.mean(np.abs(dz[cut:])), np.mean(horiz))))
    return az_deg, dip_deg


def trajectory_in_dip_frame(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    azimuth_deg: float | None = None,
    dip_deg: float | None = None,
) -> dict[str, np.ndarray | float]:
    """Return dip-frame coordinates + estimated (or given) azimuth/dip."""
    if azimuth_deg is None or dip_deg is None:
        az_e, dip_e = estimate_lateral_azimuth_dip_deg(x, y, z)
        azimuth_deg = az_e if azimuth_deg is None else azimuth_deg
        dip_deg = dip_e if dip_deg is None else dip_deg
    xyz = np.column_stack([np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)])
    # center before rotate (pure orthogonal frame, not origin-dependent)
    origin = xyz[0].copy()
    xyz0 = xyz - origin
    xyz_d = apply_orthogonal_dip_rotation(xyz0, float(azimuth_deg), float(dip_deg))
    return {
        "x_dip": xyz_d[:, 0],
        "y_dip": xyz_d[:, 1],
        "z_dip": xyz_d[:, 2],
        "azimuth_deg": float(azimuth_deg),
        "dip_deg": float(dip_deg),
    }


def rot_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Minimal SO(3) rotation taking unit vector u onto v (det = +1).

    Port of topology-sees-sequence holonomy_extract._rot_between.
    """
    u = np.asarray(u, dtype=float).reshape(3)
    v = np.asarray(v, dtype=float).reshape(3)
    u = u / (norm(u) + 1e-15)
    v = v / (norm(v) + 1e-15)
    c = float(np.clip(u @ v, -1, 1))
    ax = np.cross(u, v)
    s = float(norm(ax))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        helper = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        ax = np.cross(u, helper)
        ax = ax / (norm(ax) + 1e-15)
        return 2.0 * np.outer(ax, ax) - np.eye(3)
    ax = ax / s
    th = float(np.arctan2(s, c))
    K = np.array(
        [[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]], dtype=float
    )
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def tangents_from_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Unit tangents along open polyline; length n-1 for n stations."""
    p = np.column_stack([np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)])
    d = np.diff(p, axis=0)
    nn = norm(d, axis=1, keepdims=True)
    nn = np.maximum(nn, 1e-12)
    return d / nn


def discrete_curvature(T: np.ndarray) -> np.ndarray:
    """Turning angle (rad) between consecutive unit tangents. Length n-2 for |T|=n-1."""
    if len(T) < 2:
        return np.zeros(0)
    out = []
    for i in range(len(T) - 1):
        c = float(np.clip(T[i] @ T[i + 1], -1, 1))
        out.append(float(np.arccos(c)))
    return np.asarray(out, dtype=float)


def discrete_frenet_frame(T: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Discrete Frenet-ish frames along unit tangents T (m,3).

    Returns (T_f, N, B) each shape (m-1, 3) aligned to segments between stations.
    Principal normal from consecutive tangent change; binormal B = T × N.
    """
    m = len(T)
    if m < 2:
        z = np.zeros((0, 3))
        return z, z, z
    Tf, Nf, Bf = [], [], []
    for i in range(m - 1):
        t0, t1 = T[i], T[i + 1]
        # discrete curvature direction
        dn = t1 - t0
        dn = dn - (dn @ t0) * t0
        nn = norm(dn)
        if nn < 1e-12:
            # straight: carry previous normal or pick arbitrary
            if i > 0 and len(Nf):
                n = Nf[-1] - (Nf[-1] @ t0) * t0
                if norm(n) < 1e-12:
                    helper = np.array([0.0, 0.0, 1.0]) if abs(t0[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
                    n = np.cross(t0, helper)
            else:
                helper = np.array([0.0, 0.0, 1.0]) if abs(t0[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
                n = np.cross(t0, helper)
            n = n / (norm(n) + 1e-15)
        else:
            n = dn / nn
        b = np.cross(t0, n)
        bn = norm(b)
        if bn < 1e-12:
            b = np.array([0.0, 0.0, 1.0])
        else:
            b = b / bn
        # re-orthogonalize n
        n = np.cross(b, t0)
        n = n / (norm(n) + 1e-15)
        Tf.append(t0)
        Nf.append(n)
        Bf.append(b)
    return np.asarray(Tf), np.asarray(Nf), np.asarray(Bf)


def discrete_torsion(T: np.ndarray) -> np.ndarray:
    """
    Discrete torsion τ (rad) between consecutive binormals.

    Length m-2 for |T|=m-1. This is the open-curve holonomy-relevant signal
    (rotation of the osculating plane), not pure-Bishop roll (which is ~0).
    """
    _Tf, _N, B = discrete_frenet_frame(T)
    if len(B) < 2:
        return np.zeros(0)
    out = []
    for i in range(len(B) - 1):
        # angle of B_i → B_{i+1} about tangent T[i+1]
        t = T[min(i + 1, len(T) - 1)]
        b0, b1 = B[i], B[i + 1]
        # project into plane ⟂ t
        b0p = b0 - (b0 @ t) * t
        b1p = b1 - (b1 @ t) * t
        n0, n1 = norm(b0p), norm(b1p)
        if n0 < 1e-12 or n1 < 1e-12:
            out.append(0.0)
            continue
        b0p, b1p = b0p / n0, b1p / n1
        c = float(np.clip(b0p @ b1p, -1, 1))
        s = float(np.cross(b0p, b1p) @ t)
        out.append(float(np.arctan2(s, c)))
    return np.asarray(out, dtype=float)


def open_bishop_transport(T: np.ndarray) -> dict[str, Any]:
    """
    Open-curve frame transport along unit tangents T (shape (m,3)).

    Uses discrete Frenet binormal torsion as the twist signal (non-null on helices).
    Bishop parallel transport of a normal is still computed for frame tracking;
    we do **not** claim closed holonomy = 0.
    """
    m = len(T)
    if m < 2:
        return {
            "n_segments": m,
            "cum_twist": 0.0,
            "abs_cum_twist": 0.0,
            "twist_per_seg": np.zeros(0),
            "twist_std": 0.0,
            "mean_curvature": 0.0,
            "mean_abs_torsion": 0.0,
            "final_bishop_normal": np.array([0.0, 1.0, 0.0]),
        }
    # Bishop frame tracking (reference)
    u = np.cross(T[0], np.array([0.0, 0.0, 1.0]))
    if norm(u) < 1e-8:
        u = np.cross(T[0], np.array([0.0, 1.0, 0.0]))
    u = u / (norm(u) + 1e-15)
    for i in range(m - 1):
        R = rot_between(T[i], T[i + 1])
        u = R @ u
        u = u - (u @ T[i + 1]) * T[i + 1]
        nu = norm(u)
        if nu > 1e-12:
            u = u / nu

    # Real topological signal: discrete torsion of osculating plane
    torsion = discrete_torsion(T)
    # pad torsion to segment-ish length for stats (m-2 samples)
    twists_a = torsion if len(torsion) else np.zeros(0)
    return {
        "n_segments": m,
        "cum_twist": float(np.sum(twists_a)),
        "abs_cum_twist": float(np.sum(np.abs(twists_a))),
        "twist_per_seg": twists_a,
        "twist_std": float(np.std(twists_a)) if len(twists_a) else 0.0,
        "mean_curvature": float(np.mean(discrete_curvature(T))) if m >= 2 else 0.0,
        "mean_abs_torsion": float(np.mean(np.abs(twists_a))) if len(twists_a) else 0.0,
        "final_bishop_normal": u,
    }


def trajectory_geometry_features(
    md: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> dict[str, Any]:
    """Per-well summary geometry features (safe under GroupKFold-by-well)."""
    md = np.asarray(md, float)
    x, y, z = map(lambda a: np.asarray(a, float), (x, y, z))
    n = len(md)
    if n < 3:
        return {
            "n_stations": n,
            "md_span": float(md[-1] - md[0]) if n else 0.0,
            "delta_z": float(z[-1] - z[0]) if n else 0.0,
            "path_length": 0.0,
            "mean_curv": 0.0,
            "max_curv": 0.0,
            "cum_twist": 0.0,
            "abs_cum_twist": 0.0,
            "twist_std": 0.0,
            "mean_abs_torsion": 0.0,
            "mean_e_ref": 0.0,
            "max_e_ref": 0.0,
            "tortuosity_q3d_proxy": 0.0,
            "signed_azimuth": 0.0,
            "azimuth_x": 0.0,
            "azimuth_y": 0.0,
        }
    T = tangents_from_xyz(x, y, z)
    curv = discrete_curvature(T)
    bishop = open_bishop_transport(T)
    eref = e_ref_joint_strain(x, y, z)
    segs = norm(np.diff(np.column_stack([x, y, z]), axis=0), axis=1)
    path_len = float(np.sum(segs))
    chord = float(norm(np.array([x[-1] - x[0], y[-1] - y[0], z[-1] - z[0]])))
    tort = path_len / max(chord, 1e-6) - 1.0
    horiz = T.copy()
    horiz[:, 2] = 0.0
    hn = norm(horiz, axis=1, keepdims=True)
    horiz = np.where(hn > 1e-8, horiz / np.maximum(hn, 1e-8), 0.0)
    mean_h = np.mean(horiz, axis=0)
    signed_azi = float(np.arctan2(mean_h[1], mean_h[0] + 1e-15))
    return {
        "n_stations": n,
        "md_span": float(md[-1] - md[0]),
        "delta_z": float(z[-1] - z[0]),
        "path_length": path_len,
        "mean_curv": float(np.mean(curv)) if len(curv) else 0.0,
        "max_curv": float(np.max(curv)) if len(curv) else 0.0,
        "cum_twist": bishop["cum_twist"],
        "abs_cum_twist": bishop["abs_cum_twist"],
        "twist_std": bishop["twist_std"],
        "mean_abs_torsion": bishop.get("mean_abs_torsion", 0.0),
        "mean_e_ref": float(eref["mean_e_ref"]),
        "max_e_ref": float(eref["max_e_ref"]),
        "tortuosity_q3d_proxy": float(tort),
        "signed_azimuth": signed_azi,
        "azimuth_x": float(mean_h[0]),
        "azimuth_y": float(mean_h[1]),
    }


def so3_log_angle(R: np.ndarray) -> float:
    """Rotation angle (rad) of SO(3) matrix via trace formula."""
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.arccos(c))


def se3_screw_norm(R: np.ndarray, t: np.ndarray, *, length_scale: float = 100.0) -> float:
    """
    Public E_ref-style magnitude: ||log SE(3)|| screw proxy.

    Combines SO(3) angle with scaled translation (ft). Structure ported from
    TSS derive.se3_log spirit — open-curve survey joints, not closed loops.
    """
    th = so3_log_angle(R)
    tn = float(norm(np.asarray(t, float)))
    # dimensionless blend: angle + translation / characteristic joint scale
    return float(np.sqrt(th * th + (tn / max(length_scale, 1e-6)) ** 2))


def e_ref_joint_strain(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    length_scale: float | None = None,
) -> dict[str, np.ndarray | float]:
    """
    Per-station SE(3) joint strain E_ref between consecutive survey points.

    E_ref(s) ≈ || log_SE(3)(g_s^{-1} g_{s+1}) ||_screw
    with R = rot_between(T_s, T_{s+1}) and t = Δxyz.
    length_scale defaults to median segment length (scale-coherent).
    """
    x, y, z = map(lambda a: np.asarray(a, float), (x, y, z))
    n = len(x)
    e = np.zeros(n)
    if n < 2:
        return {
            "e_ref": e,
            "mean_e_ref": 0.0,
            "max_e_ref": 0.0,
            "p95_e_ref": 0.0,
            "length_scale": float(length_scale if length_scale is not None else 100.0),
        }
    p = np.column_stack([x, y, z])
    segs = norm(np.diff(p, axis=0), axis=1)
    if length_scale is None:
        length_scale = float(np.median(segs)) if len(segs) else 100.0
        length_scale = max(length_scale, 1e-3)
    T = tangents_from_xyz(x, y, z)
    for i in range(len(T)):
        j = min(i + 1, len(T) - 1)
        R = rot_between(T[i], T[j]) if j != i else np.eye(3)
        tvec = p[min(i + 1, n - 1)] - p[i]
        e[min(i + 1, n - 1)] = se3_screw_norm(R, tvec, length_scale=float(length_scale))
    if n >= 2:
        e[0] = e[1]
    return {
        "e_ref": e,
        "mean_e_ref": float(np.mean(e)),
        "max_e_ref": float(np.max(e)),
        "p95_e_ref": float(np.percentile(e, 95)),
        "length_scale": float(length_scale),
    }


def station_local_geometry(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    window: int = 5,
) -> dict[str, np.ndarray]:
    """
    Per-station local curvature / torsion / E_ref / Δz / horiz step (n stations).
    """
    n = len(x)
    if n < 2:
        z0 = np.zeros(n)
        return {
            "local_curv": z0.copy(),
            "local_curv_roll": z0.copy(),
            "local_torsion": z0.copy(),
            "local_torsion_roll": z0.copy(),
            "e_ref": z0.copy(),
            "e_ref_roll": z0.copy(),
            "d_z": z0.copy(),
            "d_horiz": z0.copy(),
        }
    T = tangents_from_xyz(x, y, z)
    curv = discrete_curvature(T)
    tors = discrete_torsion(T)
    eref = e_ref_joint_strain(x, y, z)
    curv_s = np.zeros(n)
    tors_s = np.zeros(n)
    if len(curv):
        curv_s[1 : 1 + len(curv)] = curv
        curv_s[0] = curv_s[1]
        curv_s[-1] = curv_s[-2] if n > 1 else 0.0
    if len(tors):
        tors_s[2 : 2 + len(tors)] = tors
        tors_s[0] = tors_s[2] if n > 2 else 0.0
        tors_s[1] = tors_s[2] if n > 2 else 0.0
        if n > 3:
            tors_s[-1] = tors_s[-2]
    dz = np.gradient(np.asarray(z, float))
    dx = np.gradient(np.asarray(x, float))
    dy = np.gradient(np.asarray(y, float))
    from rwt.simd import as_f64_contig, vectorized_rolling_mean

    w = max(1, int(window))
    e_arr = as_f64_contig(np.asarray(eref["e_ref"], dtype=float))
    return {
        "local_curv": curv_s,
        "local_curv_roll": vectorized_rolling_mean(curv_s, w),
        "local_torsion": tors_s,
        "local_torsion_roll": vectorized_rolling_mean(tors_s, w),
        "e_ref": e_arr,
        "e_ref_roll": vectorized_rolling_mean(e_arr, w),
        "d_z": dz,
        "d_horiz": np.sqrt(dx * dx + dy * dy),
    }
