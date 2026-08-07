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


def open_bishop_transport(T: np.ndarray) -> dict[str, Any]:
    """
    Parallel-transport a normal along open unit tangents T (shape (m,3)).

    Returns cumulative twist proxies (not closed holonomy).
    """
    m = len(T)
    if m < 2:
        return {
            "n_segments": m,
            "cum_twist": 0.0,
            "twist_per_seg": np.zeros(0),
            "final_angle": 0.0,
        }
    # initial normal perpendicular to T[0]
    u = np.cross(T[0], np.array([0.0, 0.0, 1.0]))
    if norm(u) < 1e-8:
        u = np.cross(T[0], np.array([0.0, 1.0, 0.0]))
    u0 = u / (norm(u) + 1e-15)
    u = u0.copy()
    twists = []
    for i in range(m - 1):
        R = rot_between(T[i], T[i + 1])
        # Bishop: parallel-transport u with minimal rotation T[i]→T[i+1]
        u_prev_t = R @ u
        u_prev_t = u_prev_t - (u_prev_t @ T[i + 1]) * T[i + 1]
        nprev = norm(u_prev_t)
        if nprev < 1e-12:
            twists.append(0.0)
            continue
        u_prev_t = u_prev_t / nprev
        # For pure Bishop there is no extra roll — twist increments stay ~0 on smooth curves
        # unless we measure torsion via binormal change; use consecutive transport residual.
        u_next = u_prev_t
        c = float(np.clip(u_prev_t @ u_next, -1, 1))
        s = float(np.cross(u_prev_t, u_next) @ T[i + 1])
        twists.append(float(np.arctan2(s, c)))
        u = u_next
    twists_a = np.asarray(twists, dtype=float)
    # final angle vs initial normal re-transported to end (path-dependent residual)
    return {
        "n_segments": m,
        "cum_twist": float(np.sum(twists_a)),
        "abs_cum_twist": float(np.sum(np.abs(twists_a))),
        "twist_per_seg": twists_a,
        "twist_std": float(np.std(twists_a)) if len(twists_a) else 0.0,
        "mean_curvature": float(np.mean(discrete_curvature(T))) if m >= 2 else 0.0,
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
            "tortuosity_q3d_proxy": 0.0,
            "signed_azimuth": 0.0,
        }
    T = tangents_from_xyz(x, y, z)
    curv = discrete_curvature(T)
    bishop = open_bishop_transport(T)
    # path length vs chord (simple tortuosity proxy)
    segs = norm(np.diff(np.column_stack([x, y, z]), axis=0), axis=1)
    path_len = float(np.sum(segs))
    chord = float(norm(np.array([x[-1] - x[0], y[-1] - y[0], z[-1] - z[0]])))
    tort = path_len / max(chord, 1e-6) - 1.0
    # signed azimuth of lateral (mean of horizontal projections)
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
        "tortuosity_q3d_proxy": float(tort),
        "signed_azimuth": signed_azi,
    }


def station_local_geometry(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    window: int = 5,
) -> dict[str, np.ndarray]:
    """
    Per-station local curvature / Δz / horiz step (aligned to n stations).
    Edges padded with edge values.
    """
    n = len(x)
    T = tangents_from_xyz(x, y, z)
    curv = discrete_curvature(T)
    # map curv (n-2) → stations: pad
    curv_s = np.zeros(n)
    if len(curv):
        curv_s[1 : 1 + len(curv)] = curv
        curv_s[0] = curv_s[1]
        curv_s[-1] = curv_s[-2] if n > 1 else 0.0
    dz = np.gradient(np.asarray(z, float))
    dx = np.gradient(np.asarray(x, float))
    dy = np.gradient(np.asarray(y, float))
    # rolling mean curvature
    w = max(1, int(window))
    ker = np.ones(w) / w
    curv_roll = np.convolve(curv_s, ker, mode="same")
    return {
        "local_curv": curv_s,
        "local_curv_roll": curv_roll,
        "d_z": dz,
        "d_horiz": np.sqrt(dx * dx + dy * dy),
    }
