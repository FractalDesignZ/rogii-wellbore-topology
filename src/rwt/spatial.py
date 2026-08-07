"""
Spatial neighbor graph for public sheaf-lite TVT transfer.

Public layer only: k-NN on lateral XY centroids, azimuth gate, nearest-XY TVT
restriction. Proprietary sheaf Laplacians stay in private_kernel.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class WellSpatialNode:
    well_id: str
    cx: float
    cy: float
    azimuth_x: float
    azimuth_y: float
    md: np.ndarray
    x: np.ndarray
    y: np.ndarray
    tvt: np.ndarray  # known / train labels; NaN where unknown


@dataclass
class SpatialIndex:
    nodes: dict[str, WellSpatialNode] = field(default_factory=dict)
    max_dist: float = 600.0
    close_dist: float = 150.0

    def add(self, node: WellSpatialNode) -> None:
        self.nodes[node.well_id] = node

    def neighbors(
        self,
        well_id: str,
        *,
        k: int = 5,
        max_dist: float | None = None,
        min_cos_azimuth: float = -0.25,
    ) -> list[tuple[str, float]]:
        """Return [(neighbor_id, xy_dist)] within max_dist, azimuth-compatible."""
        if well_id not in self.nodes:
            return []
        max_dist = float(self.max_dist if max_dist is None else max_dist)
        a = self.nodes[well_id]
        if not (np.isfinite(a.cx) and np.isfinite(a.cy)):
            return []
        hits: list[tuple[str, float]] = []
        for wid, b in self.nodes.items():
            if wid == well_id:
                continue
            if not (np.isfinite(b.cx) and np.isfinite(b.cy)):
                continue
            d = float(np.hypot(a.cx - b.cx, a.cy - b.cy))
            if not np.isfinite(d) or d > max_dist:
                continue
            cos = float(a.azimuth_x * b.azimuth_x + a.azimuth_y * b.azimuth_y)
            na = float(np.hypot(a.azimuth_x, a.azimuth_y))
            nb = float(np.hypot(b.azimuth_x, b.azimuth_y))
            if na > 1e-9 and nb > 1e-9:
                cos = cos / (na * nb)
            if cos < min_cos_azimuth:
                continue
            hits.append((wid, d))
        hits.sort(key=lambda t: t[1])
        return hits[:k]


def well_spatial_node_from_lateral(
    well_id: str,
    df: pd.DataFrame,
    *,
    tvt_col: str = "TVT",
    tvt_input_col: str = "TVT_input",
) -> WellSpatialNode:
    x = df["X"].to_numpy(dtype=float)
    y = df["Y"].to_numpy(dtype=float)
    md = df["MD"].to_numpy(dtype=float) if "MD" in df.columns else np.arange(len(df), dtype=float)
    n = len(x)
    if n == 0:
        return WellSpatialNode(
            well_id=str(well_id),
            cx=0.0,
            cy=0.0,
            azimuth_x=1.0,
            azimuth_y=0.0,
            md=md,
            x=x,
            y=y,
            tvt=np.zeros(0),
        )
    # centroid of lateral body (prefer last 60% / higher MD); guard empty slice
    cut = min(max(0, int(0.4 * n)), n - 1)
    xs, ys = x[cut:], y[cut:]
    if len(xs) == 0 or not np.isfinite(xs).any():
        cx = float(np.nanmean(x[np.isfinite(x)])) if np.isfinite(x).any() else 0.0
        cy = float(np.nanmean(y[np.isfinite(y)])) if np.isfinite(y).any() else 0.0
    else:
        cx = float(np.nanmean(xs[np.isfinite(xs)])) if np.isfinite(xs).any() else float(x[-1])
        cy = float(np.nanmean(ys[np.isfinite(ys)])) if np.isfinite(ys).any() else float(y[-1])
    if not np.isfinite(cx):
        cx = 0.0
    if not np.isfinite(cy):
        cy = 0.0
    dx = float(x[-1] - x[cut]) if n > cut else float(x[-1] - x[0])
    dy = float(y[-1] - y[cut]) if n > cut else float(y[-1] - y[0])
    hn = float(np.hypot(dx, dy)) + 1e-15
    ax, ay = dx / hn, dy / hn
    if tvt_col in df.columns and df[tvt_col].notna().any():
        tvt = df[tvt_col].to_numpy(dtype=float)
    elif tvt_input_col in df.columns:
        tvt = df[tvt_input_col].to_numpy(dtype=float)
    else:
        tvt = np.full(n, np.nan)
    return WellSpatialNode(
        well_id=str(well_id),
        cx=cx,
        cy=cy,
        azimuth_x=ax,
        azimuth_y=ay,
        md=md,
        x=x,
        y=y,
        tvt=tvt,
    )


def neighbor_tvt_transfer(
    target: pd.DataFrame,
    neighbor: WellSpatialNode,
    *,
    close_dist: float = 150.0,
) -> dict[str, np.ndarray]:
    """Nearest neighbor XY sample's TVT (vectorized)."""
    tx = target["X"].to_numpy(dtype=float)
    ty = target["Y"].to_numpy(dtype=float)
    n = len(tx)
    out_tvt = np.full(n, np.nan)
    out_d = np.full(n, np.nan)
    nx, ny, ntvt = neighbor.x, neighbor.y, neighbor.tvt
    finite = np.isfinite(ntvt) & np.isfinite(nx) & np.isfinite(ny)
    if not finite.any():
        return {
            "neighbor_tvt": out_tvt,
            "neighbor_xy_dist": out_d,
            "neighbor_transfer_ok": np.zeros(n, dtype=float),
        }
    nx, ny, ntvt = nx[finite], ny[finite], ntvt[finite]
    # (n, m) distances via broadcasting in chunks if huge
    d2 = (nx[None, :] - tx[:, None]) ** 2 + (ny[None, :] - ty[:, None]) ** 2
    j = np.argmin(d2, axis=1)
    out_tvt = ntvt[j].astype(float)
    out_d = np.sqrt(d2[np.arange(n), j])
    ok = (out_d <= close_dist).astype(float)
    return {
        "neighbor_tvt": out_tvt,
        "neighbor_xy_dist": out_d,
        "neighbor_transfer_ok": ok,
    }


def attach_spatial_features(
    stations: pd.DataFrame,
    lateral: pd.DataFrame,
    index: SpatialIndex | None,
    well_id: str,
    *,
    max_dist: float | None = None,
    close_dist: float | None = None,
) -> pd.DataFrame:
    """Add neighbor transfer columns to a station frame (no-op if no index)."""
    out = stations.copy()
    out["neighbor_tvt"] = np.nan
    out["neighbor_xy_dist"] = np.nan
    out["neighbor_transfer_ok"] = 0.0
    out["neighbor_count"] = 0.0
    out["neighbor_min_dist"] = np.nan
    out["neighbor_offset"] = 0.0
    if index is None or well_id not in index.nodes:
        return out
    max_dist = float(index.max_dist if max_dist is None else max_dist)
    close_dist = float(index.close_dist if close_dist is None else close_dist)
    neigh = index.neighbors(well_id, k=3, max_dist=max_dist)
    out["neighbor_count"] = float(len(neigh))
    if not neigh:
        return out
    out["neighbor_min_dist"] = float(neigh[0][1])
    best_id, best_d = neigh[0]
    node = index.nodes[best_id]
    tr = neighbor_tvt_transfer(lateral, node, close_dist=close_dist)
    # Heel affine offset: align neighbor TVT to this well's heel TVT when both finite
    # (public crown: neighbors share structure but absolute TVT level can bias).
    offset = 0.0
    heel_tvt = None
    for col in ("TVT_input", "TVT", "TVT_true"):
        if col in lateral.columns and lateral[col].notna().any():
            heel_tvt = lateral[col].to_numpy(dtype=float)
            break
    if heel_tvt is not None:
        tr_full = tr["neighbor_tvt"]
        m = np.isfinite(heel_tvt) & np.isfinite(tr_full) & (tr["neighbor_transfer_ok"] > 0.5)
        if m.sum() >= 8:
            offset = float(np.median(heel_tvt[m] - tr_full[m]))
            # refuse runaway offsets (pad mismatch)
            if abs(offset) > 40.0:
                offset = 0.0
    if "station" in out.columns:
        st = out["station"].to_numpy(dtype=int)
        n_lat = len(lateral)
        st = np.clip(st, 0, max(n_lat - 1, 0))
        out["neighbor_tvt"] = tr["neighbor_tvt"][st] + offset
        out["neighbor_xy_dist"] = tr["neighbor_xy_dist"][st]
        out["neighbor_transfer_ok"] = tr["neighbor_transfer_ok"][st]
        out["neighbor_offset"] = float(offset)
        if best_d > close_dist:
            out["neighbor_transfer_ok"] = out["neighbor_transfer_ok"] * 0.25
    return out


def build_spatial_index(
    wells: list[tuple[str, pd.DataFrame]],
    *,
    max_dist: float | None = None,
    close_dist: float | None = None,
) -> SpatialIndex:
    """
    Build index; if radii unset, use field quantiles of centroid distances
    so sparse pads still form a useful graph (coherence on real ROGII scales).
    """
    idx = SpatialIndex()
    for wid, df in wells:
        idx.add(well_spatial_node_from_lateral(wid, df))
    nodes = [n for n in idx.nodes.values() if np.isfinite(n.cx) and np.isfinite(n.cy)]
    if len(nodes) >= 2 and (max_dist is None or close_dist is None):
        pts = np.array([[n.cx, n.cy] for n in nodes], dtype=float)
        # vectorized pairwise upper triangle
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
        iu = np.triu_indices(len(pts), k=1)
        darr = d[iu]
        darr = darr[np.isfinite(darr)]
        if len(darr):
            q05 = float(np.percentile(darr, 5))
            q15 = float(np.percentile(darr, 15))
            if close_dist is None:
                close_dist = max(150.0, q05)
            if max_dist is None:
                max_dist = max(600.0, q15, close_dist * 1.5)
    idx.close_dist = float(close_dist if close_dist is not None else 150.0)
    idx.max_dist = float(max_dist if max_dist is not None else 600.0)
    return idx
