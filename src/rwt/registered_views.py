"""Register supplied image traces to allowed wellbore measurements.

The image is retained as a paired source. Only an explicitly declared region
and color enter the numerical trace; truth-aligned panels remain evaluator data.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np


REGION_ROLES = {
    "trajectory": "allowed_source",
    "raw_gr": "allowed_source",
    "full_tvt_alignment": "evaluator_only",
    "formation_tops": "evaluator_only",
}


def region_metadata(kind: str, roi: tuple[int, int, int, int]) -> dict[str, Any]:
    if kind not in REGION_ROLES:
        raise ValueError(f"Unknown image-region kind: {kind!r}")
    return {"kind": kind, "roi_xyxy": list(roi), "role": REGION_ROLES[kind]}


def endpoint_azimuth(x: np.ndarray, y: np.ndarray) -> float:
    """A disclosed geometry-derived fallback, not recovery of a PNG's camera."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError("X and Y must have matching one-dimensional shapes")
    valid = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    if len(valid) < 2:
        raise ValueError("At least two finite XY stations are needed to derive azimuth")
    delta = np.array([x[valid[-1]] - x[valid[0]], y[valid[-1]] - y[valid[0]]])
    if np.linalg.norm(delta) <= 1e-12:
        raise ValueError("Endpoint azimuth is undefined for coincident endpoints")
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def project_trajectory(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    azimuth_deg: float,
    *,
    station_ids: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Rotate XY into along/cross coordinates; preserve Z, IDs and all gaps."""
    x, y, z = (np.asarray(value, dtype=float) for value in (x, y, z))
    if x.ndim != 1 or y.shape != x.shape or z.shape != x.shape or not len(x):
        raise ValueError("XYZ must have equal, nonempty one-dimensional shapes")
    if not np.isfinite(azimuth_deg):
        raise ValueError("Azimuth must be finite")
    ids = np.arange(len(x), dtype=np.int64) if station_ids is None else np.asarray(station_ids).copy()
    if ids.shape != x.shape or len(np.unique(ids)) != len(ids):
        raise ValueError("Station IDs must be unique and match XYZ rows")
    valid_xy = np.isfinite(x) & np.isfinite(y)
    rows = np.flatnonzero(valid_xy)
    if not len(rows):
        raise ValueError("Projection requires at least one finite XY station")
    origin = np.array([x[rows[0]], y[rows[0]]])
    theta = np.radians(azimuth_deg)
    cosine, sine = np.cos(theta), np.sin(theta)
    dx, dy = x - origin[0], y - origin[1]
    u = np.where(valid_xy, dx * cosine + dy * sine, np.nan)
    cross = np.where(valid_xy, -dx * sine + dy * cosine, np.nan)
    z = np.where(np.isfinite(z), z, np.nan)
    return {"station_ids": ids, "u": u, "cross": cross, "z": z, "origin_xy": origin, "azimuth_deg": float(azimuth_deg), "valid_path": np.isfinite(u) & np.isfinite(z)}


def inverse_projection(u: np.ndarray, cross: np.ndarray, z: np.ndarray, origin_xy: np.ndarray, azimuth_deg: float) -> np.ndarray:
    u, cross, z = (np.asarray(value, float) for value in (u, cross, z))
    origin = np.asarray(origin_xy, float)
    if u.ndim != 1 or cross.shape != u.shape or z.shape != u.shape or origin.shape != (2,) or not np.isfinite(origin).all() or not np.isfinite(azimuth_deg):
        raise ValueError("Inverse projection requires matching vectors and a finite XY origin/azimuth")
    theta = np.radians(azimuth_deg)
    return np.column_stack([origin[0] + u * np.cos(theta) - cross * np.sin(theta), origin[1] + u * np.sin(theta) + cross * np.cos(theta), z])


@dataclass(frozen=True)
class AffineCalibration:
    coefficients: np.ndarray  # [u, Z, 1] @ coefficients = [pixel_x, pixel_y]
    rank: int
    control_residual_px: np.ndarray

    def to_pixels(self, world: np.ndarray) -> np.ndarray:
        values = np.asarray(world, float)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("World points must have shape (n, 2)")
        return np.column_stack([values, np.ones(len(values))]) @ self.coefficients

    def to_world(self, pixels: np.ndarray) -> np.ndarray:
        values = np.asarray(pixels, float)
        if values.ndim != 2 or values.shape[1] != 2:
            raise ValueError("Pixel points must have shape (n, 2)")
        return (values - self.coefficients[2]) @ np.linalg.inv(self.coefficients[:2])

    def metadata(self) -> dict[str, Any]:
        return {"status": "control_points_fitted", "independently_validated": False, "rank": self.rank, "coefficients": self.coefficients.tolist(), "control_points": len(self.control_residual_px), "control_rmse_px": float(np.sqrt(np.mean(self.control_residual_px ** 2))), "control_max_residual_px": float(np.max(self.control_residual_px))}


def fit_affine(world: np.ndarray, pixels: np.ndarray, *, max_residual_px: float | None = None) -> AffineCalibration:
    """Fit an invertible affine camera from explicit noncollinear controls.

    Residuals measure agreement with those controls, not independent accuracy.
    Neither a 45-degree construction nor text printed on a raster is assumed.
    """
    world, pixels = np.asarray(world, float), np.asarray(pixels, float)
    if world.ndim != 2 or world.shape[1] != 2 or pixels.shape != world.shape or len(world) < 3:
        raise ValueError("At least three matching two-dimensional control points are required")
    if not np.isfinite(world).all() or not np.isfinite(pixels).all():
        raise ValueError("Control points must be finite")
    design = np.column_stack([world, np.ones(len(world))])
    coefficients, _, rank, _ = np.linalg.lstsq(design, pixels, rcond=None)
    if rank != 3 or np.linalg.matrix_rank(np.column_stack([pixels, np.ones(len(pixels))])) != 3 or np.linalg.matrix_rank(coefficients[:2]) != 2:
        raise ValueError("Control points must define a nondegenerate, invertible affine map")
    residual = np.linalg.norm(design @ coefficients - pixels, axis=1)
    if max_residual_px is not None:
        if not np.isfinite(max_residual_px) or max_residual_px < 0:
            raise ValueError("Control residual tolerance must be finite and nonnegative")
        if np.max(residual) > max_residual_px:
            raise ValueError("Control-point residual exceeds the declared pixel tolerance")
    return AffineCalibration(coefficients, int(rank), residual)


def extract_color_trace(
    image_path: Path,
    roi: tuple[int, int, int, int],
    *,
    color: tuple[int, int, int] = (0, 0, 255),
    tolerance: int = 40,
) -> dict[str, Any]:
    """Read every matching pixel inside the caller's explicit rectangular ROI.

    Candidates are unordered pixel support, not an inferred station sequence.
    No pixels outside the ROI influence the returned numerical candidates.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install rwt[images] to inspect supplied image traces") from exc
    if len(roi) != 4 or any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) for v in roi):
        raise ValueError("ROI must contain four integer coordinates")
    if len(color) != 3 or any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) or not 0 <= v <= 255 for v in color):
        raise ValueError("Color must contain three integer RGB values from 0 to 255")
    if not isinstance(tolerance, (int, np.integer)) or isinstance(tolerance, bool) or not 0 <= tolerance <= 255:
        raise ValueError("Color tolerance must be an integer from 0 to 255")
    image_path = Path(image_path)
    image_bytes = image_path.read_bytes()
    with Image.open(BytesIO(image_bytes)) as image:
        width, height = image.size
        x0, y0, x1, y1 = roi
        if not 0 <= x0 < x1 <= width or not 0 <= y0 < y1 <= height:
            raise ValueError("ROI must be nonempty and strictly contained within the image bounds")
        rgb = np.asarray(image.convert("RGB"))[y0:y1, x0:x1].astype(np.int16)
    mask = np.max(np.abs(rgb - np.asarray(color, dtype=np.int16)), axis=2) <= tolerance
    ys, xs = np.nonzero(mask)
    pixels = np.column_stack([xs + x0, ys + y0]).astype(np.int64)
    return {"candidate_pixels": pixels, "image_dimensions_xy": [width, height], "roi_xyxy": list(roi), "color_rgb": list(color), "color_tolerance": int(tolerance), "image_sha256": hashlib.sha256(image_bytes).hexdigest(), "candidate_order": "unordered raster support; no station correspondence claimed"}


def compare_registered_path(
    projected_xy: np.ndarray,
    candidate_pixels: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    radius_px: float,
    exclude_rois: tuple[tuple[int, int, int, int], ...] = (),
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Measure spatial agreement in both directions, keeping every station row.

    The indices refer to the original station and pixel array rows. Several
    observations can share a nearest neighbour. Distances describe the raster
    rendering at its original resolution.
    """
    from scipy.spatial import cKDTree

    station = np.asarray(projected_xy, dtype=float)
    pixel = np.asarray(candidate_pixels)
    if station.ndim != 2 or station.shape[1] != 2 or not len(station):
        raise ValueError("Projected stations must be a nonempty N by 2 array")
    if pixel.ndim != 2 or pixel.shape[1] != 2 or not np.issubdtype(pixel.dtype, np.integer):
        raise ValueError("Candidate pixels must be an integer N by 2 array")
    if not np.isfinite(radius_px) or radius_px <= 0:
        raise ValueError("Path agreement radius must be finite and positive")

    def in_box(xy, box):
        x0, y0, x1, y1 = box
        return ((x0 <= xy[:, 0]) & (xy[:, 0] < x1)
                & (y0 <= xy[:, 1]) & (xy[:, 1] < y1))

    x0, y0, x1, y1 = roi
    if x0 >= x1 or y0 >= y1 or not in_box(pixel, roi).all():
        raise ValueError("Candidate pixels must lie inside the declared path ROI")
    retained = np.ones(len(pixel), dtype=bool)
    for box in exclude_rois:
        a, b, c, d = box
        if not (x0 <= a < c <= x1 and y0 <= b < d <= y1):
            raise ValueError("Excluded rectangles must be contained in the declared path ROI")
        retained &= ~in_box(pixel, box)

    stations = np.flatnonzero(np.isfinite(station).all(axis=1))
    pixels = np.flatnonzero(retained)
    station_dist = np.full(len(station), np.nan)
    pixel_dist = np.full(len(pixel), np.nan)
    nearest_pixel = np.full(len(station), -1, dtype=np.int64)
    nearest_station = np.full(len(pixel), -1, dtype=np.int64)
    if len(stations) and len(pixels):
        distances, indices = cKDTree(pixel[pixels]).query(station[stations])
        station_dist[stations], nearest_pixel[stations] = distances, pixels[indices]
        distances, indices = cKDTree(station[stations]).query(pixel[pixels])
        pixel_dist[pixels], nearest_station[pixels] = distances, stations[indices]

    def summary(distances):
        finite = distances[np.isfinite(distances)]
        return {
            "total": len(distances), "measured": len(finite),
            "fraction_within_radius": float(np.sum(finite <= radius_px) / len(distances)) if len(distances) else None,
            "median_measured_px": float(np.median(finite)) if len(finite) else None,
            "p95_measured_px": float(np.percentile(finite, 95)) if len(finite) else None,
        }

    arrays = {"station_to_pixel_distance_px": station_dist, "nearest_candidate_row": nearest_pixel,
              "pixel_to_station_distance_px": pixel_dist, "nearest_station_row": nearest_station,
              "retained_candidate_mask": retained}
    report = {
        "radius_px": float(radius_px), "excluded_rois_xyxy": [list(box) for box in exclude_rois],
        "excluded_pixel_count": int((~retained).sum()), "stations_with_finite_projection": len(stations),
        "projected_stations_inside_path_roi": int(in_box(station, roi).sum()),
        "stations_to_retained_pixels": summary(station_dist), "retained_pixels_to_stations": summary(pixel_dist[pixels]),
    }
    return arrays, report


def allowed_measurements(lateral: Any) -> dict[str, np.ndarray]:
    """Select allowed channels, decoding GR nulls and preserving finite XYZ/MD.

    Coordinate sentinel conventions require an explicit source declaration;
    GR null codes alone do not establish one. No target/formation field is read.
    """
    required = ("MD", "X", "Y", "Z")
    if any(name not in lateral for name in required):
        raise ValueError("Lateral input requires MD, X, Y, Z")
    arrays = {name: np.asarray(lateral[name], float).copy() for name in required}
    n = len(arrays["MD"])
    arrays["GR"] = np.asarray(lateral["GR"], float).copy() if "GR" in lateral else np.full(n, np.nan)
    if any(value.shape != (n,) for value in arrays.values()):
        raise ValueError("Allowed channels must have matching one-dimensional shapes")
    from rwt.load import GR_ABSURD_LE, normalize_curve_nulls
    arrays["GR"] = normalize_curve_nulls(arrays["GR"], absurd_le=GR_ABSURD_LE)
    for name, values in list(arrays.items()):
        values[~np.isfinite(values)] = np.nan
        arrays[name + "__present"] = np.isfinite(values)
    return arrays
