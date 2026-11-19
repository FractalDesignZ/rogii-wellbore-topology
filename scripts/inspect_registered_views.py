#!/usr/bin/env python3
"""Inspect a supplied PNG as a paired, explicitly registered measurement view."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwt.provenance import write_sealed_json  # noqa: E402
from rwt.registered_views import (  # noqa: E402
    allowed_measurements, compare_registered_path, endpoint_azimuth, extract_color_trace, fit_affine,
    project_trajectory, region_metadata,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--lateral", type=Path, required=True)
    parser.add_argument("--azimuth-deg", type=float, help="declared projection azimuth; fallback uses XY endpoints")
    parser.add_argument("--depth-unit", choices=("ft", "m"), required=True, help="declared table coordinate unit; no raster label is trusted")
    parser.add_argument("--path-roi", type=int, nargs=4, required=True, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--path-color", type=int, nargs=3, default=(0, 0, 255), metavar=("R", "G", "B"))
    parser.add_argument("--color-tolerance", type=int, default=40)
    parser.add_argument("--control-points", type=Path, help="JSON object with world:[[u,Z],...] and pixels:[[x,y],...]")
    parser.add_argument("--max-control-residual-px", type=float)
    parser.add_argument("--check-path-radius-px", type=float, help="Measure path and pixel coverage within this radius after registration")
    parser.add_argument("--exclude-path-roi", type=int, nargs=4, action="append", default=[],
                        metavar=("X0", "Y0", "X1", "Y1"), help="Exclude a legend or other off-path pixels from the distance check")
    parser.add_argument("--out", type=Path, required=True, help="fresh artifact directory")
    args = parser.parse_args(argv)
    if args.check_path_radius_px is not None and args.control_points is None:
        parser.error("--check-path-radius-px requires --control-points")
    if args.exclude_path_roi and args.check_path_radius_px is None:
        parser.error("--exclude-path-roi requires --check-path-radius-px")
    if args.out.exists():
        raise FileExistsError(args.out)
    frame = pd.read_csv(args.lateral, usecols=lambda column: column in {"MD", "X", "Y", "Z", "GR"})
    arrays = allowed_measurements(frame)
    angle = args.azimuth_deg if args.azimuth_deg is not None else endpoint_azimuth(arrays["X"], arrays["Y"])
    projection = project_trajectory(arrays["X"], arrays["Y"], arrays["Z"], angle)
    trace = extract_color_trace(args.image, tuple(args.path_roi), color=tuple(args.path_color), tolerance=args.color_tolerance)
    numerical = {**arrays, **{name: projection[name] for name in ("station_ids", "u", "cross", "z", "origin_xy", "valid_path")}, "candidate_pixels": trace.pop("candidate_pixels")}
    calibration = {"status": "unregistered_candidates", "independently_validated": False}
    control_hash = None
    if args.control_points:
        raw_controls = args.control_points.read_bytes()
        controls = json.loads(raw_controls)
        world, pixels = np.asarray(controls["world"], float), np.asarray(controls["pixels"], float)
        camera = fit_affine(world, pixels, max_residual_px=args.max_control_residual_px)
        numerical["control_world"] = world
        numerical["control_pixels"] = pixels
        numerical["projected_station_pixels"] = camera.to_pixels(np.column_stack([projection["u"], projection["z"]]))
        calibration = camera.metadata()
        control_hash = hashlib.sha256(raw_controls).hexdigest()
    elif args.max_control_residual_px is not None:
        parser.error("--max-control-residual-px requires --control-points")
    evidence = {
        "schema": "rwt.registered-image-view.v1",
        "source_image": str(args.image.resolve()),
        "source_image_availability": "training_only",
        "source_image_sha256_is_provenance_not_feature": True,
        "new_serve_time_measurement_claim": False,
        "source_columns": ["MD", "X", "Y", "Z", "GR"],
        "region": region_metadata("trajectory", tuple(args.path_roi)),
        "other_region_roles": {"raw_gr": "allowed_source", "full_tvt_alignment": "evaluator_only", "formation_tops": "evaluator_only"},
        "view_kind": "supplied_png_paired_with_allowed_table_projection",
        "source_depth_unit": args.depth_unit,
        "azimuth_deg": float(angle),
        "azimuth_source": "declared" if args.azimuth_deg is not None else "XY_endpoint_fallback_not_recovered_PNG_camera",
        "station_count": len(arrays["MD"]),
        "valid_path_stations": int(np.sum(projection["valid_path"])),
        "candidate_pixels": len(numerical["candidate_pixels"]),
        "trace": trace,
        "calibration": calibration,
        "control_points_sha256": control_hash,
        "note": "Candidate pixel support has no station ordering or inferred target TVT; paired truth panels are not conditioning inputs.",
    }
    if args.check_path_radius_px is not None:
        distances, evidence["path_agreement"] = compare_registered_path(
            numerical["projected_station_pixels"], numerical["candidate_pixels"], tuple(args.path_roi),
            radius_px=args.check_path_radius_px, exclude_rois=tuple(map(tuple, args.exclude_path_roi)),
        )
        numerical.update(distances)
    args.out.mkdir(parents=True, exist_ok=False)
    array_path = args.out / "registered_view_arrays.npz"
    np.savez_compressed(array_path, **numerical)
    evidence["arrays_sha256"] = hashlib.sha256(array_path.read_bytes()).hexdigest()
    write_sealed_json(args.out / "registration_evidence.json", evidence, kind="registered_image_ingress")
    print(json.dumps({"out": str(args.out.resolve()), "candidate_pixels": evidence["candidate_pixels"], "calibration": calibration, "source_image_availability": "training_only"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
