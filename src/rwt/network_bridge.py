"""Label-free wellbore inputs for an optional, separately installed Network Engine.

This module defines an interchange boundary, not a dynamics implementation.
It neither imports nor bundles the proprietary engine on the public path.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import warnings

import numpy as np
import pandas as pd

from rwt.provenance import write_sealed_json
from rwt.load import GR_ABSURD_LE, LAS_NULL_SENTINELS, normalize_curve_nulls


SCHEMA = "rwt.network-input.v1"
DEPTH_SCALES = {"ft": 0.3048, "m": 1.0}
SOURCE_CHANNELS = ("MD", "Z", "GR")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass
class NetworkInput:
    data: dict[str, dict[str, np.ndarray]]
    spec: dict[str, Any]
    mode: dict[str, Any]
    manifest: dict[str, Any]

    def write(self, directory: Path) -> None:
        """Write into a fresh directory; never overwrite an earlier receipt."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(
            directory / "data.npz",
            **{f"{stream}/{key}": array for stream, values in self.data.items() for key, array in values.items()},
        )
        for name, value in (("spec", self.spec), ("mode", self.mode)):
            (directory / f"{name}.json").write_bytes(_canonical(value) + b"\n")
        write_sealed_json(directory / "manifest.json", self.manifest, kind=SCHEMA)


def build_network_input(
    lateral: pd.DataFrame,
    *,
    well_id: str,
    depth_unit: str,
) -> NetworkInput:
    """Export raw MD/Z/GR only, preserving acquisition order and signal gaps.

    Depths are converted to metres. The native container's ``t`` is a sample
    ordinal, explicitly not measured seconds. The analysis index is MD; native
    carried sample-order records remain nonphysical metadata.
    TVT, TVT_true, TVT_input, formation tops, and model/private columns are never
    read. Z and GR may be absent, but MD must be finite on every row.
    """
    if depth_unit not in DEPTH_SCALES:
        raise ValueError("depth_unit must be explicitly declared as 'ft' or 'm'")
    if len(lateral) < 4:
        raise ValueError("Network input needs at least four stations")
    if "well_id" in lateral and lateral["well_id"].dropna().astype(str).nunique() > 1:
        raise ValueError("Network input must contain one well; split multi-well tables first")
    if "MD" not in lateral:
        raise ValueError("Network input requires MD")
    md = lateral["MD"].to_numpy(dtype=float, copy=True)
    if not np.isfinite(md).all():
        raise ValueError("MD must be finite; do not fill or silently drop backbone rows")
    factor = DEPTH_SCALES[depth_unit]
    preprocessing = {"las_null_sentinels": list(LAS_NULL_SENTINELS), "gr_absurd_le": GR_ABSURD_LE, "gap_fill": False}
    values: dict[str, np.ndarray] = {"t": np.arange(len(lateral), dtype=np.float64)}
    channels: list[dict[str, Any]] = []
    support: dict[str, Any] = {}
    for name in SOURCE_CHANNELS:
        raw = lateral[name].to_numpy(dtype=float, copy=True) if name in lateral else np.full(len(lateral), np.nan)
        if name in ("GR", "Z"):
            raw = normalize_curve_nulls(raw, absurd_le=GR_ABSURD_LE if name == "GR" else None)
        present = np.isfinite(raw)
        raw[~present] = np.nan
        if name in ("MD", "Z"):
            raw *= factor
            unit: str | dict[str, Any] = "m"
        else:
            # An opaque instrument unit, not a fabricated physical range.
            unit = {"dims": {}, "extra": {"GR_source": "1"}, "scale": 1.0, "symbol": "GR_source"}
        values[name] = raw
        values[name + "__present"] = present
        channels.append({"name": name, "unit": unit})
        starts = np.flatnonzero(present & ~np.r_[False, present[:-1]])
        support[name] = {
            "observed": int(present.sum()),
            "missing": int((~present).sum()),
            "contiguous_spans": int(len(starts)),
        }
    spec = {
        "description": "Label-free wellbore measurements; station_order is ordinal, analysis axis is MD in metres.",
        "streams": [{"name": "well", "clock": "station_order", "channels": channels}],
        "selected_indices": [{"stream": "well", "channel": "MD", "direction": 1}],
        "relations": [
            {"type": "time_adjacency", "stream": "well", "channels": [name], "sigma": {name: 1.0}}
            for name in ("Z", "GR")
        ] + [{"type": "dynamic_linear", "stream": "well", "src": "Z", "dst": "GR", "sigma": 1.0}],
    }
    mode = {
        "preset": "index_default",
        "detector": "slice",
        "events": [],
        "weights": {"basis": "observed"},
        "relation_types": ["time_adjacency", "dynamic_linear"],
        # Slice detection is independent of the legacy windowed dynamics level.
        "levels": {"relation_nodes": 0, "dynamics": False, "variance": False, "coupling": False, "self_similarity": False},
    }
    digest = hashlib.sha256()
    for key, array in sorted(values.items()):
        canonical_array = np.ascontiguousarray(array.astype("<f8") if array.dtype.kind == "f" else array)
        digest.update(_canonical({"key": key, "dtype": canonical_array.dtype.str, "shape": list(array.shape)}))
        digest.update(canonical_array.tobytes())
    data_hash = digest.hexdigest()
    settings_hash = hashlib.sha256(_canonical({"spec": spec, "mode": mode, "preprocessing": preprocessing})).hexdigest()
    manifest = {
        "schema": SCHEMA,
        "well_id": str(well_id),
        "n_stations": len(lateral),
        "channels": list(SOURCE_CHANNELS),
        "source_depth_unit": depth_unit,
        "analysis_axis": {"channel": "MD", "unit": "m", "order": "acquisition", "sort_or_resample": False},
        "container_clock": "sample ordinal; not measured time or seconds",
        "label_policy": "no target labels or model features exported",
        "support": support,
        "preprocessing": preprocessing,
        "data_sha256": data_hash,
        "settings_sha256": settings_hash,
        "input_sha256": hashlib.sha256(_canonical({"data": data_hash, "settings": settings_hash, "well_id": str(well_id)})).hexdigest(),
        "purpose": "offline relational diagnostics; no predictive improvement or physical-cause claim",
    }
    return NetworkInput({"well": values}, spec, mode, manifest)


def run_local_engine(bundle: NetworkInput, directory: Path) -> dict[str, Any]:
    """Execute the installed native slice API. Missing/unsupported engines fail.

    Native outputs remain local. This does not train, replace a TVT predictor,
    change engine governance, or fall back to a different detector.
    """
    try:
        from network_engine.kernel import run
    except ImportError as exc:
        raise RuntimeError("Install your separately licensed Network Engine locally to use --run-local") from exc
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    result = run(bundle.data, bundle.spec, bundle.mode)
    summary = json.loads(result.summary_json())
    slices = summary.get("levels", {}).get("slices")
    if not isinstance(slices, dict) or slices.get("detector") != "slice":
        raise RuntimeError("Installed engine did not execute the required slice detector")
    if summary.get("levels", {}).get("dynamics") is not None:
        raise RuntimeError("Legacy windowed dynamics must remain disabled")
    if "S_slice" not in summary.get("operators_applied", []):
        raise RuntimeError("Installed engine did not apply the native slice operator")
    if summary.get("all_property_checks_hold") is not True:
        raise RuntimeError("Native engine property checks failed")
    directory.mkdir(parents=True, exist_ok=False)
    with warnings.catch_warnings(record=True) as save_warnings:
        warnings.simplefilter("always")
        result.save(str(directory))
    artifacts = {}
    for name in ("summary.json", "mode.json", "arrays.npz"):
        path = directory / name
        if not path.is_file():
            raise RuntimeError(f"Native save omitted required artifact {name}")
        artifacts[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if json.loads((directory / "summary.json").read_text(encoding="utf-8")) != summary:
        raise RuntimeError("Saved native summary does not match the inspected result")
    saved_mode = json.loads((directory / "mode.json").read_text(encoding="utf-8"))
    if saved_mode != summary.get("mode") or hashlib.sha256(_canonical(saved_mode)).hexdigest() != summary.get("mode_hash"):
        raise RuntimeError("Saved native mode does not match the inspected result")
    with np.load(directory / "arrays.npz", allow_pickle=False) as arrays:
        for key in arrays.files:
            arrays[key]  # reject object/pickle payloads before issuing a receipt
    nodes = [record for record in slices.get("nodes", []) if record.get("channel") != "__t__"]
    relations = slices.get("relations", [])
    supported_nodes = sum("skipped" not in record and "changes" in record for record in nodes)
    supported_relations = sum("skipped" not in record and "segments" in record for record in relations)
    provider_module = sys.modules.get(run.__module__)
    provider_file = getattr(provider_module, "__file__", None)
    provider_hash = None
    if provider_file:
        # Bind the locally used implementation without copying any source out.
        root = Path(provider_file).resolve().parent.parent
        source_digest = hashlib.sha256()
        for path in sorted(root.rglob("*.py")):
            source_digest.update(path.relative_to(root).as_posix().encode())
            source_digest.update(hashlib.sha256(path.read_bytes()).digest())
        provider_hash = source_digest.hexdigest()
    receipt = {
        "schema": "rwt.network-run.v1",
        "input_sha256": bundle.manifest["input_sha256"],
        "kernel_version": summary.get("kernel_version"),
        "provider_source_sha256": provider_hash,
        "mode_hash": summary.get("mode_hash"),
        "spec_hash": summary.get("spec_hash"),
        "data_hash": summary.get("data_hash"),
        "detector": "slice",
        "all_property_checks_hold": True,
        "node_records": len(nodes),
        "relation_records": len(relations),
        "supported_node_records": supported_nodes,
        "analyzed_relation_records": supported_relations,
        "native_artifact_sha256": artifacts,
        "save_warnings": [str(warning.message) for warning in save_warnings],
        "viewer_involvement_available": (directory / "involvement.json").is_file(),
        "status": "diagnostics_only" if supported_relations else "insufficient_relation_support",
    }
    write_sealed_json(directory / "bridge_receipt.json", receipt, kind="network_diagnostics")
    return receipt
