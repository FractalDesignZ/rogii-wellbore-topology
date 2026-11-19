#!/usr/bin/env python3
"""Freeze filename-only well groups and copy a development-only pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any


SPLIT_SALT = "rwt-master-v1"
LATERAL_SUFFIX = "__horizontal_well.csv"
TYPEWELL_SUFFIX = "__typewell.csv"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def rank_well(well_id: str) -> str:
    return hashlib.sha256(f"{SPLIT_SALT}/{well_id}".encode("utf-8")).hexdigest()


def assign_wells(well_ids: list[str]) -> dict[str, Any]:
    """Assign groups using identifiers only, before reading measurement bytes."""
    if not well_ids:
        raise ValueError("No training wells found")
    if len({wid.casefold() for wid in well_ids}) != len(well_ids):
        raise ValueError("Duplicate well identifiers are not allowed")
    ordered = sorted(well_ids, key=lambda wid: (rank_well(wid), wid))
    confirmation = [wid for wid in ordered if int(rank_well(wid)[:8], 16) % 5 == 0]
    confirmation_set = set(confirmation)
    development = [wid for wid in ordered if wid not in confirmation_set]
    if not development or not confirmation:
        raise ValueError("Both development and confirmation groups must be nonempty")
    assignment = {
        "split_salt": SPLIT_SALT,
        "rule": "SHA256(salt + '/' + well_id); confirmation iff int(first8hex,16) % 5 == 0",
        "selection_order": "ascending full SHA256, then well_id",
        "all_well_ids": sorted(well_ids),
        "development_well_ids": development,
        "confirmation_well_ids": confirmation,
    }
    assignment["assignment_sha256"] = hashlib.sha256(canonical(assignment)).hexdigest()
    return assignment


def discover_pairs(data_root: Path) -> dict[str, tuple[Path, Path]]:
    """Read filenames and metadata only; require a complete one-to-one pairing."""
    train = (data_root / "train").resolve(strict=True)
    if not train.is_dir():
        raise ValueError("Dataset needs a train directory")
    paths: dict[str, dict[str, Path]] = {"lateral": {}, "typewell": {}}
    folded: dict[str, str] = {}
    for kind, suffix in (("lateral", LATERAL_SUFFIX), ("typewell", TYPEWELL_SUFFIX)):
        for path in sorted(train.glob("*" + suffix)):
            if path.is_symlink() or not path.is_file() or path.resolve().parent != train:
                raise ValueError("Training files must be regular files inside train")
            wid = path.name[: -len(suffix)]
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", wid):
                raise ValueError(f"Invalid filename well identifier: {wid!r}")
            previous = folded.setdefault(wid.casefold(), wid)
            if previous != wid or wid in paths[kind]:
                raise ValueError("Duplicate or case-inconsistent well identifiers")
            paths[kind][wid] = path
    if not paths["lateral"]:
        raise ValueError("No training wells found")
    if paths["lateral"].keys() != paths["typewell"].keys():
        raise ValueError("Every horizontal well requires exactly one matching typewell; orphan files are refused")
    return {wid: (paths["lateral"][wid], paths["typewell"][wid]) for wid in sorted(paths["lateral"])}


def hash_file(path: Path) -> str:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed while hashing: {path.name}")
    return digest.hexdigest()


def overlap(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def prepare_master_cohort(
    data_root: Path,
    out_dir: Path,
    pilot_data: Path,
    *,
    pilot_wells: int = 12,
) -> dict[str, Any]:
    """Prepare a grouped pilot without parsing labels or changing source data.

    Both output locations must be fresh and disjoint from each other and the
    source dataset. Only development CSV pairs enter the trainer's train/ tree.
    """
    data_root = Path(data_root).resolve(strict=True)
    out_dir, pilot_data = Path(out_dir).resolve(), Path(pilot_data).resolve()
    if overlap(out_dir, pilot_data) or overlap(data_root, out_dir) or overlap(data_root, pilot_data):
        raise ValueError("Source, artifact, and pilot directories must be disjoint")
    if os.path.lexists(out_dir) or os.path.lexists(pilot_data):
        raise FileExistsError("Artifact and pilot directories must both be fresh")
    pairs = discover_pairs(data_root)
    assignment = assign_wells(list(pairs))  # no file contents read before this assignment
    if not isinstance(pilot_wells, int) or isinstance(pilot_wells, bool) or not 1 <= pilot_wells <= len(assignment["development_well_ids"]):
        raise ValueError("pilot_wells must be positive and no larger than the development group")
    pilot_ids = assignment["development_well_ids"][:pilot_wells]
    file_records: dict[str, list[dict[str, Any]]] = {}
    for wid, files in pairs.items():
        file_records[wid] = [
            {"path": path.relative_to(data_root).as_posix(), "bytes": path.stat().st_size, "sha256": hash_file(path)}
            for path in files
        ]
    if list(discover_pairs(data_root)) != list(pairs):
        raise RuntimeError("Source membership changed during preparation")
    frozen = {"assignment": assignment, "files": file_records, "pilot_well_ids": pilot_ids}
    manifest = {
        "schema": "rwt.master-cohort.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_data_root": str(data_root),
        "pilot_data_root": str(pilot_data),
        "trainer_data_argument": str(pilot_data),
        **assignment,
        "pilot_well_ids": pilot_ids,
        "counts": {"all": len(pairs), "development": len(assignment["development_well_ids"]), "confirmation": len(assignment["confirmation_well_ids"]), "pilot": len(pilot_ids)},
        "files": file_records,
        "input_sha256": hashlib.sha256(canonical(frozen)).hexdigest(),
        "confirmation_unmeasured": True,
        "confirmation_use": "membership and file hashes only; no parsing, measurement, or pilot copy",
        "status": "prepared_development_only",
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    pilot_data.mkdir(parents=True, exist_ok=False)
    train_out = pilot_data / "train"
    train_out.mkdir()
    for wid in pilot_ids:
        for source, record in zip(pairs[wid], file_records[wid]):
            target = train_out / source.name
            target.resolve().relative_to(pilot_data)
            if os.path.lexists(target):
                raise FileExistsError(target)
            shutil.copy2(source, target)
            if hash_file(target) != record["sha256"]:
                raise RuntimeError(f"Pilot copy does not match frozen input: {source.name}")
    with (out_dir / "cohort_manifest.json").open("x", encoding="utf-8") as target:
        json.dump(manifest, target, indent=2, allow_nan=False)
        target.write("\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="dataset root containing train/")
    parser.add_argument("--out", type=Path, required=True, help="fresh artifact directory")
    parser.add_argument("--pilot-data", type=Path, required=True, help="fresh trainer data root; train/ is created inside")
    parser.add_argument("--pilot-wells", type=int, default=12)
    args = parser.parse_args(argv)
    manifest = prepare_master_cohort(args.data, args.out, args.pilot_data, pilot_wells=args.pilot_wells)
    print(json.dumps({key: manifest[key] for key in ("schema", "counts", "assignment_sha256", "input_sha256", "trainer_data_argument", "confirmation_unmeasured", "status")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
