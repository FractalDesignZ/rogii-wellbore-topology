"""Cryptographic provenance seals for artifacts (prior-art trail)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def public_code_digest(
    roots: list[Path] | None = None,
) -> dict[str, Any]:
    """SHA-256 over concatenated hashes of public Python sources."""
    roots = roots or [
        ROOT / "src" / "rwt",
        ROOT / "scripts",
        ROOT / "tests",
    ]
    files: list[Path] = []
    for r in roots:
        if r.is_dir():
            files.extend(sorted(r.rglob("*.py")))
        elif r.is_file():
            files.append(r)
    # never include private_kernel
    files = [p for p in files if "private_kernel" not in p.parts]
    per_file = {}
    h = hashlib.sha256()
    for p in files:
        digest = _sha256_file(p)
        if digest:
            rel = str(p.relative_to(ROOT)).replace("\\", "/")
            per_file[rel] = digest
            h.update(digest.encode())
            h.update(rel.encode())
    return {
        "n_files": len(per_file),
        "tree_sha256": h.hexdigest(),
        "files": per_file,
    }


def git_head() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(ROOT),
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except Exception:
        return None


def seal_artifact(
    payload: dict[str, Any] | list | str | bytes,
    *,
    kind: str = "artifact",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a seal document: payload digest + public code tree digest + git HEAD.
    """
    if isinstance(payload, (dict, list)):
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    elif isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raw = payload
    code = public_code_digest()
    seal = {
        "artifact": "RWT_PROVENANCE_SEAL",
        "kind": kind,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "payload_sha256": _sha256_bytes(raw),
        "public_code_tree_sha256": code["tree_sha256"],
        "public_code_n_files": code["n_files"],
        "git_head": git_head(),
        "kernel_mode": (os.environ.get("RWT_KERNEL") or "public").lower(),
        "license": "AGPL-3.0-or-later",
        "notice": "Technical prior-art trail; not a patent. Consult IP counsel for filings.",
    }
    if extra:
        seal["extra"] = extra
    # self-hash
    seal["seal_sha256"] = _sha256_bytes(
        json.dumps(seal, sort_keys=True, default=str).encode("utf-8")
    )
    return seal


def write_sealed_json(
    path: Path,
    payload: dict[str, Any],
    *,
    kind: str = "artifact",
) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seal = seal_artifact(payload, kind=kind)
    doc = {"payload": payload, "provenance": seal}
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return seal
