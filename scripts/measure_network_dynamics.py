#!/usr/bin/env python3
"""Prepare label-free wellbore inputs and optionally run a local slice engine."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rwt.network_bridge import build_network_input, run_local_engine  # noqa: E402


def synthetic_lateral() -> pd.DataFrame:
    """Practice fixture: changing GR/Z relation plus an explicit GR gap."""
    rng = np.random.default_rng(20261118)
    n = 240
    md = np.arange(n, dtype=float) * 2.0
    z = -2500.0 + 3.0 * np.sin(np.arange(n) / 17.0)
    gr = 80.0 + np.where(np.arange(n) < n // 2, 4.0, -4.0) * (z + 2500.0)
    gr += rng.normal(0.0, 0.2, n)
    gr[60:68] = np.nan
    return pd.DataFrame({"MD": md, "Z": z, "GR": gr})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="one raw lateral CSV, preserving source row order")
    source.add_argument("--synthetic", action="store_true", help="practice-only built-in metre fixture")
    parser.add_argument("--well-id", help="defaults to CSV filename or 'practice' for synthetic input")
    parser.add_argument("--depth-unit", choices=("ft", "m"), help="required for --csv; fixture is in m")
    parser.add_argument("--out", type=Path, help="fresh output directory; never overwritten")
    parser.add_argument("--run-local", action="store_true", help="invoke separately installed native Network Engine")
    args = parser.parse_args(argv)
    if args.csv and args.depth_unit is None:
        parser.error("--csv requires --depth-unit; units are never inferred")
    if args.synthetic and args.depth_unit not in (None, "m"):
        parser.error("the synthetic fixture is declared in metres")
    lateral = synthetic_lateral() if args.synthetic else pd.read_csv(args.csv)
    well_id = args.well_id or ("practice" if args.synthetic else args.csv.stem)
    bundle = build_network_input(lateral, well_id=well_id, depth_unit=args.depth_unit or "m")
    bundle.manifest["synthetic"] = bool(args.synthetic)
    out = args.out or Path("artifacts") / ("network_dynamics_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    bundle.write(out)
    report = {"input": str(out.resolve()), "input_sha256": bundle.manifest["input_sha256"], "status": "prepared"}
    if args.run_local:
        report["native"] = run_local_engine(bundle, out / "native")
        report["status"] = "diagnostics_complete" if report["native"]["status"] == "diagnostics_only" else "insufficient_relation_support"
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
