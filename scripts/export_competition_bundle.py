#!/usr/bin/env python3
"""
Competition isolation export.

Exports ONLY:
  - submission-shaped CSV (predictions)
  - optional public feature matrix (watermarked)
  - provenance seal

NEVER copies private_kernel source into the bundle.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rwt.provenance import seal_artifact, write_sealed_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Export competition-safe bundle")
    ap.add_argument(
        "--pred",
        required=True,
        help="path to predictions CSV (well_id,MD,TVT or Id,TVT)",
    )
    ap.add_argument(
        "--features",
        default="",
        help="optional public feature CSV (will strip private_* columns)",
    )
    ap.add_argument(
        "--out",
        default=str(ROOT / "submissions" / "competition_bundle"),
    )
    ap.add_argument(
        "--note",
        default="Late entry predictions only — proprietary kernel not included.",
    )
    args = ap.parse_args()

    pred_path = Path(args.pred)
    if not pred_path.is_file():
        print(f"FAIL missing predictions: {pred_path}", file=sys.stderr)
        return 2

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # copy predictions only
    dest_pred = out / "submission.csv"
    shutil.copy2(pred_path, dest_pred)

    manifest: dict = {
        "bundle": "rwt_competition_isolation",
        "includes": ["submission.csv", "MANIFEST.sealed.json"],
        "excludes": [
            "private_kernel/**",
            "src/rwt/** (source not bundled)",
            "raw training data",
        ],
        "note": args.note,
        "kernel_mode_env": "export uses precomputed preds — no private source",
    }

    if args.features:
        import pandas as pd

        fpath = Path(args.features)
        if fpath.is_file():
            df = pd.read_csv(fpath)
            # strip anything that looks proprietary (incl. private beam residual cols)
            drop = [
                c
                for c in df.columns
                if c.startswith("private_")
                or c.startswith("sheaf_")
                or c.startswith("prior_beam")
            ]
            df = df.drop(columns=drop, errors="ignore")
            # keep watermark if present
            pub = out / "features_public.csv"
            df.to_csv(pub, index=False)
            manifest["includes"].append("features_public.csv")
            manifest["stripped_columns"] = drop

    seal = write_sealed_json(out / "MANIFEST.sealed.json", manifest, kind="competition_bundle")
    # belt: ensure private_kernel never present
    assert not (out / "private_kernel").exists()
    print(json.dumps({"ok": True, "out": str(out), "seal": seal.get("seal_sha256")}, indent=2))
    print("COMPETITION_BUNDLE_OK — upload submission.csv only; keep kernel private")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
