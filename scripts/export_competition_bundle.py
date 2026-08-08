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

    import pandas as pd

    # Validate prediction schema before any copy (reject private-prefixed cols).
    pred_df = pd.read_csv(pred_path)
    priv_prefix = (
        "private_",
        "sheaf_",
        "prior_beam",
        "prior_lattice",
        "prior_heatmap",
    )
    bad = [
        c
        for c in pred_df.columns
        if any(str(c).casefold().startswith(p) for p in priv_prefix)
    ]
    if bad:
        print(f"FAIL proprietary columns in --pred: {bad}", file=sys.stderr)
        return 2
    cols = {str(c).casefold() for c in pred_df.columns}
    ok_schema = (
        {"well_id", "md", "tvt"}.issubset(cols)
        or {"id", "tvt"}.issubset(cols)
        or {"well_id", "tvt"}.issubset(cols)
    )
    if not ok_schema:
        print(
            "FAIL --pred must be well_id,MD,TVT or Id,TVT (plus optional well_id,TVT); "
            f"got columns={list(pred_df.columns)}",
            file=sys.stderr,
        )
        return 2

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # write validated predictions only (no proprietary columns)
    dest_pred = out / "submission.csv"
    pred_df.to_csv(dest_pred, index=False)

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
        "pred_columns": list(pred_df.columns),
    }

    if args.features:
        fpath = Path(args.features)
        if fpath.is_file():
            df = pd.read_csv(fpath)
            # strip anything that looks proprietary (incl. private beam residual cols)
            drop = [
                c
                for c in df.columns
                if any(
                    str(c).casefold().startswith(p)
                    for p in (
                        "private_",
                        "sheaf_",
                        "prior_beam",
                        "prior_lattice",
                        "prior_heatmap",
                    )
                )
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
