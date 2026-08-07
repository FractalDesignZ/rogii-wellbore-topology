#!/usr/bin/env python3
"""Probe downloaded Kaggle layout; write data/processed/layout_report.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rwt.load import discover_layout, iter_well_frames, read_well_csv  # noqa: E402


def main() -> int:
    raw = ROOT / "data" / "raw"
    if not raw.is_dir():
        print("NEED_INFO: data/raw missing — run scripts/download_kaggle.py", file=sys.stderr)
        return 2
    layout = discover_layout(raw)
    wells = []
    try:
        for i, (wid, df) in enumerate(iter_well_frames(raw, "train")):
            wells.append(
                {
                    "well_id": wid,
                    "n": len(df),
                    "cols": list(df.columns)[:40],
                    "has_TVT": "TVT" in df.columns,
                    "has_GR": "GR" in df.columns,
                    "has_XYZ": all(c in df.columns for c in ("X", "Y", "Z")),
                }
            )
            if i >= 24:
                break
    except Exception as e:
        layout["iter_error"] = str(e)[:400]

    # peek first csv if no wells
    if not wells:
        csvs = list(raw.rglob("*.csv"))[:5]
        for p in csvs:
            try:
                df = read_well_csv(p)
                wells.append(
                    {
                        "path": str(p.relative_to(raw)),
                        "n": len(df),
                        "cols": list(df.columns)[:40],
                    }
                )
            except Exception as e:
                wells.append({"path": str(p), "error": str(e)[:200]})

    report = {
        "layout": layout,
        "wells_sample": wells,
        "n_wells_sampled": len(wells),
        "ok": bool(wells),
        "next": "python scripts/eda_report.py --data data/raw",
    }
    out = ROOT / "data" / "processed" / "layout_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2)[:8000])
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
