#!/usr/bin/env python3
"""
EDA + abstraction-mode trace on real or synthetic data.
Writes artifacts/eda_report.json for the late-entry certificate trail.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rwt.abstraction import run_well_modes  # noqa: E402
from rwt.load import iter_well_frames, load_typewell, synthetic_well  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="", help="Kaggle data root (data/raw)")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--max-wells", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "eda_report.json"))
    args = ap.parse_args()

    traces = []
    if args.synthetic or not args.data:
        lat, tw = synthetic_well(n=200, seed=3)
        st, tr = run_well_modes(lat, tw, well_id="SYN01", y_true_col="TVT_true")
        traces.append(
            {
                "well_id": "SYN01",
                "trace": tr.to_dict(),
                "n_stations": len(st),
                "n_predict": int(st["is_predict"].sum()),
            }
        )
    else:
        root = Path(args.data)
        tw = None
        for cand in root.rglob("*typewell*.csv"):
            tw = load_typewell(cand)
            break
        n = 0
        for wid, df in iter_well_frames(root, "train"):
            # attach true TVT for known-only diagnostics if present
            ycol = "TVT" if "TVT" in df.columns else None
            st, tr = run_well_modes(df, tw, well_id=wid, y_true_col=ycol)
            traces.append(
                {
                    "well_id": wid,
                    "trace": tr.to_dict(),
                    "n_stations": len(st),
                    "n_predict": int(st["is_predict"].sum()) if "is_predict" in st else None,
                }
            )
            n += 1
            if n >= args.max_wells:
                break
        if not traces:
            print("NEED_INFO: no wells parsed — check layout via probe_data.py", file=sys.stderr)
            return 2

    report = {
        "artifact": "eda_report",
        "thesis": "abstraction modes M0–M6 (TSS lineage) on wellbore TVT late entry",
        "n_wells": len(traces),
        "wells": traces,
        "ok": True,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "n_wells": len(traces), "out": str(out)}, indent=2))
    # print first well mode trail
    if traces:
        print("modes:", traces[0]["trace"]["modes_executed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
