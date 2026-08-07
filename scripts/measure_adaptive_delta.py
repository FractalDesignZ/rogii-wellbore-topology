"""Measure adaptive residual serve vs global residual / pure prior.

Usage
-----
  # synthetic sanity
  python scripts/measure_adaptive_delta.py --synthetic --out artifacts/adaptive_delta_synth

  # real slice
  python scripts/measure_adaptive_delta.py --data data/raw --max-wells 24 --folds 3 \\
      --out artifacts/adaptive_delta_24w --workers 6

Emits:
  - cv_report.json (full train seal with adaptive fields)
  - adaptive_delta.json (compact comparison table)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# allow running from repo root without install
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rwt.train import _assemble_dataset, train_lgbm_residual  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Measure adaptive residual delta")
    ap.add_argument("--data", default="", help="Kaggle data root")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="artifacts/adaptive_delta")
    ap.add_argument("--max-wells", type=int, default=0)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--prior-col", default=None)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stations, gate2 = _assemble_dataset(
        Path(args.data) if args.data else None,
        synthetic=args.synthetic or not args.data,
        max_wells=args.max_wells,
        workers=args.workers if args.workers > 0 else None,
    )
    stations.to_csv(out / "stations.csv", index=False)

    result = train_lgbm_residual(
        stations,
        n_splits=args.folds,
        prior_col=args.prior_col,
        gate2_meta=gate2,
    )
    report = result["report"]
    (out / "cv_report.json").write_text(json.dumps(report, indent=2) + "\n")

    delta = {
        "n_wells": report.get("n_wells"),
        "n_stations": report.get("n_stations"),
        "oof_rmse_global_residual": report.get("oof_rmse"),
        "oof_rmse_selected": report.get("oof_rmse_selected"),
        "use_residual": report.get("use_residual"),
        "adaptive_oof_rmse": report.get("adaptive_oof_rmse"),
        "adaptive_n_fire": report.get("adaptive_n_fire"),
        "adaptive_n_fallback": report.get("adaptive_n_fallback"),
        "baseline_rmse": report.get("baseline_rmse"),
        "final_prior_type": report.get("final_prior_type"),
        "delta_adaptive_vs_global": None,
        "delta_adaptive_vs_best_floor": None,
    }
    a = report.get("adaptive_oof_rmse")
    g = report.get("oof_rmse")
    floors = report.get("baseline_rmse") or {}
    best_floor = min((v for v in floors.values() if v is not None), default=None)
    if a is not None and g is not None:
        delta["delta_adaptive_vs_global"] = float(a) - float(g)
    if a is not None and best_floor is not None:
        delta["delta_adaptive_vs_best_floor"] = float(a) - float(best_floor)

    (out / "adaptive_delta.json").write_text(json.dumps(delta, indent=2) + "\n")
    print(json.dumps(delta, indent=2))
    print(f"\nwrote {out / 'adaptive_delta.json'}")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
