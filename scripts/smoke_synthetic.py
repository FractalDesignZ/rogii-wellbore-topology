#!/usr/bin/env python3
"""End-to-end smoke without Kaggle data (TSS geometry + baselines + tiny LGBM)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rwt.baselines import (  # noqa: E402
    baseline_flat_formation,
    baseline_last_known,
    evaluate_baselines_on_known_holdout,
    rmse,
)
from rwt.geometry import trajectory_geometry_features  # noqa: E402
from rwt.load import synthetic_well  # noqa: E402
from rwt.middle_out import split_by_mask  # noqa: E402
from rwt.train import train_lgbm_residual  # noqa: E402
from rwt.features import build_station_frame  # noqa: E402


def main() -> int:
    lat, tw = synthetic_well(n=180, seed=7)
    sp = split_by_mask(lat, well_id="SYN01")
    geom = trajectory_geometry_features(
        lat["MD"].to_numpy(), lat["X"].to_numpy(), lat["Y"].to_numpy(), lat["Z"].to_numpy()
    )
    base = evaluate_baselines_on_known_holdout(lat.assign(TVT=lat["TVT_true"]), sp)
    # true toe RMSE for baselines using TVT_true
    y_toe = lat.iloc[sp.predict_idx]["TVT_true"].to_numpy()
    p_last = baseline_last_known(lat.assign(TVT=lat["TVT_true"]), sp)
    p_flat = baseline_flat_formation(lat.assign(TVT=lat["TVT_true"]), sp)
    stations = build_station_frame(lat, tw, sp)
    # train on known true TVT
    stations.loc[stations["is_known"], "TVT"] = lat.loc[
        stations.loc[stations["is_known"], "station"].to_numpy(), "TVT_true"
    ].to_numpy()
    result = train_lgbm_residual(stations, n_splits=3)
    report = {
        "geometry": {k: geom[k] for k in ("cum_twist", "abs_cum_twist", "tortuosity_q3d_proxy", "mean_curv")},
        "baselines_holdout": base,
        "baselines_toe_rmse": {
            "last_known": rmse(y_toe, p_last),
            "flat_formation": rmse(y_toe, p_flat),
        },
        "lgbm_cv": result["report"],
        "lineage": "topology-sees-sequence open-curve transport + middle-out prior",
        "ok": True,
    }
    out = ROOT / "artifacts" / "smoke_synthetic.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"SMOKE_OK wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
