#!/usr/bin/env python3
"""End-to-end smoke without Kaggle data (TSS geometry + baselines + tiny LGBM)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rwt.baselines import (
    baseline_flat_formation,
    baseline_last_known,
    evaluate_baselines_on_known_holdout,
    rmse,
)
from rwt.geometry import trajectory_geometry_features
from rwt.load import synthetic_well
from rwt.middle_out import split_by_mask
from rwt.train import train_lgbm_residual
from rwt.features import build_station_frame


def smoke_main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path.cwd() / "artifacts" / "smoke_synthetic.json")
    args = parser.parse_args()
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
    # labels on predict toe (competition zone) from synthetic TVT_true
    pred_mask = stations["is_predict"].to_numpy()
    st_idx = stations.loc[pred_mask, "station"].to_numpy()
    stations.loc[pred_mask, "TVT"] = lat.loc[st_idx, "TVT_true"].to_numpy()
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
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"SMOKE_OK wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(smoke_main())
