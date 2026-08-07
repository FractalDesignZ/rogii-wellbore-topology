"""GroupKFold-by-well training (LightGBM residual on prior_flat_tvt)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from rwt.baselines import rmse
from rwt.features import FEATURE_COLS_DEFAULT, build_station_frame, feature_matrix
from rwt.load import iter_well_frames, load_typewell, synthetic_well
from rwt.middle_out import split_by_mask, split_by_tvt_nan


def _assemble_dataset(
    data_root: Path | None,
    *,
    synthetic: bool = False,
    max_wells: int = 0,
) -> pd.DataFrame:
    frames = []
    if synthetic or data_root is None:
        lat, tw = synthetic_well()
        from rwt.middle_out import split_by_mask

        sp = split_by_mask(lat, well_id="SYN01")
        frames.append(build_station_frame(lat, tw, sp))
        return pd.concat(frames, ignore_index=True)

    root = Path(data_root)
    # optional shared typewell
    tw_path = None
    for cand in root.rglob("*typewell*.csv"):
        tw_path = cand
        break
    typewell = load_typewell(tw_path) if tw_path else None

    n = 0
    for wid, df in iter_well_frames(root, "train"):
        if "is_predict" in df.columns:
            sp = split_by_mask(df, well_id=wid)
        else:
            sp = split_by_tvt_nan(df, well_id=wid)
        # per-well typewell if present
        tw = typewell
        tw_w = root / "train" / f"{wid}_typewell.csv"
        if tw_w.is_file():
            tw = load_typewell(tw_w)
        frames.append(build_station_frame(df, tw, sp))
        n += 1
        if max_wells and n >= max_wells:
            break
    if not frames:
        raise FileNotFoundError(f"No wells under {root}")
    return pd.concat(frames, ignore_index=True)


def train_lgbm_residual(
    stations: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    # train only on known stations with TVT
    train_df = stations[stations["is_known"] & stations["TVT"].notna()].copy()
    if len(train_df) < 50:
        train_df = stations[stations["TVT"].notna()].copy()
    X, cols = feature_matrix(train_df)
    y = train_df["TVT"].to_numpy(dtype=float)
    # residual vs flat prior
    prior = train_df["prior_flat_tvt"].to_numpy(dtype=float)
    y_res = y - prior
    groups = train_df["well_id"].astype(str).to_numpy()

    try:
        import lightgbm as lgb
    except ImportError as e:
        raise SystemExit("pip install lightgbm") from e

    n_groups = len(np.unique(groups))
    oof = np.zeros(len(train_df))
    models = []
    fold_scores = []
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": seed,
    }

    def _fit_fold(tr: np.ndarray, va: np.ndarray, fold: int):
        dtr = lgb.Dataset(X[tr], label=y_res[tr], feature_name=cols)
        dva = lgb.Dataset(X[va], label=y_res[va], feature_name=cols, reference=dtr)
        p = dict(params)
        p["seed"] = seed + fold
        model = lgb.train(
            p,
            dtr,
            num_boost_round=400,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        pred_res = model.predict(X[va])
        oof[va] = pred_res
        sc = rmse(y[va], prior[va] + pred_res)
        fold_scores.append(sc)
        models.append(model)

    if n_groups < 2:
        # synthetic / single-well smoke: last 25% stations as holdout
        n = len(train_df)
        cut = max(1, int(n * 0.75))
        tr = np.arange(0, cut)
        va = np.arange(cut, n)
        if len(va) == 0:
            va = tr[-max(1, n // 5) :]
            tr = tr[: -len(va)]
        _fit_fold(tr, va, 0)
    else:
        gkf = GroupKFold(n_splits=min(n_splits, n_groups))
        for fold, (tr, va) in enumerate(gkf.split(X, y_res, groups)):
            _fit_fold(tr, va, fold)

    oof_tvt = prior + oof
    from rwt.kernel_api import resolve_kernel
    from rwt.provenance import seal_artifact

    k = resolve_kernel()
    report = {
        "ok": True,
        "n_stations": int(len(train_df)),
        "n_wells": int(len(np.unique(groups))),
        "feature_cols": cols,
        "fold_rmse": fold_scores,
        "cv_rmse_mean": float(np.mean(fold_scores)),
        "cv_rmse_std": float(np.std(fold_scores)),
        "oof_rmse": rmse(y, oof_tvt),
        "kernel_id": k.kernel_id(),
        "kernel_private": k.is_private(),
        "law": (
            "GroupKFold by well; residual on flat-formation prior; "
            "open-core public kernel default; private kernel optional"
        ),
        "license": "AGPL-3.0-or-later",
    }
    report["provenance"] = seal_artifact(report, kind="cv_report")
    return {"report": report, "models": models, "feature_cols": cols}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train RWT LGBM residual TVT model")
    ap.add_argument("--data", default="", help="path to Kaggle data root")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic well only")
    ap.add_argument("--out", default="artifacts/run1")
    ap.add_argument("--max-wells", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stations = _assemble_dataset(
        Path(args.data) if args.data else None,
        synthetic=args.synthetic or not args.data,
        max_wells=args.max_wells,
    )
    stations.to_parquet(out / "stations.parquet") if False else stations.to_csv(
        out / "stations.csv", index=False
    )
    result = train_lgbm_residual(stations, n_splits=args.folds)
    report = result["report"]
    (out / "cv_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # save last model only for smoke; full ensemble later
    try:
        import joblib

        joblib.dump(
            {"models": result["models"], "feature_cols": result["feature_cols"]},
            out / "model.joblib",
        )
    except Exception:
        pass
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
