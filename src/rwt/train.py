"""GroupKFold-by-well training (LightGBM residual on heel last-known prior)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from rwt.baselines import rmse
from rwt.features import build_station_frame, feature_matrix
from rwt.load import iter_well_frames, load_typewell, synthetic_well, typewell_path_for_well
from rwt.middle_out import MiddleOutSplit, resolve_middle_out, split_by_mask
from rwt.priors import DEFAULT_PRIOR_COL, resolve_prior_column
from rwt.spatial import SpatialIndex, build_spatial_index


def _typewell_for_well(root: Path, well_id: str, split: str = "train") -> Path | None:
    """Back-compat alias — prefer ``typewell_path_for_well``."""
    return typewell_path_for_well(root, well_id, split)


def _assemble_dataset(
    data_root: Path | None,
    *,
    synthetic: bool = False,
    max_wells: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assemble train stations + Gate 2 summary (passed/rejected well ids)."""
    frames: list[pd.DataFrame] = []
    gate2_meta: dict[str, Any] = {
        "name": "MIDDLE_OUT_MASK",
        "passed_wells": [],
        "rejected_wells": [],
        "n_passed": 0,
        "n_rejected": 0,
    }
    if synthetic or data_root is None:
        lat, tw = synthetic_well()
        sp = split_by_mask(lat, well_id="SYN01")
        frames.append(build_station_frame(lat, tw, sp))
        gate2_meta["passed_wells"] = ["SYN01"]
        gate2_meta["n_passed"] = 1
        gate2_meta["passed"] = True
        return pd.concat(frames, ignore_index=True), gate2_meta

    root = Path(data_root)

    loaded: list[tuple[str, pd.DataFrame, MiddleOutSplit, pd.DataFrame | None]] = []
    for wid, df in iter_well_frames(root, "train"):
        sp = resolve_middle_out(df, well_id=wid)
        if not sp.n_predict:
            continue
        if "TVT" not in df.columns and "TVT_true" not in df.columns:
            continue
        tw_path = typewell_path_for_well(root, wid, "train")
        tw = load_typewell(tw_path) if tw_path else None
        loaded.append((wid, df, sp, tw))
        if max_wells and len(loaded) >= max_wells:
            break
    if not loaded:
        raise FileNotFoundError(f"No wells under {root}")
    spatial: SpatialIndex = build_spatial_index([(w, d) for w, d, _s, _t in loaded])

    from rwt.gates import gate_middle_out_mask

    n = 0
    for wid, df, sp, tw in loaded:
        g2 = gate_middle_out_mask(df, wid)
        if not g2.passed:
            print(f"[GATE2 SKIP] {wid}: {g2.detail}", flush=True)
            gate2_meta["rejected_wells"].append(
                {"well_id": wid, "detail": g2.detail}
            )
            continue
        tvt_col = "TVT" if "TVT" in df.columns else "TVT_true"
        tvt_arr = df[tvt_col].to_numpy()
        pred_idx = np.asarray(sp.predict_idx, dtype=int)
        labeled_pred = pred_idx[pd.notna(tvt_arr[pred_idx])]
        if len(labeled_pred) == 0:
            gate2_meta["rejected_wells"].append(
                {"well_id": wid, "detail": {"reason": "no_labeled_predict"}}
            )
            continue
        frames.append(
            build_station_frame(
                df,
                tw,
                sp,
                station_idx=labeled_pred,
                spatial_index=spatial,
            )
        )
        gate2_meta["passed_wells"].append(wid)
        n += 1
        if n % 10 == 0:
            print(f"assembled {n} wells…", flush=True)
    if not frames:
        raise FileNotFoundError(f"No labeled predict stations under {root}")
    gate2_meta["n_passed"] = len(gate2_meta["passed_wells"])
    gate2_meta["n_rejected"] = len(gate2_meta["rejected_wells"])
    gate2_meta["passed"] = gate2_meta["n_passed"] > 0
    return pd.concat(frames, ignore_index=True), gate2_meta


def train_lgbm_residual(
    stations: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
    prior_col: str | None = None,
    gate2_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Residual learner on labeled *toe* (is_predict).

    Target = TVT − prior. Default residual base is DEFAULT_PRIOR_COL (pf_blend).
    Explicit prior_col is strict (must exist); omitted uses DEFAULT with fallbacks.
    """
    if "is_predict" in stations.columns:
        train_df = stations[stations["is_predict"] & stations["TVT"].notna()].copy()
    else:
        train_df = stations[stations["TVT"].notna()].copy()
    if prior_col is None:
        prior_col = resolve_prior_column(train_df, DEFAULT_PRIOR_COL, strict=False)
    else:
        prior_col = resolve_prior_column(train_df, prior_col, strict=True)
    # require finite TVT and selected prior (PF returns NaN when no TVT ref)
    fin = np.isfinite(train_df["TVT"].to_numpy(dtype=float)) & np.isfinite(
        train_df[prior_col].to_numpy(dtype=float)
    )
    train_df = train_df.loc[fin].copy()
    if len(train_df) < 50:
        raise ValueError(
            f"Insufficient labeled prediction-zone rows for residual train "
            f"(n={len(train_df)}). Need finite TVT and {prior_col} values."
        )
    X, cols = feature_matrix(train_df)
    y = train_df["TVT"].to_numpy(dtype=float)
    prior = train_df[prior_col].to_numpy(dtype=float)
    y_res = y - prior
    groups = train_df["well_id"].astype(str).to_numpy()

    try:
        import lightgbm as lgb
    except ImportError as e:
        raise SystemExit("pip install lightgbm") from e

    n_groups = len(np.unique(groups))
    oof = np.zeros(len(train_df))
    models = []
    fold_scores: list[float] = []
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

    def _fit_fold(tr: np.ndarray, va: np.ndarray, fold: int) -> None:
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
        fold_scores.append(rmse(y[va], prior[va] + pred_res))
        models.append(model)

    oof_mask = np.zeros(len(train_df), dtype=bool)
    if n_groups < 2:
        n = len(train_df)
        cut = max(1, int(n * 0.75))
        tr = np.arange(0, cut)
        va = np.arange(cut, n)
        if len(va) == 0:
            va = tr[-max(1, n // 5) :]
            tr = tr[: -len(va)]
        _fit_fold(tr, va, 0)
        oof_mask[va] = True
    else:
        gkf = GroupKFold(n_splits=min(n_splits, n_groups))
        for fold, (tr, va) in enumerate(gkf.split(X, y_res, groups)):
            _fit_fold(tr, va, fold)
            oof_mask[va] = True

    oof_tvt = prior + oof
    from rwt.kernel_api import resolve_kernel
    from rwt.provenance import seal_artifact

    k = resolve_kernel()
    # Baselines and OOF share the same finite OOF-row set so beats_* is valid
    row_m = oof_mask if oof_mask.any() else np.ones(len(train_df), dtype=bool)
    y_m = y[row_m]
    oof_m = oof_tvt[row_m]
    shared = np.isfinite(y_m) & np.isfinite(oof_m)
    y_s = y_m[shared]
    oof_s = oof_m[shared]
    oof_r = rmse(y_s, oof_s)
    baseline_rmse: dict[str, float] = {}
    for name, col in (
        ("last_known", "prior_last_tvt"),
        ("flat_formation", "prior_flat_tvt"),
        ("linear_md_extrap", "prior_extrap_tvt"),
        ("dual_transport_mid", "prior_mid"),
        ("heel_dip_stack", "prior_dip_stack"),
        ("particle_filter", "prior_pf_tvt"),
        ("pf_blend_last_known", "prior_pf_blend"),
        ("pf_bag", "prior_pf_bag"),
        ("prior_stack", "prior_stack"),
    ):
        if col not in train_df.columns:
            continue
        b = train_df[col].to_numpy(dtype=float)[row_m]
        if not np.isfinite(b[shared]).all():
            continue  # skip baselines with holes on the OOF row set
        baseline_rmse[name] = rmse(y_s, b[shared])
    beats = {
        name: bool(oof_r < br) for name, br in baseline_rmse.items() if np.isfinite(br)
    }

    # ── Gates 1+3 (Gate 2 is structural at assemble time) ───────────────
    from rwt.gates import gate_ip_fence, gate_residual_acceptance
    from rwt.simd import cpu_feature_report

    prior_m = prior[row_m][shared]
    lk_m = None
    if "prior_last_tvt" in train_df.columns:
        lk_full = train_df["prior_last_tvt"].to_numpy(dtype=float)[row_m]
        if np.isfinite(lk_full[shared]).all():
            lk_m = lk_full[shared]
    gate3 = gate_residual_acceptance(
        y_s,
        oof_s,
        prior_m,
        prior_name=prior_col,
        last_known_pred=lk_m,
    )
    gate1 = gate_ip_fence()
    use_residual = bool(gate3.passed)
    # On reject: choose pure prior with best OOF among prior_col and last_known
    serve_prior_col = prior_col
    if use_residual:
        print(
            f"[GATE PASS] Residual head accepted: OOF {gate3.detail['oof_rmse_residual']:.3f} "
            f"<= prior({prior_col}) {gate3.detail['oof_rmse_prior']:.3f}",
            flush=True,
        )
        final_models = models
        final_prior_type = f"{prior_col}_plus_lgbm_residual"
        final_oof = oof_r
    else:
        cand = {prior_col: float(gate3.detail["oof_rmse_prior"])}
        # exact 0.0 is finite and valid — never use truthiness on the score
        lk_rmse = gate3.detail.get("oof_rmse_last_known")
        if lk_m is not None and lk_rmse is not None and np.isfinite(lk_rmse):
            cand["prior_last_tvt"] = float(lk_rmse)
        serve_prior_col = min(cand, key=cand.get)
        final_oof = cand[serve_prior_col]
        final_models = []
        final_prior_type = f"pure_{serve_prior_col}"
        print(
            f"[GATE REJECT] Residual overfitted "
            f"(OOF {gate3.detail['oof_rmse_residual']:.3f}). "
            f"Falling back to pure prior '{serve_prior_col}' (OOF {final_oof:.3f}).",
            flush=True,
        )

    report = {
        "ok": True,
        "n_stations": int(len(train_df)),
        "n_wells": int(len(np.unique(groups))),
        "zone": "predict_toe_labeled",
        "feature_cols": cols,
        "fold_rmse": fold_scores,
        "cv_rmse_mean": float(np.mean(fold_scores)),
        "cv_rmse_std": float(np.std(fold_scores)),
        "oof_rmse": oof_r,
        "oof_rmse_selected": final_oof,
        "baseline_rmse": baseline_rmse,
        "beats_baselines": beats,
        "kernel_id": k.kernel_id(),
        "kernel_private": k.is_private(),
        "prior_column": serve_prior_col if not use_residual else prior_col,
        "train_prior_column": prior_col,
        "final_prior_type": final_prior_type,
        "use_residual": use_residual,
        "gates": [
            gate1.to_dict(),
            gate2_meta
            if gate2_meta is not None
            else {
                "name": "MIDDLE_OUT_MASK",
                "passed": True,
                "note": "not provided (caller assembled stations externally)",
            },
            gate3.to_dict(),
        ],
        "cpu": cpu_feature_report(),
        "law": (
            "GroupKFold by well; residual gated by prior OOF; "
            "particle-filter typewell track; SO(3)/SE(3)/spatial features; "
            "open-core public kernel fail-closed"
        ),
        "license": "AGPL-3.0-or-later",
    }
    report["provenance"] = seal_artifact(report, kind="cv_report")
    return {
        "report": report,
        "models": final_models,
        "feature_cols": cols,
        "prior_column": report["prior_column"],
        "use_residual": use_residual,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train RWT LGBM residual TVT model")
    ap.add_argument("--data", default="", help="path to Kaggle data root")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic well only")
    ap.add_argument("--out", default="artifacts/run1")
    ap.add_argument("--max-wells", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument(
        "--prior-col",
        default=None,
        help=f"residual base prior column (default: {DEFAULT_PRIOR_COL})",
    )
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stations, gate2_meta = _assemble_dataset(
        Path(args.data) if args.data else None,
        synthetic=args.synthetic or not args.data,
        max_wells=args.max_wells,
    )
    stations.to_csv(out / "stations.csv", index=False)
    result = train_lgbm_residual(
        stations,
        n_splits=args.folds,
        prior_col=args.prior_col,
        gate2_meta=gate2_meta,
    )
    report = result["report"]
    (out / "cv_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    try:
        import joblib

        joblib.dump(
            {
                "models": result["models"],
                "feature_cols": result["feature_cols"],
                "prior_column": result["prior_column"],
                "use_residual": result.get("use_residual", True),
            },
            out / "model.joblib",
        )
    except Exception as e:
        print(f"FAIL model.joblib: {e}", flush=True)
        report["ok"] = False
        report["model_error"] = str(e)[:400]
        (out / "cv_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
