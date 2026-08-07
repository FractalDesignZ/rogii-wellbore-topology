"""Build Kaggle submission CSV: ``id,tvt`` with ``{WELLNAME}_{row_index}``."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rwt.features import build_station_frame
from rwt.load import iter_well_frames, load_typewell, typewell_path_for_well
from rwt.middle_out import resolve_middle_out
from rwt.priors import MAX_TVT_DEV, PRIOR_FALLBACKS, resolve_prior_column
from rwt.spatial import build_spatial_index


def submission_id(well_id: str, station: int) -> str:
    """ROGII sample_submission format: ``{WELLNAME}_{row_index}``."""
    return f"{well_id}_{int(station)}"


def format_submission(
    stations: pd.DataFrame,
    preds: np.ndarray,
    *,
    predict_only: bool = True,
) -> pd.DataFrame:
    st = stations.reset_index(drop=True)
    y = np.asarray(preds, dtype=float)
    if predict_only and "is_predict" in st.columns:
        mask = st["is_predict"].astype(bool).to_numpy()
        if not mask.any():
            raise ValueError("predict_only=True but no is_predict rows")
        if not bool(mask.all()):
            st = st.loc[mask].reset_index(drop=True)
            y = y[mask]
    if len(st) != len(y):
        raise ValueError(f"stations ({len(st)}) vs preds ({len(y)}) length mismatch")
    ids = [
        submission_id(str(w), int(s))
        for w, s in zip(st["well_id"].tolist(), st["station"].tolist())
    ]
    # stable order for Kaggle: sort by id string (matches sample_submission sort)
    sub = pd.DataFrame({"id": ids, "tvt": y}).sort_values("id").reset_index(drop=True)
    return sub


def predict_stations(model_blob: dict, stations: pd.DataFrame) -> np.ndarray:
    from rwt.features import impute_feature_frame

    cols = model_blob["feature_cols"]
    # Train/serve parity: use *stored* train medians (never batch-fit on serve).
    # Legacy blobs without feature_medians get deterministic zero-fill so results
    # do not depend on serving-batch composition.
    stored_med = model_blob.get("feature_medians")
    if isinstance(stored_med, dict) and stored_med:
        medians = {str(k): float(v) for k, v in stored_med.items()}
    else:
        medians = {str(c): 0.0 for c in cols}
    missing = [c for c in cols if c not in stations.columns]
    if missing:
        raise ValueError(
            f"serving frame missing {len(missing)} trained feature columns: "
            f"{missing[:10]} — refuse silent median fill; rebuild station frame"
        )
    Xn = impute_feature_frame(
        stations.reindex(columns=cols), medians=medians
    ).to_numpy(dtype=float)
    # Train-serve contract: stored prior_column is strict (no silent swap).
    # Legacy blobs without the key fall back through PRIOR_FALLBACKS.
    stored_prior = model_blob.get("prior_column")
    if stored_prior:
        prior_col = resolve_prior_column(stations, str(stored_prior), strict=True)
    else:
        # Legacy blob: walk PRIOR_FALLBACKS (stack before best); not last_tvt-first
        prior_col = resolve_prior_column(
            stations, PRIOR_FALLBACKS[0], strict=False
        )
    prior = stations[prior_col].to_numpy(dtype=float)
    if not np.isfinite(prior).all():
        n_bad = int((~np.isfinite(prior)).sum())
        raise ValueError(
            f"non-finite serving prior '{prior_col}' on {n_bad}/{len(prior)} rows; "
            "refuse silent fill — fix assemble/prior before submit"
        )
    max_dev = float(model_blob.get("residual_max_dev", MAX_TVT_DEV))
    smooth_w = int(model_blob.get("residual_smooth_window") or 0)
    wids = (
        stations["well_id"].to_numpy()
        if "well_id" in stations.columns
        else None
    )

    def _head_tvt(prior_arr: np.ndarray, models: list) -> np.ndarray:
        if not models:
            return prior_arr
        preds = np.vstack([np.asarray(m.predict(Xn), dtype=float) for m in models])
        # Match train OOF: mean seed bag (median+std-shrink rejected by measure)
        res = np.mean(preds, axis=0)
        # Clamp residual Δ vs residual *base* only. Do not re-clamp to last_known.
        res = np.clip(res, -max_dev, max_dev)
        if smooth_w > 1 and wids is not None:
            # Skip when well_id missing — avoid cross-well rolling blend
            from rwt.baselines import smooth_residual_along_well

            order_key = None
            if "MD" in stations.columns:
                order_key = stations["MD"].to_numpy(dtype=float)
            elif "station" in stations.columns:
                order_key = stations["station"].to_numpy(dtype=float)
            res = smooth_residual_along_well(
                res, wids, window=smooth_w, order_key=order_key
            )
            res = np.clip(res, -max_dev, max_dev)
        return prior_arr + res

    # Dual residual: mean over residual_heads (set-15); legacy uses models alone
    from rwt.gates import adaptive_residual_mask, apply_adaptive_residual

    heads = model_blob.get("residual_heads") or []
    use_residual = bool(model_blob.get("use_residual", True))
    out: np.ndarray | None = None
    if use_residual and heads:
        outs: list[np.ndarray] = []
        out_bases: list[str] = []
        # Weights are aligned to residual_heads order at train time (by base name)
        w_raw = model_blob.get("dual_residual_weights")
        name_to_w: dict[str, float] = {}
        if isinstance(w_raw, (list, tuple)):
            for h, wi in zip(heads, w_raw, strict=False):
                if isinstance(h, dict):
                    name_to_w[str(h.get("prior_column") or "")] = float(wi)
        for h in heads:
            if not isinstance(h, dict):
                continue
            h_models = h.get("models") or []
            if not h_models:
                continue
            h_prior_name = str(h.get("prior_column") or stored_prior or "prior_best")
            h_col = resolve_prior_column(stations, h_prior_name, strict=True)
            h_prior = stations[h_col].to_numpy(dtype=float)
            if not np.isfinite(h_prior).all():
                n_bad = int((~np.isfinite(h_prior)).sum())
                raise ValueError(
                    f"non-finite serving prior '{h_col}' on {n_bad}/{len(h_prior)} rows"
                )
            outs.append(_head_tvt(h_prior, list(h_models)))
            out_bases.append(h_prior_name)
        if outs:
            stacked = np.vstack(outs)
            ww = np.asarray(
                [name_to_w.get(b, float("nan")) for b in out_bases], dtype=float
            )
            if np.isfinite(ww).all() and float(np.sum(ww)) > 0:
                ww = ww / ww.sum()
                out = np.average(stacked, axis=0, weights=ww)
            else:
                out = np.mean(stacked, axis=0)
    if out is None:
        models = model_blob.get("models") or []
        if not use_residual or not models:
            return prior
        out = _head_tvt(prior, list(models))

    # Adaptive serve: shared mask contract with train adaptive_oof (station-conf
    # by default; optional well_fire map if serialized on the blob).
    adaptive = bool(model_blob.get("adaptive_serve", False))
    if adaptive and use_residual:
        conf_thresh = float(model_blob.get("adaptive_conf_threshold", 0.35))
        conf = (
            stations["pf_confidence"].to_numpy(dtype=float)
            if "pf_confidence" in stations.columns
            else None
        )
        wf_raw = model_blob.get("well_fire")
        well_fire = wf_raw if isinstance(wf_raw, dict) and wf_raw else None
        fire = adaptive_residual_mask(
            len(out),
            pf_confidence=conf if well_fire is None else None,
            conf_threshold=conf_thresh,
            well_ids=wids if well_fire is not None else None,
            well_fire=well_fire,
        )
        # Fallback pure prior is the stored residual base (primary prior_column)
        out = apply_adaptive_residual(out, prior, fire)
    if not np.isfinite(out).all():
        n_bad = int((~np.isfinite(out)).sum())
        raise ValueError(
            f"non-finite residual predictions on {n_bad}/{len(out)} rows"
        )
    return out


def assemble_split_stations(
    root: Path,
    split: str,
    *,
    max_wells: int = 0,
    max_neighbor_wells: int = 0,
    predict_only: bool = True,
) -> pd.DataFrame:
    """Build station frames for a split (train or test)."""
    train_wells: list[tuple[str, pd.DataFrame]] = []
    for i, (wid, df) in enumerate(iter_well_frames(root, "train")):
        train_wells.append((wid, df))
        if max_neighbor_wells and i + 1 >= max_neighbor_wells:
            break
    spatial = build_spatial_index(train_wells) if train_wells else None

    frames: list[pd.DataFrame] = []
    n = 0
    for wid, df in iter_well_frames(root, split):
        sp = resolve_middle_out(df, well_id=wid)
        tw_path = typewell_path_for_well(root, wid, split)
        tw = load_typewell(tw_path) if tw_path else None
        if predict_only:
            if not sp.n_predict:
                continue
            idx = sp.predict_idx
        else:
            idx = None
        frames.append(
            build_station_frame(
                df, tw, sp, station_idx=idx, spatial_index=spatial
            )
        )
        n += 1
        if max_wells and n >= max_wells:
            break
    if not frames:
        raise FileNotFoundError(f"No {split} wells under {root}")
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Predict TVT and write Kaggle submission.csv")
    ap.add_argument("--model", required=True, help="joblib model from rwt.train")
    ap.add_argument(
        "--stations",
        default="",
        help="optional prebuilt stations CSV; else --data + --split",
    )
    ap.add_argument("--data", default="data/raw", help="Kaggle data root")
    ap.add_argument("--split", default="test", choices=("train", "test"))
    ap.add_argument("--out", default="submissions/submission.csv")
    ap.add_argument("--max-wells", type=int, default=0)
    ap.add_argument(
        "--all-stations",
        action="store_true",
        help="emit all stations (default: prediction zone only)",
    )
    args = ap.parse_args(argv)

    import joblib

    blob = joblib.load(args.model)
    if args.stations:
        st = pd.read_csv(args.stations)
    else:
        st = assemble_split_stations(
            Path(args.data),
            args.split,
            max_wells=args.max_wells,
            predict_only=not args.all_stations,
        )
    pred = predict_stations(blob, st)
    sub = format_submission(st, pred, predict_only=not args.all_stations)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"wrote {out} rows={len(sub)} cols={list(sub.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
