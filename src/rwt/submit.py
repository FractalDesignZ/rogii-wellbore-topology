"""Build Kaggle submission CSV: ``id,tvt`` with ``{WELLNAME}_{row_index}``."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rwt.features import build_station_frame
from rwt.load import iter_well_frames, load_typewell, typewell_path_for_well
from rwt.middle_out import resolve_middle_out
from rwt.priors import resolve_prior_column
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
    cols = model_blob["feature_cols"]
    X = (
        stations.reindex(columns=cols)
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    Xn = X.to_numpy()
    # Must match residual base used at train time (stored on model)
    preferred = model_blob.get("prior_column") or "prior_last_tvt"
    # strict: serve with the same prior column as training (no silent fallback)
    try:
        prior_col = resolve_prior_column(stations, preferred, strict=True)
    except KeyError:
        prior_col = resolve_prior_column(stations, preferred, strict=False)
    prior = stations[prior_col].to_numpy(dtype=float)
    models = model_blob.get("models") or []
    use_residual = bool(model_blob.get("use_residual", True)) and len(models) > 0
    if not use_residual:
        return prior
    preds = [m.predict(Xn) for m in models]
    res = np.mean(np.vstack(preds), axis=0)
    return prior + res


def assemble_split_stations(
    data_root: Path,
    split: str = "test",
    *,
    max_wells: int = 0,
    predict_only: bool = True,
    max_neighbor_wells: int = 0,
) -> pd.DataFrame:
    """Build station frames for train/test wells (toe by default).

    Spatial index uses the same train laterals as training (default: all train
    wells) so neighbor transfer matches the train-time graph contract.
    """
    root = Path(data_root)
    train_wells = []
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
