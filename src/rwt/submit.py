"""Build Kaggle submission CSV (schema may need tweak after data inspect)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def predict_stations(model_blob: dict, stations: pd.DataFrame) -> np.ndarray:
    cols = model_blob["feature_cols"]
    X = stations.reindex(columns=cols).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    Xn = X.to_numpy()
    prior = stations["prior_flat_tvt"].to_numpy(dtype=float)
    preds = []
    for m in model_blob["models"]:
        preds.append(m.predict(Xn))
    res = np.mean(np.vstack(preds), axis=0)
    return prior + res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stations", required=True, help="stations CSV with features for test wells")
    ap.add_argument("--out", default="submissions/submission.csv")
    ap.add_argument("--id-col", default="Id")
    args = ap.parse_args(argv)

    import joblib

    blob = joblib.load(args.model)
    st = pd.read_csv(args.stations)
    pred = predict_stations(blob, st)
    # only prediction zone if column present
    if "is_predict" in st.columns:
        mask = st["is_predict"].astype(bool)
        st = st.loc[mask].copy()
        pred = pred[mask.to_numpy()]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # generic submission: require id column or synthesize
    if args.id_col in st.columns:
        sub = pd.DataFrame({args.id_col: st[args.id_col], "TVT": pred})
    else:
        sub = pd.DataFrame(
            {
                "well_id": st["well_id"],
                "MD": st["MD"],
                "TVT": pred,
            }
        )
    sub.to_csv(out, index=False)
    print(f"wrote {out} rows={len(sub)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
