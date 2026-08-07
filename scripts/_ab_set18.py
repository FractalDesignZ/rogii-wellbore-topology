"""Set-18 deep param A/B — unbuffered (RESTRICT no submit)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lightgbm as lgb  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from rwt.baselines import rmse  # noqa: E402
from rwt.features import (  # noqa: E402
    compute_feature_medians,
    feature_matrix,
    impute_feature_frame,
)
from rwt.priors import DEFAULT_PRIOR_COL, MAX_TVT_DEV, resolve_prior_column  # noqa: E402


def oof_deep(stations: pd.DataFrame, patch: dict, n_seed: int = 3) -> float:
    train_df = stations[stations["is_predict"] & stations["TVT"].notna()].copy()
    prior_col = resolve_prior_column(train_df, DEFAULT_PRIOR_COL, strict=False)
    fin = np.isfinite(train_df["TVT"].to_numpy(float)) & np.isfinite(
        train_df[prior_col].to_numpy(float)
    )
    train_df = train_df.loc[fin].copy()
    X_raw, cols = feature_matrix(train_df, impute=False)
    y = train_df["TVT"].to_numpy(float)
    prior = train_df[prior_col].to_numpy(float)
    y_res = y - prior
    groups = train_df["well_id"].astype(str).to_numpy()
    rel = np.abs(train_df["relative_md"].to_numpy(float))
    rel = np.where(np.isfinite(rel), rel, 0.0)
    sample_w = 1.0 + np.clip(rel, 0.0, 6000.0) / 3000.0
    max_dev = float(MAX_TVT_DEV)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.02,
        "num_leaves": 127,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "lambda_l1": 0.05,
        "max_bin": 511,
        "num_threads": 4,
        "verbosity": -1,
        "seed": 42,
    }
    params.update(patch)
    oof = np.zeros(len(train_df))
    mask = np.zeros(len(train_df), dtype=bool)
    for fold, (tr, va) in enumerate(GroupKFold(3).split(X_raw, y_res, groups)):
        tr_f = pd.DataFrame(X_raw[tr], columns=cols)
        va_f = pd.DataFrame(X_raw[va], columns=cols)
        med = compute_feature_medians(tr_f)
        Xtr = impute_feature_frame(tr_f, medians=med).to_numpy(float)
        Xva = impute_feature_frame(va_f, medians=med).to_numpy(float)
        dtr = lgb.Dataset(Xtr, label=y_res[tr], weight=sample_w[tr], feature_name=cols)
        dva = lgb.Dataset(
            Xva, label=y_res[va], weight=sample_w[va], feature_name=cols, reference=dtr
        )
        bag = []
        for b in range(n_seed):
            p = dict(params)
            p["seed"] = 42 + fold * 17 + b * 101
            m = lgb.train(
                p,
                dtr,
                num_boost_round=2500,
                valid_sets=[dva],
                callbacks=[lgb.early_stopping(120, verbose=False)],
            )
            bag.append(np.asarray(m.predict(Xva), float))
        oof[va] = np.clip(np.mean(np.vstack(bag), 0), -max_dev, max_dev)
        mask[va] = True
    return rmse(y[mask], prior[mask] + oof[mask])


def main() -> None:
    st = pd.read_csv(ROOT / "artifacts" / "crown_12w_set16b" / "stations.csv")
    print(f"loaded {len(st)}", flush=True)
    variants = [
        ("baseline", {}, 3),
        ("lr_0.015", {"learning_rate": 0.015}, 3),
        ("leaves_95", {"num_leaves": 95}, 3),
        ("min_leaf_25", {"min_data_in_leaf": 25}, 3),
        ("l2_2.0", {"lambda_l2": 2.0}, 3),
        ("seeds_4", {}, 4),
        ("reg_mid", {"lambda_l2": 1.5, "min_data_in_leaf": 50}, 3),
    ]
    ranked = []
    for name, patch, seeds in variants:
        s = oof_deep(st, patch, seeds)
        ranked.append((s, name))
        print(f"{name:16s} oof={s:.4f}", flush=True)
    ranked.sort()
    print("--- best ---", flush=True)
    for s, n in ranked:
        print(f"  {s:.4f} {n}", flush=True)


if __name__ == "__main__":
    main()
