#!/usr/bin/env python3
"""
WP8 MEASURE — multi-well pad synthetic GR@TVT falsifier.

  python scripts/measure_manifold_gr_synth.py --out artifacts/manifold_gr_synth

Construct:
  - Shared typewell book G_tw(TVT)→GR
  - Continuous TVT field on pad XY
  - Train wells: full TVT labels; test wells: toe withheld (competition asymmetry)
  - Metrics: placement RMSE (last_known, neighbor IDW, field-oracle) +
    GR mismatch @ TVT_true vs @ predicted

RESTRICT: local MEASURE only; never Kaggle submit; no TVT_true in serve features.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    import pandas as pd

    from rwt.baselines import rmse
    from rwt.load import apply_curve_nulls_frame
    from rwt.pad_synth import (
        decode_gr_nulls,
        gr_mismatch_stats,
        last_known_series,
        make_pad_laterals,
        neighbor_idw_tvt,
    )

    ap = argparse.ArgumentParser(description="Pad synthetic GR@TVT MEASURE (WP8)")
    ap.add_argument("--n-wells", type=int, default=6)
    ap.add_argument("--n-stations", type=int, default=180)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gr-noise", type=float, default=3.0)
    ap.add_argument(
        "--out", default=str(ROOT / "artifacts" / "manifold_gr_synth")
    )
    a = ap.parse_args(argv)

    pad = make_pad_laterals(
        n_wells=int(a.n_wells),
        n_stations=int(a.n_stations),
        n_train=a.n_train,
        seed=int(a.seed),
        gr_noise=float(a.gr_noise),
    )
    tw = pad["typewell"]
    train_ids = set(pad["train_ids"])
    test_ids = list(pad["test_ids"])
    donors = [
        apply_curve_nulls_frame(w)
        for w in pad["wells"]
        if str(w["well_id"].iloc[0]) in train_ids
    ]

    rows: list[dict] = []
    for w in pad["wells"]:
        w = apply_curve_nulls_frame(w)
        wid = str(w["well_id"].iloc[0])
        role = str(w["pad_role"].iloc[0])
        pred = w["is_predict"].to_numpy(dtype=bool)
        y = w["TVT_true"].to_numpy(dtype=float)
        gr = decode_gr_nulls(w["GR"].to_numpy(dtype=float))
        heel = ~pred
        known_tvt = np.where(
            heel, w["TVT_input"].to_numpy(dtype=float), np.nan
        )
        # placement priors
        last = last_known_series(w)
        nb = neighbor_idw_tvt(w, donors, max_dist=900.0, k=4)
        # field-oracle: true field without residual walk — use true TVT as oracle
        # (upper bound: GR book consistency only)
        m = pred & np.isfinite(y)
        if m.sum() < 8:
            continue

        r_last = float(rmse(y[m], last[m])) if np.isfinite(last[m]).all() else float(
            "nan"
        )
        # neighbor: fall back to last where nb nan
        nb_f = np.where(np.isfinite(nb), nb, last)
        r_nb = float(rmse(y[m], nb_f[m]))
        r_oracle = 0.0  # perfect placement

        # GR corroboration
        mm_true = gr_mismatch_stats(
            gr, tw, y, known_tvt=known_tvt, mask=m & np.isfinite(gr)
        )
        mm_last = gr_mismatch_stats(
            gr, tw, last, known_tvt=known_tvt, mask=m & np.isfinite(gr)
        )
        mm_nb = gr_mismatch_stats(
            gr, tw, nb_f, known_tvt=known_tvt, mask=m & np.isfinite(gr)
        )

        rows.append(
            {
                "well_id": wid,
                "pad_role": role,
                "n_toe": int(m.sum()),
                "rmse_last_known": r_last,
                "rmse_neighbor_idw": r_nb,
                "rmse_oracle_true": r_oracle,
                "gr_mae_at_true_tvt": mm_true["mean_abs_mismatch"],
                "gr_mae_at_last_known": mm_last["mean_abs_mismatch"],
                "gr_mae_at_neighbor": mm_nb["mean_abs_mismatch"],
                "gr_n": mm_true["n"],
                "neighbor_beats_last": bool(
                    np.isfinite(r_nb) and np.isfinite(r_last) and r_nb < r_last - 1e-9
                ),
                "gr_true_better_than_last": bool(
                    np.isfinite(mm_true["mean_abs_mismatch"])
                    and np.isfinite(mm_last["mean_abs_mismatch"])
                    and mm_true["mean_abs_mismatch"]
                    < mm_last["mean_abs_mismatch"] - 1e-9
                ),
            }
        )

    R = pd.DataFrame(rows)
    test_R = R[R["pad_role"] == "test"] if len(R) else R
    train_R = R[R["pad_role"] == "train"] if len(R) else R

    def _mean(df: pd.DataFrame, col: str) -> float | None:
        if len(df) == 0 or col not in df.columns:
            return None
        v = df[col].to_numpy(dtype=float)
        return float(np.nanmean(v)) if np.isfinite(v).any() else None

    # Falsifier gates (construct law)
    gr_true_mae = _mean(R, "gr_mae_at_true_tvt")
    gr_last_mae = _mean(R, "gr_mae_at_last_known")
    test_nb = _mean(test_R, "rmse_neighbor_idw")
    test_last = _mean(test_R, "rmse_last_known")
    book_ok = bool(
        gr_true_mae is not None
        and gr_last_mae is not None
        and gr_true_mae < gr_last_mae
        and gr_true_mae < 8.0  # noise floor band for default gr_noise=3
    )
    pad_ok = bool(
        test_nb is not None
        and test_last is not None
        and test_nb < test_last
    )

    summary = {
        "ok": bool(len(R) > 0 and book_ok),
        "construct": "GR@TVT pad manifold WP8",
        "n_wells": int(len(pad["well_ids"])),
        "n_train": int(len(pad["train_ids"])),
        "n_test": int(len(pad["test_ids"])),
        "n_stations": int(pad["n_stations"]),
        "seed": int(pad["seed"]),
        "field_params": pad["field_params"],
        "train_ids": list(pad["train_ids"]),
        "test_ids": list(pad["test_ids"]),
        "mean_rmse_last_known_test": test_last,
        "mean_rmse_neighbor_idw_test": test_nb,
        "mean_rmse_last_known_train": _mean(train_R, "rmse_last_known"),
        "mean_rmse_neighbor_idw_train": _mean(train_R, "rmse_neighbor_idw"),
        "mean_gr_mae_at_true_tvt": gr_true_mae,
        "mean_gr_mae_at_last_known": gr_last_mae,
        "mean_gr_mae_at_neighbor": _mean(R, "gr_mae_at_neighbor"),
        "n_test_wells_neighbor_beats_last": int(
            test_R["neighbor_beats_last"].sum()
        )
        if len(test_R)
        else 0,
        "falsifier_book_ok": book_ok,
        "falsifier_pad_transfer_ok": pad_ok,
        "wave_a_12w_residual_floor": 10.741,
        "note": (
            "Placement priors only (no residual). GR MAE @ true TVT should beat "
            "MAE @ last_known (book identity). Neighbor IDW should beat last_known "
            "on test toes when pad field is smooth."
        ),
        "law": (
            "RESTRICT: local MEASURE; never Kaggle submit; TVT_true offline only; "
            "no formation tops; LAS nulls→NaN"
        ),
    }

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if len(R):
        R.to_csv(out / "per_well.csv", index=False)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    # shared typewell book artifact
    tw.to_csv(out / "typewell_book.csv", index=False)

    print(json.dumps(summary, indent=2), flush=True)
    print(
        f"WP8 book_ok={book_ok} pad_transfer_ok={pad_ok} wrote {out}",
        flush=True,
    )
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
