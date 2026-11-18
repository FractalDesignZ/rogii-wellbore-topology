"""
Multi-well pad synthetic for GR@TVT construct (WP8).

Shared typewell book G_tw(TVT)→GR; continuous TVT field on pad XY;
laterals read GR = G_tw(TVT_true)+noise. Train wells expose TVT; test wells
mirror competition (toe TVT withheld; offline TVT_true only).
"""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd


def make_typewell_book(
    *,
    seed: int = 0,
    n_tw: int = 400,
    tvt_lo: float = 40.0,
    tvt_hi: float = 220.0,
) -> pd.DataFrame:
    """Shared formation book: dense (TVT, GR) typewell curve."""
    rng = np.random.default_rng(int(seed) + 11)
    tw_tvt = np.linspace(tvt_lo, tvt_hi, int(n_tw))
    # multi-scale formation signature (not pure sine)
    tw_gr = (
        55.0
        + 28.0 * np.sin(tw_tvt / 12.0)
        + 12.0 * np.sin(tw_tvt / 5.5)
        + 6.0 * np.cos(tw_tvt / 28.0)
        + rng.normal(0.0, 1.2, len(tw_tvt))
    )
    return pd.DataFrame({"TVT": tw_tvt, "GR": tw_gr})


def tvt_field_xy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    t0: float = 100.0,
    gx: float = 0.004,
    gy: float = -0.006,
    wave_amp: float = 2.5,
    wave_lx: float = 800.0,
    wave_ly: float = 400.0,
) -> np.ndarray:
    """Smooth pad TVT field (planar dip + mild corrugation)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    base = t0 + gx * x + gy * y
    wave = wave_amp * np.sin(2.0 * np.pi * x / wave_lx) * np.cos(
        2.0 * np.pi * y / wave_ly
    )
    return base + wave


def make_pad_laterals(
    *,
    n_wells: int = 6,
    n_stations: int = 180,
    ps_frac: float = 0.65,
    n_train: int | None = None,
    seed: int = 0,
    gr_noise: float = 3.0,
    path_noise: float = 1.5,
) -> dict[str, Any]:
    """
    Build multi-well pad laterals + shared typewell.

    Returns dict:
      typewell, wells (list of DataFrames), train_ids, test_ids, field_params
    """
    rng = np.random.default_rng(int(seed))
    tw = make_typewell_book(seed=seed)
    tw_tvt = tw["TVT"].to_numpy(dtype=float)
    tw_gr = tw["GR"].to_numpy(dtype=float)

    n_wells = int(max(2, n_wells))
    n_train = int(n_train) if n_train is not None else max(1, n_wells // 2)
    n_train = min(n_train, n_wells - 1)  # at least one test well
    n_stations = int(max(40, n_stations))
    ps = int(n_stations * float(ps_frac))
    ps = max(8, min(ps, n_stations - 8))

    # field params (shared geology)
    t0 = 100.0 + float(rng.normal(0, 2))
    gx = 0.0035 + float(rng.normal(0, 0.0004))
    gy = -0.0055 + float(rng.normal(0, 0.0004))

    wells: list[pd.DataFrame] = []
    well_ids: list[str] = []
    for i in range(n_wells):
        wid = f"PAD{i:02d}"
        well_ids.append(wid)
        # parallel laterals offset in Y; X progresses along MD
        y0 = 200.0 * i + float(rng.normal(0, 15))
        md = np.arange(n_stations, dtype=float)
        x = np.linspace(0.0, 3500.0, n_stations) + rng.normal(0, path_noise, n_stations)
        y = y0 + 0.02 * md + rng.normal(0, path_noise, n_stations)
        z = np.concatenate(
            [
                np.linspace(0.0, -750.0, n_stations // 3),
                np.linspace(-750.0, -780.0, n_stations - n_stations // 3),
            ]
        )
        # true TVT from shared field + mild well-specific walk
        tvt = tvt_field_xy(x, y, t0=t0, gx=gx, gy=gy) + np.cumsum(
            rng.normal(0.0, 0.03, n_stations)
        )
        # clip into typewell span with margin
        tvt = np.clip(tvt, float(tw_tvt[2]), float(tw_tvt[-3]))
        gr = np.interp(tvt, tw_tvt, tw_gr) + rng.normal(0.0, gr_noise, n_stations)
        # optional LAS nulls on ~2% stations
        null_m = rng.random(n_stations) < 0.02
        gr = gr.copy()
        gr[null_m] = -999.25

        is_pred = np.zeros(n_stations, dtype=bool)
        is_pred[ps:] = True
        role = "train" if i < n_train else "test"
        # train: full TVT labels (competition train); test: toe withheld
        if role == "train":
            tvt_col = tvt.copy()
            tvt_in = np.where(is_pred, np.nan, tvt)
        else:
            tvt_col = np.where(is_pred, np.nan, tvt)
            tvt_in = np.where(is_pred, np.nan, tvt)

        wells.append(
            pd.DataFrame(
                {
                    "well_id": wid,
                    "MD": md,
                    "X": x,
                    "Y": y,
                    "Z": z,
                    "GR": gr,
                    "TVT": tvt_col,
                    "TVT_input": tvt_in,
                    "is_predict": is_pred,
                    "TVT_true": tvt,  # offline eval only
                    "pad_role": role,
                }
            )
        )

    return {
        "typewell": tw,
        "wells": wells,
        "well_ids": well_ids,
        "train_ids": well_ids[:n_train],
        "test_ids": well_ids[n_train:],
        "field_params": {"t0": t0, "gx": gx, "gy": gy, "ps": ps},
        "n_stations": n_stations,
        "seed": int(seed),
    }


def decode_gr_nulls(gr: np.ndarray) -> np.ndarray:
    """LAS sentinels → NaN (match load law)."""
    from rwt.load import GR_ABSURD_LE, normalize_curve_nulls

    return normalize_curve_nulls(gr, absurd_le=GR_ABSURD_LE)


def last_known_series(df: pd.DataFrame) -> np.ndarray:
    """Forward-fill TVT_input / known TVT along MD order."""
    n = len(df)
    out = np.full(n, np.nan, dtype=float)
    fill = np.nan
    if "TVT_input" in df.columns:
        ti = df["TVT_input"].to_numpy(dtype=float)
    else:
        ti = np.full(n, np.nan)
    tv = df["TVT"].to_numpy(dtype=float) if "TVT" in df.columns else np.full(n, np.nan)
    for i in range(n):
        if np.isfinite(ti[i]):
            fill = float(ti[i])
        elif np.isfinite(tv[i]) and not bool(df["is_predict"].iloc[i]):
            fill = float(tv[i])
        out[i] = fill
    return out


def neighbor_idw_tvt(
    target: pd.DataFrame,
    donors: list[pd.DataFrame],
    *,
    max_dist: float = 800.0,
    k: int = 4,
    label_policy: Literal["observed_only", "full_tvt"] = "observed_only",
) -> np.ndarray:
    """
    Pad placement prior: IDW of explicitly visible donor labels at nearest XY.

    ``observed_only`` reads TVT_input only, without a fallback to full labels.
    ``full_tvt`` reads TVT only and requires every non-self donor to be marked
    pad_role='train' throughout. Callers must restrict those donors to the
    training partition; this function does not construct validation folds.

    Each nonempty frame must have one unambiguous well_id. Self donors are
    excluded by that identity. Predictor inputs never include TVT_true. Missing
    target/donor coordinates or labels remain unsupported rather than filled.
    """
    if label_policy not in ("observed_only", "full_tvt"):
        raise ValueError(f"unknown donor label policy: {label_policy!r}")
    if not np.isfinite(max_dist) or max_dist < 0:
        raise ValueError("max_dist must be finite and nonnegative")
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    n = len(target)
    est = np.full(n, np.nan, dtype=float)
    if not n:
        return est

    def frame_identity(frame: pd.DataFrame) -> str:
        if "well_id" not in frame or frame["well_id"].isna().any():
            raise ValueError("each nonempty frame needs one non-null well_id")
        ids = frame["well_id"].astype(str).str.strip().unique()
        if len(ids) != 1 or not ids[0]:
            raise ValueError("each nonempty frame needs one unambiguous well_id")
        return str(ids[0])

    target_id = frame_identity(target)
    tx = target["X"].to_numpy(dtype=float)
    ty = target["Y"].to_numpy(dtype=float)
    target_rows = np.flatnonzero(np.isfinite(tx) & np.isfinite(ty))
    source = "TVT" if label_policy == "full_tvt" else "TVT_input"
    from rwt.load import normalize_curve_nulls

    donor_pts: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    seen_ids: set[str] = set()
    for d in donors:
        if not len(d):
            continue
        donor_id = frame_identity(d)
        if donor_id == target_id:
            continue
        if donor_id in seen_ids:
            raise ValueError(f"duplicate donor well_id: {donor_id!r}")
        seen_ids.add(donor_id)
        if label_policy == "full_tvt" and (
            "pad_role" not in d or not d["pad_role"].eq("train").all()
        ):
            raise ValueError("full_tvt requires explicitly declared training donors")
        tv = normalize_curve_nulls(d[source]) if source in d else np.full(len(d), np.nan)
        dx = d["X"].to_numpy(dtype=float)
        dy = d["Y"].to_numpy(dtype=float)
        m = np.isfinite(tv) & np.isfinite(dx) & np.isfinite(dy)
        if m.sum() < 4:
            continue
        donor_pts.append((donor_id, dx[m], dy[m], tv[m]))
    if not donor_pts:
        return est

    for i in target_rows:
        dists: list[tuple[float, str, float]] = []
        for donor_id, dx, dy, dt in donor_pts:
            distance = np.hypot(dx - tx[i], dy - ty[i])
            j = int(np.argmin(distance))
            dist = float(distance[j])
            if dist <= max_dist:
                dists.append((dist, donor_id, float(dt[j])))
        if not dists:
            continue
        dists.sort(key=lambda z: (z[0], z[1]))
        take = dists[:int(k)]
        wsum = 0.0
        vsum = 0.0
        for dist, _donor_id, val in take:
            w = 1.0 / (dist + 15.0)
            wsum += w
            vsum += w * val
        if wsum > 0:
            est[i] = vsum / wsum
    return est


def gr_mismatch_stats(
    lateral_gr: np.ndarray,
    typewell: pd.DataFrame,
    query_tvt: np.ndarray,
    *,
    known_tvt: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Mean |GR_lat − G_tw(t)| on mask (heel-calibrated when known_tvt given)."""
    from rwt.baselines import rmse
    from rwt.gr_match import gr_mismatch_at_tvt

    gr = decode_gr_nulls(lateral_gr)
    mm = gr_mismatch_at_tvt(
        gr,
        typewell["GR"].to_numpy(dtype=float),
        typewell["TVT"].to_numpy(dtype=float),
        query_tvt,
        known_tvt=known_tvt,
    )
    if mask is None:
        mask = np.isfinite(gr) & np.isfinite(query_tvt) & np.isfinite(mm)
    else:
        mask = np.asarray(mask, dtype=bool) & np.isfinite(gr) & np.isfinite(mm)
    if not mask.any():
        return {
            "mean_abs_mismatch": float("nan"),
            "rmse_mismatch": float("nan"),
            "n": 0,
        }
    err = mm[mask]
    return {
        "mean_abs_mismatch": float(np.mean(np.abs(err))),
        "rmse_mismatch": float(rmse(np.zeros(len(err)), err)),
        "n": int(mask.sum()),
    }
