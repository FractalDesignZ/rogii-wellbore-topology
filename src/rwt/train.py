"""GroupKFold-by-well training (LightGBM residual on heel last-known prior)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from rwt.baselines import rmse, smooth_residual_along_well
from rwt.features import build_station_frame, feature_matrix
from rwt.load import iter_well_frames, load_typewell, synthetic_well, typewell_path_for_well
from rwt.middle_out import MiddleOutSplit, resolve_middle_out, split_by_mask
from rwt.priors import DEFAULT_PRIOR_COL, MAX_TVT_DEV, resolve_prior_column
from rwt.spatial import SpatialIndex, build_spatial_index

# Residual MD smooth window (odd). 0/1 = off.
# 12w measure: w=21 → residual 11.09 vs hold 11.06 (rejected). Keep hook for re-A/B.
RESIDUAL_SMOOTH_WINDOW = 0

# Dual residual: ensemble (prior_best+Δ) with (prior_stack+Δ) when both finite.
# Set-15 equal-mean 10.985 rejected; weighted dual shallow 10.948.
# Set-16 dual deep 10.836; set-17 single deep prior_best **10.830** beats dual.
# Keep dual infrastructure for opt-in A/B (DUAL_RESIDUAL_ENSEMBLE=True).
DUAL_RESIDUAL_ENSEMBLE = False
# Weights by *base name* (not list position) so --prior-col reordering is safe.
DUAL_RESIDUAL_WEIGHT_BY_BASE: dict[str, float] = {
    "prior_best": 0.75,
    "prior_stack": 0.25,
}
# Crown residual LGBM: deep profile is the measure-tuned default (set-16/17).
CROWN_DEEP_RESIDUAL = True
# Serve refit floor under deep (fold best_iters often early-stop << 2000).
N_REFIT_FLOOR_DEEP = 400
N_REFIT_FLOOR_SHALLOW = 200

# Private / proprietary station columns never published in stations.csv
PRIV_STATION_PREFIXES: tuple[str, ...] = (
    "prior_beam",
    "prior_lattice",
    "prior_heatmap",
    "private_",
    "sheaf_",
)


def public_station_columns(stations: pd.DataFrame) -> list[str]:
    """Columns safe to write to public stations.csv (no private stalks)."""
    return [
        c
        for c in stations.columns
        if not any(str(c).casefold().startswith(p) for p in PRIV_STATION_PREFIXES)
    ]


def write_stations_csv(
    stations: pd.DataFrame,
    out_dir: Path,
    *,
    write_private_full: bool = False,
) -> dict[str, Any]:
    """
    Write stations.csv without proprietary columns.

    Residual train should use the full in-memory frame. Optional
    stations_private_full.csv is opt-in only (artifacts/** gitignored) —
    never default-on under caller-controlled --out.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    public_cols = public_station_columns(stations)
    stripped = [c for c in stations.columns if c not in public_cols]
    stations[public_cols].to_csv(out_dir / "stations.csv", index=False)
    manifest: dict[str, Any] = {
        "public_csv": "stations.csv",
        "stripped_columns": stripped,
        "n_public_cols": len(public_cols),
        "n_stripped": len(stripped),
        "write_private_full": bool(write_private_full),
        "note": "stripped columns are private stalks; do not publish or bundle",
    }
    if stripped and write_private_full:
        stations.to_csv(out_dir / "stations_private_full.csv", index=False)
        manifest["private_full_csv"] = "stations_private_full.csv"
    (out_dir / "stations_strip_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def dual_residual_weights_for_bases(
    bases: list[str] | tuple[str, ...],
    weight_by_base: dict[str, float] | None = None,
) -> list[float]:
    """
    Resolve per-head ensemble weights for residual_bases order.

    Uses named weights (default DUAL_RESIDUAL_WEIGHT_BY_BASE). Missing names get
    equal share of remaining mass; if none named, equal weights.
    """
    bases = list(bases)
    n = len(bases)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    wmap = dict(weight_by_base or DUAL_RESIDUAL_WEIGHT_BY_BASE)
    named = [float(wmap[b]) if b in wmap else float("nan") for b in bases]
    if all(np.isfinite(named)):
        s = float(sum(named))
        if s <= 0:
            return [1.0 / n] * n
        return [float(x) / s for x in named]
    # partial names: fill missing with equal share of leftover
    known = sum(x for x in named if np.isfinite(x))
    n_miss = sum(1 for x in named if not np.isfinite(x))
    leftover = max(0.0, 1.0 - known) if known <= 1.0 else 0.0
    fill = leftover / n_miss if n_miss else 0.0
    out = [float(x) if np.isfinite(x) else fill for x in named]
    s = float(sum(out))
    if s <= 0:
        return [1.0 / n] * n
    return [float(x) / s for x in out]


def _typewell_for_well(root: Path, well_id: str, split: str = "train") -> Path | None:
    """Back-compat alias — prefer ``typewell_path_for_well``."""
    return typewell_path_for_well(root, well_id, split)


def select_n_refit(
    fold_best_iters: list[int] | tuple[int, ...] | np.ndarray,
    n_rounds: int,
    *,
    floor: int = 200,
) -> int:
    """
    Serve refit boost rounds from OOF early-stop history.

    Uses max(floor, median, p75) of fold/seed best_iteration, capped at n_rounds.
    When floor ≥ 400 (deep residual), also takes p90 so long early-stop tails
    are not under-refit on serve. Empty history → n_rounds.
    """
    n_rounds = max(1, int(n_rounds))
    floor = max(1, int(floor))
    if fold_best_iters is None or len(fold_best_iters) == 0:
        return n_rounds
    bests = np.asarray([int(x) for x in fold_best_iters], dtype=float)
    if not np.isfinite(bests).any():
        return n_rounds
    bests = bests[np.isfinite(bests)]
    if len(bests) == 0:
        return n_rounds
    p75 = int(np.percentile(bests, 75))
    p90 = int(np.percentile(bests, 90))
    # deep floor (≥400): include p90; shallow: p75 only (set-18 measure)
    tail = p90 if floor >= 400 else p75
    n_refit = max(floor, int(np.median(bests)), p75, tail)
    return int(min(n_refit, n_rounds))


def _assemble_dataset(
    data_root: Path | None,
    *,
    synthetic: bool = False,
    max_wells: int = 0,
    workers: int | None = None,
    progress: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assemble train stations + Gate 2 summary (multi-process when workers>1)."""
    if synthetic or data_root is None:
        lat, tw = synthetic_well()
        sp = split_by_mask(lat, well_id="SYN01")
        frames = [build_station_frame(lat, tw, sp)]
        gate2_meta = {
            "name": "MIDDLE_OUT_MASK",
            "passed": True,
            "n_passed": 1,
            "n_rejected": 0,
            "passed_wells": ["SYN01"],
            "rejected_wells": [],
            "unlabeled_skips": [],
            "results": [{"well_id": "SYN01", "passed": True, "detail": {}}],
        }
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

    from rwt.parallel import assemble_wells_parallel, configure_compute_env

    cfg = configure_compute_env(workers=workers)
    print(
        f"[COMPUTE] workers={cfg['workers']} blas_threads={cfg['blas_threads']} "
        f"cpus={cfg['cpu_count']} — {cfg['note']}",
        flush=True,
    )
    frames, gate2_meta = assemble_wells_parallel(
        loaded,
        spatial,
        workers=int(cfg["workers"]),
        progress=progress,
    )
    if not frames:
        raise FileNotFoundError(f"No labeled predict stations under {root}")
    print(
        f"assembled {len(frames)} wells "
        f"(workers={gate2_meta.get('workers')}, gate2_reject={gate2_meta.get('n_rejected')})",
        flush=True,
    )
    return pd.concat(frames, ignore_index=True), gate2_meta


def train_lgbm_residual(
    stations: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
    prior_col: str | None = None,
    gate2_meta: dict[str, Any] | None = None,
    max_precision: bool = False,
) -> dict[str, Any]:
    """
    Residual learner on labeled *toe* (is_predict).

    Target = TVT − prior. Default residual base is DEFAULT_PRIOR_COL (prior_best).
    Explicit prior_col is strict (must exist); omitted uses DEFAULT with fallbacks.

    max_precision=True: deeper trees, more rounds, more leaves — for 24c/32GB kits
    hunting sub-5ft RMSE (still Gate3 gated).
    """
    if "is_predict" in stations.columns:
        train_df = stations[stations["is_predict"] & stations["TVT"].notna()].copy()
    else:
        train_df = stations[stations["TVT"].notna()].copy()
    if prior_col is None:
        prior_col = resolve_prior_column(train_df, DEFAULT_PRIOR_COL, strict=False)
    else:
        prior_col = resolve_prior_column(train_df, prior_col, strict=True)
    from rwt.features import compute_feature_medians, impute_feature_frame

    # Residual bases: primary + optional dual ensemble partner (best+stack)
    residual_bases: list[str] = [prior_col]
    if DUAL_RESIDUAL_ENSEMBLE:
        for partner in ("prior_best", "prior_stack"):
            if partner != prior_col and partner in train_df.columns:
                residual_bases.append(partner)
                break  # one partner only: best↔stack pair
    residual_bases = list(dict.fromkeys(residual_bases))

    # require finite TVT and *all* residual bases (PF may NaN without typewell)
    fin = np.isfinite(train_df["TVT"].to_numpy(dtype=float))
    for b in residual_bases:
        if b not in train_df.columns:
            raise ValueError(f"residual base column missing: {b}")
        fin &= np.isfinite(train_df[b].to_numpy(dtype=float))
    train_df = train_df.loc[fin].copy()
    if len(train_df) < 50:
        raise ValueError(
            f"Insufficient labeled prediction-zone rows for residual train "
            f"(n={len(train_df)}). Need finite TVT and bases {residual_bases}."
        )

    # Raw features (NaN kept) — impute fold-local medians only (no OOF leak)
    X_raw, cols = feature_matrix(train_df, impute=False)
    y = train_df["TVT"].to_numpy(dtype=float)
    priors_map = {
        b: train_df[b].to_numpy(dtype=float) for b in residual_bases
    }
    y_res_map = {b: y - priors_map[b] for b in residual_bases}
    prior = priors_map[prior_col]
    groups = train_df["well_id"].astype(str).to_numpy()
    # Toe-weighted residual: far-from-PS stations dominate public error
    # (well-inverse-count balancing regressed 12w residual 11.06→11.39)
    if "relative_md" in train_df.columns:
        rel = np.abs(train_df["relative_md"].to_numpy(dtype=float))
        rel = np.where(np.isfinite(rel), rel, 0.0)
        sample_w = 1.0 + np.clip(rel, 0.0, 6000.0) / 3000.0
    else:
        sample_w = np.ones(len(train_df), dtype=float)

    try:
        import lightgbm as lgb
    except ImportError as e:
        raise SystemExit("pip install lightgbm") from e

    n_groups = len(np.unique(groups))
    oof_res_map = {b: np.zeros(len(train_df)) for b in residual_bases}
    fold_best_iters: list[int] = []
    fold_scores: list[float] = []
    def _env_threads(*keys: str, default: int | None = None) -> int:
        for k in keys:
            v = os.environ.get(k)
            if v is None or not str(v).strip():
                continue
            try:
                return max(1, int(v))
            except ValueError:
                continue
        return max(1, default if default is not None else (os.cpu_count() or 4))

    lgb_threads = _env_threads(
        "RWT_LGBM_THREADS", "OMP_NUM_THREADS", "RWT_NUM_THREADS"
    )
    # Clamp residual Δ so head cannot leave pure-prior band (clamp law parity)
    residual_max_dev = float(MAX_TVT_DEV)
    # Set-17/18: single deep prior_best 4-seed; dual opt-in only
    use_deep = bool(max_precision) or bool(CROWN_DEEP_RESIDUAL)
    n_seed_bag = 4 if use_deep else 2
    n_refit_floor = N_REFIT_FLOOR_DEEP if use_deep else N_REFIT_FLOOR_SHALLOW
    if use_deep:
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
            "num_threads": lgb_threads,
            "verbosity": -1,
            "seed": seed,
        }
        n_rounds, early = 2500, 120
    else:
        # L2 residual (Huber regressed 12w OOF 11.27→11.56); seed-bag for variance
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.035,
            "num_leaves": 63,
            "min_data_in_leaf": 80,
            "feature_fraction": 0.80,
            "bagging_fraction": 0.88,
            "bagging_freq": 1,
            "lambda_l2": 2.0,
            "lambda_l1": 0.08,
            "max_bin": 255,
            "num_threads": lgb_threads,
            "verbosity": -1,
            "seed": seed,
        }
        n_rounds, early = 1200, 80

    head_weights = dual_residual_weights_for_bases(residual_bases)

    def _ensemble_heads(parts: list[np.ndarray]) -> np.ndarray:
        """Weighted mean of residual heads (named weights via residual_bases)."""
        stacked = np.vstack(parts)
        if stacked.shape[0] == 1:
            return stacked[0]
        w = np.asarray(head_weights[: stacked.shape[0]], dtype=float)
        if len(w) != stacked.shape[0] or not np.isfinite(w).all() or float(w.sum()) <= 0:
            return np.mean(stacked, axis=0)
        w = w / w.sum()
        return np.average(stacked, axis=0, weights=w)

    def _fit_fold(tr: np.ndarray, va: np.ndarray, fold: int) -> None:
        # Medians from train fold only — never validation wells (CR P1)
        tr_frame = pd.DataFrame(X_raw[tr], columns=cols)
        va_frame = pd.DataFrame(X_raw[va], columns=cols)
        fold_med = compute_feature_medians(tr_frame)
        Xtr = impute_feature_frame(tr_frame, medians=fold_med).to_numpy(dtype=float)
        Xva = impute_feature_frame(va_frame, medians=fold_med).to_numpy(dtype=float)
        head_tvt: list[np.ndarray] = []
        for bi, base in enumerate(residual_bases):
            y_res_b = y_res_map[base]
            dtr = lgb.Dataset(
                Xtr, label=y_res_b[tr], weight=sample_w[tr], feature_name=cols
            )
            dva = lgb.Dataset(
                Xva,
                label=y_res_b[va],
                weight=sample_w[va],
                feature_name=cols,
                reference=dtr,
            )
            preds_bag: list[np.ndarray] = []
            for b in range(n_seed_bag):
                p = dict(params)
                p["seed"] = seed + fold * 17 + bi * 31 + b * 101
                model = lgb.train(
                    p,
                    dtr,
                    num_boost_round=n_rounds,
                    valid_sets=[dva],
                    callbacks=[lgb.early_stopping(early, verbose=False)],
                )
                preds_bag.append(np.asarray(model.predict(Xva), dtype=float))
                fold_best_iters.append(int(model.best_iteration or n_rounds))
            pred_res = np.mean(np.vstack(preds_bag), axis=0)
            pred_res = np.clip(pred_res, -residual_max_dev, residual_max_dev)
            if RESIDUAL_SMOOTH_WINDOW > 1:
                order_key = None
                if "MD" in train_df.columns:
                    order_key = train_df["MD"].to_numpy(dtype=float)[va]
                elif "station" in train_df.columns:
                    order_key = train_df["station"].to_numpy(dtype=float)[va]
                pred_res = smooth_residual_along_well(
                    pred_res,
                    groups[va],
                    window=RESIDUAL_SMOOTH_WINDOW,
                    order_key=order_key,
                )
                pred_res = np.clip(pred_res, -residual_max_dev, residual_max_dev)
            oof_res_map[base][va] = pred_res
            head_tvt.append(priors_map[base][va] + pred_res)
        oof_ens = _ensemble_heads(head_tvt)
        fold_scores.append(rmse(y[va], oof_ens))

    oof_mask = np.zeros(len(train_df), dtype=bool)
    y_res_primary = y_res_map[prior_col]
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
        for fold, (tr, va) in enumerate(gkf.split(X_raw, y_res_primary, groups)):
            _fit_fold(tr, va, fold)
            oof_mask[va] = True

    # Serve medians: full labeled toe set (no test leakage); stored on model blob
    feature_medians = compute_feature_medians(pd.DataFrame(X_raw, columns=cols))
    # Ensemble OOF TVT = weighted mean over residual heads (prior_b + Δ_b)
    head_full = [
        priors_map[b]
        + np.clip(oof_res_map[b], -residual_max_dev, residual_max_dev)
        for b in residual_bases
    ]
    oof_tvt = _ensemble_heads(head_full)
    # Note: do NOT re-clamp residual head to last±max_dev — Gate3 residual may
    # correctly leave the pure-prior band (diag: last clamp regressed 11.08→11.35).
    from rwt.kernel_api import resolve_kernel
    from rwt.provenance import seal_artifact

    k = resolve_kernel()
    row_m = oof_mask if oof_mask.any() else np.ones(len(train_df), dtype=bool)
    y_m = y[row_m]
    oof_m = oof_tvt[row_m]
    shared = np.isfinite(y_m) & np.isfinite(oof_m)
    y_s = y_m[shared]
    oof_s = oof_m[shared]
    oof_r = rmse(y_s, oof_s)
    # Single definition: baseline_rmse and Gate3 floors share this map
    _FLOOR_COL: dict[str, str] = {
        "last_known": "prior_last_tvt",
        "local_dip_ps": "prior_local_dip",
        "flat_formation": "prior_flat_tvt",
        "linear_md_extrap": "prior_extrap_tvt",
        "dual_transport_mid": "prior_mid",
        "heel_dip_stack": "prior_dip_stack",
        "particle_filter": "prior_pf_tvt",
        "pf_blend_last_known": "prior_pf_blend",
        "pf_bag": "prior_pf_bag",
        "prior_stack": "prior_stack",
        "prior_best": "prior_best",
        "prior_ncc": "prior_ncc_tvt",
        # private kernel pure floor (present only when RWT_KERNEL=private)
        "prior_beam_blend": "prior_beam_blend",
        "prior_beam_tvt": "prior_beam_tvt",
        "prior_lattice_blend": "prior_lattice_blend",
        "prior_lattice_tvt": "prior_lattice_tvt",
    }
    baseline_rmse: dict[str, float] = {}
    for name, col in _FLOOR_COL.items():
        if col not in train_df.columns:
            continue
        b = train_df[col].to_numpy(dtype=float)[row_m]
        if not np.isfinite(b[shared]).all():
            continue
        baseline_rmse[name] = rmse(y_s, b[shared])
    beats = {
        name: bool(oof_r < br) for name, br in baseline_rmse.items() if np.isfinite(br)
    }

    from rwt.gates import gate_ip_fence, gate_residual_acceptance, gate_residual_acceptance_per_well
    from rwt.simd import cpu_feature_report

    prior_m = prior[row_m][shared]
    lk_m = None
    if "prior_last_tvt" in train_df.columns:
        lk_full = train_df["prior_last_tvt"].to_numpy(dtype=float)[row_m]
        if np.isfinite(lk_full[shared]).all():
            lk_m = lk_full[shared]
    floor_rmses = {
        n: baseline_rmse[n]
        for n in _FLOOR_COL
        if n in baseline_rmse and np.isfinite(baseline_rmse[n])
    }
    gate3 = gate_residual_acceptance(
        y_s,
        oof_s,
        prior_m,
        prior_name=prior_col,
        last_known_pred=lk_m,
        floor_rmses=floor_rmses,
    )
    gate1 = gate_ip_fence()
    use_residual = bool(gate3.passed)

    pf_conf_s = None
    if "pf_confidence" in train_df.columns:
        pf_full = train_df["pf_confidence"].to_numpy(dtype=float)[row_m]
        if np.isfinite(pf_full[shared]).any():
            pf_conf_s = pf_full[shared]
    groups_s = groups[row_m][shared]
    gate3_pw = gate_residual_acceptance_per_well(
        groups_s,
        y_s,
        oof_s,
        prior_m,
        global_gate=gate3,
        pf_confidence=pf_conf_s,
    )
    serve_prior_col = prior_col
    residual_heads: list[dict[str, Any]] = []
    final_models: list[Any] = []
    if use_residual:
        print(
            f"[GATE PASS] Residual head accepted: OOF {gate3.detail['oof_rmse_residual']:.3f} "
            f"<= best_floor({gate3.detail.get('best_floor_name')}) "
            f"{gate3.detail.get('oof_rmse_best_floor'):.3f}"
            f" bases={residual_bases}",
            flush=True,
        )
        # Refit serve ensemble on *all* train rows with full-train medians so
        # serve impute matches training impute (CR P1 train/serve contract).
        X_serve = impute_feature_frame(
            pd.DataFrame(X_raw, columns=cols), medians=feature_medians
        ).to_numpy(dtype=float)
        n_refit = select_n_refit(fold_best_iters, n_rounds, floor=n_refit_floor)
        for bi, base in enumerate(residual_bases):
            d_full = lgb.Dataset(
                X_serve,
                label=y_res_map[base],
                weight=sample_w,
                feature_name=cols,
            )
            models_b: list[Any] = []
            for b in range(n_seed_bag):
                p = dict(params)
                p["seed"] = seed + 900 + bi * 47 + b * 101
                models_b.append(lgb.train(p, d_full, num_boost_round=n_refit))
            residual_heads.append({"prior_column": base, "models": models_b})
        # Legacy single-head blob fields = primary base (back-compat)
        final_models = residual_heads[0]["models"]
        if len(residual_bases) > 1:
            final_prior_type = (
                f"dual_residual({'+'.join(residual_bases)})_lgbm"
            )
        else:
            final_prior_type = f"{prior_col}_plus_lgbm_residual"
        final_oof = oof_r
        adaptive_selected = False
        # Prefer adaptive OOF when station-conf mask beats global residual (still
        # Gate3-accepted: residual head allowed only after global pass).
        # adaptive_oof uses the SAME adaptive_residual_mask rule as predict_stations.
        a_oof = gate3_pw.detail.get("adaptive_oof_rmse")
        if (
            a_oof is not None
            and np.isfinite(float(a_oof))
            and float(a_oof) < float(final_oof) - 1e-12
        ):
            final_oof = float(a_oof)
            final_prior_type = f"{final_prior_type}_adaptive"
            adaptive_selected = True
            print(
                f"[ADAPTIVE] selected OOF {final_oof:.3f} "
                f"(global residual {oof_r:.3f}; "
                f"mask={gate3_pw.detail.get('adaptive_mask_mode')} "
                f"station_fire={gate3_pw.detail.get('n_station_fire')} "
                f"station_fb={gate3_pw.detail.get('n_station_fallback')} "
                f"well_fire_diag={gate3_pw.detail.get('n_fire')}/"
                f"{gate3_pw.detail.get('n_fallback')})",
                flush=True,
            )
    else:
        adaptive_selected = False
        n_refit = None
        cand: dict[str, float] = {prior_col: float(gate3.detail["oof_rmse_prior"])}
        for n, s in floor_rmses.items():
            col = _FLOOR_COL.get(n, n)
            if col in train_df.columns:
                cand[col] = float(s)
        serve_prior_col = min(cand, key=cand.get)
        final_oof = cand[serve_prior_col]
        final_models = []
        residual_heads = []
        final_prior_type = f"pure_{serve_prior_col}"
        print(
            f"[GATE REJECT] Residual OOF {gate3.detail['oof_rmse_residual']:.3f} "
            f"loses best floor {gate3.detail.get('best_floor_name')}="
            f"{gate3.detail.get('oof_rmse_best_floor')}. "
            f"Serve pure '{serve_prior_col}' (OOF {final_oof:.3f}).",
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
        "max_precision": bool(max_precision),
        "deep_residual": bool(use_deep),
        "lgb_threads": int(lgb_threads),
        "n_rounds_cap": int(n_rounds),
        "n_refit": int(n_refit) if use_residual and n_refit is not None else None,
        "fold_best_iters": [int(x) for x in fold_best_iters],
        "n_seed_bag": int(n_seed_bag),
        "residual_bases": list(residual_bases),
        "dual_residual": bool(len(residual_bases) > 1),
        "dual_residual_weights": (
            list(head_weights) if len(residual_bases) > 1 else None
        ),
        "residual_max_dev": float(residual_max_dev),
        "residual_smooth_window": int(RESIDUAL_SMOOTH_WINDOW) if use_residual else 0,
        "feature_medians": {k: float(v) for k, v in feature_medians.items()},
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
                "passed": None,
                "evaluated": False,
                "note": "not evaluated (caller assembled stations externally)",
            },
            gate3.to_dict(),
            gate3_pw.to_dict(),
        ],
        "adaptive_oof_rmse": gate3_pw.detail.get("adaptive_oof_rmse"),
        "adaptive_n_fire": gate3_pw.detail.get("n_fire"),
        "adaptive_n_fallback": gate3_pw.detail.get("n_fallback"),
        "adaptive_n_station_fire": gate3_pw.detail.get("n_station_fire"),
        "adaptive_n_station_fallback": gate3_pw.detail.get("n_station_fallback"),
        "adaptive_mask_mode": gate3_pw.detail.get("adaptive_mask_mode"),
        "adaptive_serve": bool(adaptive_selected),
        "adaptive_conf_threshold": 0.35,
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
        "residual_heads": residual_heads,
        "dual_residual_weights": (
            list(head_weights) if use_residual and len(residual_bases) > 1 else None
        ),
        "feature_cols": cols,
        "prior_column": report["prior_column"],
        "use_residual": use_residual,
        "feature_medians": feature_medians,
        "residual_max_dev": residual_max_dev,
        "residual_smooth_window": int(RESIDUAL_SMOOTH_WINDOW) if use_residual else 0,
        "adaptive_serve": bool(adaptive_selected),
        "adaptive_conf_threshold": 0.35,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train RWT LGBM residual TVT model")
    ap.add_argument("--data", default="", help="path to Kaggle data root")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic well only")
    ap.add_argument("--out", default="artifacts/run1")
    ap.add_argument("--max-wells", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="process workers for well assemble (0=auto cpu-1)",
    )
    ap.add_argument(
        "--prior-col",
        default=None,
        help=f"residual base prior column (default: {DEFAULT_PRIOR_COL})",
    )
    ap.add_argument(
        "--max-precision",
        action="store_true",
        help="deeper LGBM residual (still Gate3; RESTRICT: no Kaggle submit)",
    )
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from rwt.parallel import configure_compute_env
    from rwt.progress import init_progress

    workers = args.workers if args.workers > 0 else None
    cfg = configure_compute_env(workers=workers)
    progress = init_progress(out)
    progress.state["workers"] = int(cfg["workers"])
    try:
        stations, gate2_meta = _assemble_dataset(
            Path(args.data) if args.data else None,
            synthetic=args.synthetic or not args.data,
            max_wells=args.max_wells,
            workers=int(cfg["workers"]),
            progress=progress,
        )
        write_stations_csv(stations, out, write_private_full=False)
        progress.set_phase(
            "train",
            total=args.folds,
            message=f"stations={len(stations)} wells≈{stations['well_id'].nunique()}",
        )
        progress.update_metrics(
            {
                "n_stations": int(len(stations)),
                "n_wells_stations": int(stations["well_id"].nunique())
                if "well_id" in stations.columns
                else 0,
            }
        )
        result = train_lgbm_residual(
            stations,
            n_splits=args.folds,
            prior_col=args.prior_col,
            gate2_meta=gate2_meta,
            max_precision=bool(args.max_precision),
        )
        report = result["report"]
        (out / "cv_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        try:
            import joblib

            joblib.dump(
                {
                    "models": result["models"],
                    "residual_heads": result.get("residual_heads") or [],
                    "dual_residual_weights": result.get("dual_residual_weights"),
                    "feature_cols": result["feature_cols"],
                    "prior_column": result["prior_column"],
                    "use_residual": result.get("use_residual", True),
                    "residual_max_dev": float(
                        result.get("residual_max_dev", MAX_TVT_DEV)
                    ),
                    "residual_smooth_window": int(
                        result.get("residual_smooth_window") or 0
                    ),
                    "feature_medians": result.get("feature_medians") or {},
                    # Adaptive serve: station-conf mask (shared with train adaptive OOF)
                    # only when adaptive path beat global residual OOF.
                    "adaptive_serve": bool(result.get("adaptive_serve", False)),
                    "adaptive_conf_threshold": float(
                        result.get("adaptive_conf_threshold", 0.35)
                    ),
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
            progress.fail(str(e)[:200])
            print(json.dumps(report, indent=2))
            return 1
        progress.finish(
            "train complete",
            oof_rmse_selected=report.get("oof_rmse_selected"),
            use_residual=report.get("use_residual"),
            final_prior_type=report.get("final_prior_type"),
            n_wells=report.get("n_wells"),
            baseline_rmse=report.get("baseline_rmse"),
        )
        print(json.dumps(report, indent=2))
        return 0 if report.get("ok") else 1
    except Exception as e:
        progress.fail(str(e)[:400])
        raise


if __name__ == "__main__":
    raise SystemExit(main())
