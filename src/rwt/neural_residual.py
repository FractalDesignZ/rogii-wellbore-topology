"""
CUDA residual head for ΔTVT — Blackwell-class path under Gate 3.

Design law (same as LGBM residual):
  predict residual only; accept only if OOF beats pure prior floors.
  Cosine / vibes never OPEN. Metric is RMSE(ft).

Architecture (public, open-core):
  - Per-well sequences of station features (GR rolls, kinematics, priors)
  - 1D temporal CNN + global pool + residual MLP → ΔTVT
  - GroupKFold by well_id; train on GPU when available

NPU: export path reserved (ONNX) — product accept still CPU/CUDA OOF + Gate3.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from rwt.baselines import rmse
from rwt.features import feature_matrix
from rwt.gates import gate_residual_acceptance
from rwt.priors import DEFAULT_PRIOR_COL, resolve_prior_column


@dataclass
class NeuralReport:
    ok: bool
    device: str
    oof_rmse: float
    oof_rmse_selected: float
    use_residual: bool
    baseline_rmse: dict[str, float]
    gate3: dict[str, Any]
    fold_rmse: list[float]
    n_stations: int
    n_wells: int
    seconds: float
    detail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "device": self.device,
            "oof_rmse": self.oof_rmse,
            "oof_rmse_selected": self.oof_rmse_selected,
            "use_residual": self.use_residual,
            "baseline_rmse": self.baseline_rmse,
            "gate3": self.gate3,
            "fold_rmse": self.fold_rmse,
            "n_stations": self.n_stations,
            "n_wells": self.n_wells,
            "seconds": self.seconds,
            "detail": self.detail,
            "target_ft": 5.0,
            "beats_5ft": self.oof_rmse_selected < 5.0,
            "law": "Gate3 residual; GPU trains ΔTVT only",
        }


def _device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _build_model(n_feat: int, hidden: int = 128):
    import torch
    import torch.nn as nn

    class ResidualTCN(nn.Module):
        """Temporal conv residual: (B, F, L) → ΔTVT per station (B, L)."""

        def __init__(self, n_feat: int, hidden: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(n_feat, hidden, kernel_size=7, padding=3),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(hidden, 1, kernel_size=1),
            )
            # zero-init last layer → residual starts near 0 (safe under Gate3)
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        def forward(self, x):
            # x: (B, L, F) → (B, F, L)
            y = self.net(x.transpose(1, 2)).squeeze(1)  # (B, L)
            return y

    return ResidualTCN(n_feat, hidden)


def _well_tensors(
    train_df: pd.DataFrame,
    cols: list[str],
    prior_col: str,
    max_len: int = 4096,
) -> list[dict[str, Any]]:
    """Pack each well as fixed-length sequence (truncate/pad).

    Stores original ``row_idx`` into ``train_df`` so OOF write-back never
    mis-aligns when sequences are truncated to ``max_len``.
    """
    out = []
    # stable row positions into train_df (GroupKFold OOF buffer)
    train_df = train_df.copy()
    train_df["_row_i"] = np.arange(len(train_df), dtype=int)
    for wid, g in train_df.groupby("well_id", sort=False):
        g = g.sort_values("station") if "station" in g.columns else g
        row_i = g["_row_i"].to_numpy(dtype=int)
        X = g[cols].to_numpy(dtype=np.float32)
        y = g["TVT"].to_numpy(dtype=np.float32)
        prior = g[prior_col].to_numpy(dtype=np.float32)
        n = len(g)
        if n > max_len:
            # keep toe-heavy: last max_len
            X, y, prior, row_i = (
                X[-max_len:],
                y[-max_len:],
                prior[-max_len:],
                row_i[-max_len:],
            )
            n = max_len
        out.append(
            {
                "well_id": str(wid),
                "X": X,
                "y": y,
                "prior": prior,
                "y_res": y - prior,
                "row_i": row_i,
                "n": n,
            }
        )
    return out


def train_neural_residual(
    stations: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
    prior_col: str | None = None,
    epochs: int = 40,
    batch_wells: int = 8,
    lr: float = 1e-3,
    max_len: int = 2048,
    hidden: int = 128,
) -> NeuralReport:
    """
    GroupKFold neural residual on GPU. Same Gate3 law as LGBM path.
    """
    t0 = time.perf_counter()
    import torch

    dev = _device()
    device = torch.device(dev)

    if "is_predict" in stations.columns:
        train_df = stations[stations["is_predict"] & stations["TVT"].notna()].copy()
    else:
        train_df = stations[stations["TVT"].notna()].copy()

    if prior_col is None:
        prior_col = resolve_prior_column(train_df, DEFAULT_PRIOR_COL, strict=False)
    else:
        prior_col = resolve_prior_column(train_df, prior_col, strict=True)

    fin = np.isfinite(train_df["TVT"].to_numpy(float)) & np.isfinite(
        train_df[prior_col].to_numpy(float)
    )
    train_df = train_df.loc[fin].copy()
    X_all, cols = feature_matrix(train_df)
    # put feature cols back for packing
    for i, c in enumerate(cols):
        train_df[c] = X_all[:, i]
    # sanitize
    for c in cols:
        train_df[c] = (
            train_df[c]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
        )

    wells = _well_tensors(train_df, cols, prior_col, max_len=max_len)
    well_ids = np.array([w["well_id"] for w in wells])
    n_groups = len(wells)
    if n_groups < 2:
        raise ValueError("need ≥2 wells for GroupKFold neural residual")

    # flatten for OOF / Gate3 (station-level)
    y_flat = train_df["TVT"].to_numpy(float)
    prior_flat = train_df[prior_col].to_numpy(float)

    gkf = GroupKFold(n_splits=min(n_splits, n_groups))
    # station-level OOF storage
    oof_pred = np.full(len(train_df), np.nan, dtype=float)
    fold_scores: list[float] = []

    torch.manual_seed(seed)
    if dev == "cuda":
        torch.cuda.manual_seed_all(seed)

    idx = np.arange(n_groups)
    for fold, (tr_w, va_w) in enumerate(gkf.split(idx, groups=well_ids)):
        tr_wells = [wells[i] for i in tr_w]
        va_wells = [wells[i] for i in va_w]
        model = _build_model(len(cols), hidden=hidden).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        # amp for Blackwell / modern CUDA
        use_amp = dev == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        def _batch_list(items: list, bs: int):
            for i in range(0, len(items), bs):
                yield items[i : i + bs]

        model.train()
        for ep in range(epochs):
            order = np.arange(len(tr_wells))
            np.random.default_rng(seed + fold * 1000 + ep).shuffle(order)
            tr_ordered = [tr_wells[i] for i in order]
            for batch in _batch_list(tr_ordered, batch_wells):
                # pad batch to max n in batch
                L = max(b["n"] for b in batch)
                F = len(cols)
                B = len(batch)
                xb = np.zeros((B, L, F), dtype=np.float32)
                yb = np.zeros((B, L), dtype=np.float32)
                mb = np.zeros((B, L), dtype=np.float32)
                for i, b in enumerate(batch):
                    n = b["n"]
                    xb[i, :n] = b["X"]
                    yb[i, :n] = b["y_res"]
                    mb[i, :n] = 1.0
                xt = torch.from_numpy(xb).to(device)
                yt = torch.from_numpy(yb).to(device)
                mt = torch.from_numpy(mb).to(device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(xt)
                    loss = ((pred - yt) ** 2 * mt).sum() / mt.sum().clamp_min(1.0)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

        # OOF predict
        model.eval()
        with torch.no_grad():
            y_true_f, y_pred_f = [], []
            for b in va_wells:
                xt = torch.from_numpy(b["X"][None, ...]).to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pr = model(xt).float().cpu().numpy().ravel()  # residual
                n = b["n"]
                pred_tvt = b["prior"] + pr[:n]
                # exact row mapping (truncation-safe)
                take = b["row_i"]
                if len(take) != n:
                    n = min(len(take), n)
                    take = take[:n]
                    pred_tvt = pred_tvt[:n]
                oof_pred[take] = pred_tvt
                y_true_f.append(b["y"][:n])
                y_pred_f.append(pred_tvt)
            yt = np.concatenate(y_true_f)
            yp = np.concatenate(y_pred_f)
            fold_scores.append(rmse(yt, yp))
        print(
            f"[neural fold {fold}] device={dev} rmse={fold_scores[-1]:.4f} "
            f"n_va_wells={len(va_wells)}",
            flush=True,
        )
        if dev == "cuda":
            torch.cuda.empty_cache()

    # Gate3 floors must share the same OOF row set as residual (no full-frame leak)
    m = np.isfinite(oof_pred) & np.isfinite(y_flat) & np.isfinite(prior_flat)
    oof_r = rmse(y_flat[m], oof_pred[m]) if m.any() else float("inf")
    prior_m = prior_flat[m]
    y_m = y_flat[m]
    oof_m = oof_pred[m]
    if not np.any(m):
        return NeuralReport(
            ok=False,
            device=dev,
            oof_rmse=float("inf"),
            oof_rmse_selected=float("inf"),
            use_residual=False,
            baseline_rmse={},
            gate3={"name": "RESIDUAL_ACCEPTANCE", "passed": False, "error": "empty_oof"},
            fold_rmse=[float(x) for x in fold_scores],
            n_stations=int(len(train_df)),
            n_wells=int(n_groups),
            seconds=round(time.perf_counter() - t0, 2),
            detail={"error": "no finite OOF predictions"},
        )
    # Gate3 floors must be scored on the *same* OOF rows as residual (LGBM parity).
    # Skip floors with holes on m so beat_all_floors is a fair comparison.
    floor_rmses: dict[str, float] = {}
    for name, col in (
        ("last_known", "prior_last_tvt"),
        ("local_dip_ps", "prior_local_dip"),
        ("flat_formation", "prior_flat_tvt"),
        ("linear_md_extrap", "prior_extrap_tvt"),
        ("particle_filter", "prior_pf_tvt"),
        ("pf_blend_last_known", "prior_pf_blend"),
        ("pf_bag", "prior_pf_bag"),
        ("prior_stack", "prior_stack"),
        ("prior_best", "prior_best"),
    ):
        if col not in train_df.columns:
            continue
        b = train_df[col].to_numpy(float)
        if not np.isfinite(b[m]).all():
            continue
        floor_rmses[name] = rmse(y_flat[m], b[m])
    baseline_rmse = dict(floor_rmses)
    lk_m = None
    if "prior_last_tvt" in train_df.columns:
        lk = train_df["prior_last_tvt"].to_numpy(float)[m]
        if np.isfinite(lk).all():
            lk_m = lk
    gate3 = gate_residual_acceptance(
        y_m,
        oof_m,
        prior_m,
        prior_name=prior_col,
        last_known_pred=lk_m,
        floor_rmses=floor_rmses,
    )
    use_res = bool(gate3.passed)
    if use_res:
        selected = oof_r
        print(
            f"[GATE PASS] Neural residual OOF {oof_r:.3f} <= best_floor "
            f"{gate3.detail.get('best_floor_name')} "
            f"{gate3.detail.get('oof_rmse_best_floor')}",
            flush=True,
        )
    else:
        # Prefer Gate3's best floor on residual OOF rows (comparable), not min of
        # per-column subsets with different missingness.
        g3_best = gate3.detail.get("oof_rmse_best_floor")
        if g3_best is not None and np.isfinite(float(g3_best)):
            selected = float(g3_best)
        elif floor_rmses:
            selected = float(min(floor_rmses.values()))
        else:
            selected = float(gate3.detail.get("oof_rmse_prior") or oof_r)
        print(
            f"[GATE REJECT] Neural residual OOF {oof_r:.3f} loses floors; "
            f"selected pure floor {selected:.3f} "
            f"({gate3.detail.get('best_floor_name') or 'prior'})",
            flush=True,
        )

    return NeuralReport(
        ok=True,
        device=dev,
        oof_rmse=float(oof_r),
        oof_rmse_selected=float(selected),
        use_residual=use_res,
        baseline_rmse=baseline_rmse,
        gate3=gate3.to_dict(),
        fold_rmse=[float(x) for x in fold_scores],
        n_stations=int(len(train_df)),
        n_wells=int(n_groups),
        seconds=round(time.perf_counter() - t0, 2),
        detail={
            "prior_col": prior_col,
            "epochs": epochs,
            "hidden": hidden,
            "max_len": max_len,
            "amp": dev == "cuda",
            "arch": "ResidualTCN_1d",
            "cuda_name": (
                torch.cuda.get_device_name(0) if dev == "cuda" else None
            ),
        },
    )


def export_onnx_for_npu(
    stations: pd.DataFrame,
    out_path: Path,
    *,
    prior_col: str | None = None,
    max_len: int = 512,
    hidden: int = 128,
    epochs: int = 5,
) -> dict[str, Any]:
    """
    Structure-only ONNX export of ResidualTCN for OpenVINO / NPU experiments.

    Does **not** re-train (avoids double MEASURE). Weights are zero-init residual
    head — not for production score. Product accept remains CUDA/CPU OOF + Gate3.
    ``epochs`` retained for API compat; ignored.
    """
    import torch

    _ = epochs  # API compat; structure export only
    if "is_predict" in stations.columns:
        df = stations[stations["is_predict"] & stations["TVT"].notna()].copy()
    else:
        df = stations[stations["TVT"].notna()].copy()
    if prior_col is None:
        prior_col = resolve_prior_column(df, DEFAULT_PRIOR_COL, strict=False)
    _X, cols = feature_matrix(df)
    model = _build_model(len(cols), hidden=hidden)
    model.eval()
    dummy = torch.zeros(1, max_len, len(cols), dtype=torch.float32)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            input_names=["features"],
            output_names=["delta_tvt"],
            dynamic_axes={
                "features": {0: "batch", 1: "len"},
                "delta_tvt": {0: "batch", 1: "len"},
            },
            opset_version=17,
            dynamo=False,
        )
        onnx_ok = True
        onnx_err = None
    except Exception as e:
        onnx_ok = False
        onnx_err = f"{type(e).__name__}: {e}"
    return {
        "onnx": str(out_path) if onnx_ok else None,
        "onnx_ok": onnx_ok,
        "onnx_error": onnx_err,
        "n_feat": len(cols),
        "max_len": max_len,
        "weights": "structure_only_zero_residual_head",
        "note": "NPU experimental structure export — not a scored model; Gate3 CUDA/CPU OOF is authority",
        "prior_col": prior_col,
    }
