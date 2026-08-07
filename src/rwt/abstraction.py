"""
Executable abstraction modes for the late-entry pipeline.

These are the *compute modes* we demonstrate — not marketing OPEN.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from rwt.baselines import (
    baseline_flat_formation,
    baseline_last_known,
    baseline_linear_md,
    rmse,
)
from rwt.features import build_station_frame, feature_matrix
from rwt.geometry import trajectory_geometry_features
from rwt.gates import gate_ip_fence, gate_middle_out_mask, run_entry_gates
from rwt.geometry import trajectory_in_dip_frame
from rwt.middle_out import MiddleOutSplit, known_prior_tvt, resolve_middle_out
from rwt.simd import cpu_feature_report


class Mode(str, Enum):
    RESTRICT = "M0_RESTRICT"
    GEOM = "M1_GEOM"
    PRIOR = "M2_PRIOR"
    MATCH = "M3_MATCH"
    MIDDLE_OUT = "M4_MIDDLE_OUT"
    RESIDUAL = "M5_RESIDUAL"
    CERTIFY = "M6_CERTIFY"


@dataclass
class ModeTrace:
    """Certificate-style trail of which modes ran."""

    modes: list[str] = field(default_factory=list)
    payloads: dict[str, Any] = field(default_factory=dict)

    def record(self, mode: Mode, payload: dict[str, Any] | None = None) -> None:
        self.modes.append(mode.value)
        if payload is not None:
            self.payloads[mode.value] = payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "modes_executed": list(self.modes),
            "payloads": self.payloads,
            "law": "GO_MEASURE abstraction modes; TVT RMSE is the only score; no production OPEN",
            "lineage": "topology-sees-sequence → open curve + middle-out prior + residual learner",
        }


def mode_restrict() -> dict[str, Any]:
    gates = run_entry_gates()
    return {
        "target": "TVT",
        "metric": "RMSE(dTVT)",
        "cv": "GroupKFold by well_id",
        "gates_boot": gates,
        "cpu": cpu_feature_report(),
        "non_goals": [
            "peptide ramachandran",
            "closed holonomy as TVT",
            "production geosteering OPEN",
        ],
    }


def mode_geom(lateral: pd.DataFrame) -> dict[str, Any]:
    g = trajectory_geometry_features(
        lateral["MD"].to_numpy(),
        lateral["X"].to_numpy(),
        lateral["Y"].to_numpy(),
        lateral["Z"].to_numpy(),
    )
    dip = trajectory_in_dip_frame(
        lateral["X"].to_numpy(),
        lateral["Y"].to_numpy(),
        lateral["Z"].to_numpy(),
    )
    g["dip_azimuth_deg"] = dip["azimuth_deg"]
    g["dip_deg"] = dip["dip_deg"]
    # JSON-safe scalars only for payload
    return {
        k: (float(v) if isinstance(v, (float, np.floating, int)) else v)
        for k, v in g.items()
        if np.isscalar(v) or isinstance(v, (str, bool))
    }


def mode_middle_out(lateral: pd.DataFrame, well_id: str) -> MiddleOutSplit:
    return resolve_middle_out(lateral, well_id=well_id)


def mode_prior(lateral: pd.DataFrame, split: MiddleOutSplit) -> dict[str, float]:
    return known_prior_tvt(lateral, split)


def mode_baselines(
    lateral: pd.DataFrame,
    split: MiddleOutSplit,
    *,
    y_true: np.ndarray | None = None,
) -> dict[str, Any]:
    preds = {
        "last_known": baseline_last_known(lateral, split),
        "flat_formation": baseline_flat_formation(lateral, split),
        "linear_md": baseline_linear_md(lateral, split),
    }
    out: dict[str, Any] = {"n_predict": int(split.n_predict)}
    if y_true is not None and len(y_true) == split.n_predict:
        out["rmse"] = {k: rmse(y_true, v) for k, v in preds.items()}
    return out


def run_well_modes(
    lateral: pd.DataFrame,
    typewell: pd.DataFrame | None,
    *,
    well_id: str,
    y_true_col: str | None = "TVT_true",
) -> tuple[pd.DataFrame, ModeTrace]:
    """
    Run M0–M4 (+ baseline MEASURE) for one well; return station frame + trace.
    Residual learner (M5) is separate (train.py) so CV stays honest.
    """
    trace = ModeTrace()
    trace.record(Mode.RESTRICT, mode_restrict())
    trace.record(Mode.GEOM, mode_geom(lateral))
    split = mode_middle_out(lateral, well_id)
    trace.record(Mode.MIDDLE_OUT, split.to_dict())
    trace.record(Mode.PRIOR, mode_prior(lateral, split))

    y_true = None
    if y_true_col and y_true_col in lateral.columns:
        y_true = lateral.iloc[split.predict_idx][y_true_col].to_numpy(dtype=float)
    elif "TVT" in lateral.columns and lateral.iloc[split.predict_idx]["TVT"].notna().any():
        y_true = lateral.iloc[split.predict_idx]["TVT"].to_numpy(dtype=float)

    lat_for_base = lateral
    if y_true is not None and "TVT" not in lateral.columns:
        lat_for_base = lateral.copy()
    elif y_true is not None and lateral["TVT"].isna().any() and y_true_col in lateral.columns:
        lat_for_base = lateral.assign(TVT=lateral[y_true_col])

    trace.record(Mode.MATCH, {"typewell_rows": 0 if typewell is None else int(len(typewell))})
    stations = build_station_frame(lateral, typewell, split)
    # MATCH features live inside station frame (ncc_*); record presence
    ncc_cols = [c for c in stations.columns if c.startswith("ncc_")]
    trace.payloads[Mode.MATCH.value] = {
        "typewell_rows": 0 if typewell is None else int(len(typewell)),
        "ncc_features": ncc_cols,
    }
    trace.record(
        Mode.CERTIFY,
        {
            "baselines": mode_baselines(lat_for_base, split, y_true=y_true),
            "n_stations": int(len(stations)),
            "feature_preview": list(stations.columns)[:24],
        },
    )
    # M5 not run here
    return stations, trace
