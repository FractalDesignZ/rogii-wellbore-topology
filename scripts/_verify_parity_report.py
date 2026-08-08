"""Verify cv_report residual beats best pure floor."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "artifacts/parity_adaptive_serve/cv_report.json")
r = json.loads(path.read_text(encoding="utf-8"))
floors = r.get("baseline_rmse") or {}
best_name = min(floors, key=floors.get)
best_v = float(floors[best_name])
res = float(r.get("oof_rmse"))
sel = float(r.get("oof_rmse_selected"))
ok = res <= best_v + 1e-12
print("=== Gate3 beat table ===")
print(f"residual OOF     : {res:.6f}")
print(f"selected OOF     : {sel:.6f}")
print(f"best pure floor  : {best_name} = {best_v:.6f}")
print(f"residual_le_floor: {ok}")
print(f"use_residual     : {r.get('use_residual')}")
print(f"adaptive_oof     : {r.get('adaptive_oof_rmse')}")
print(f"adaptive_well_oof: {r.get('adaptive_well_oof_rmse')}")
print(f"adaptive_serve   : {r.get('adaptive_serve')}")
print(f"adaptive_conf_thr: {r.get('adaptive_conf_threshold')}")
print(f"adaptive_tuned   : {r.get('adaptive_threshold_tuned')}")
print(
    f"station_fire     : {r.get('adaptive_n_station_fire')}/"
    f"{(r.get('adaptive_n_station_fire') or 0) + (r.get('adaptive_n_station_fallback') or 0)}"
)
print(
    f"well_fire_diag   : {r.get('adaptive_n_fire')}/"
    f"{(r.get('adaptive_n_fire') or 0) + (r.get('adaptive_n_fallback') or 0)}"
)
print(
    f"n_wells={r.get('n_wells')} n_stations={r.get('n_stations')} "
    f"seeds={r.get('n_seed_bag')} n_refit={r.get('n_refit')}"
)
sweep = r.get("adaptive_threshold_sweep") or []
if sweep:
    best = min(
        (s for s in sweep if s.get("adaptive_oof_rmse") is not None),
        key=lambda s: s["adaptive_oof_rmse"],
        default=None,
    )
    print(f"adaptive_sweep_n : {len(sweep)} best={best}")
print("top floors:", {k: round(v, 4) for k, v in sorted(floors.items(), key=lambda x: x[1])[:6]})
model = path.parent / "model.joblib"
print("model.joblib exists:", model.exists())
sys.exit(0 if ok and r.get("use_residual") else 1)
