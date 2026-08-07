#!/usr/bin/env python3
"""
One-shot late-entry driver:

  RESTRICT → (download if possible) → probe → EDA modes → train residual LGBM → certificate

Does not submit to Kaggle automatically (human accepts rules + uploads CSV).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run(cmd: list[str]) -> int:
    print("RUN", " ".join(cmd), flush=True)
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(ROOT / "src") + (
        (";" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else ""
    )
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


def main() -> int:
    py = sys.executable
    steps = []

    # 1 download (may NEED_INFO)
    rc = run([py, "scripts/download_kaggle.py"])
    steps.append({"step": "download", "rc": rc})
    use_synth = rc != 0
    data = ROOT / "data" / "raw"

    if not use_synth:
        rc = run([py, "scripts/probe_data.py"])
        steps.append({"step": "probe", "rc": rc})
        if rc != 0:
            use_synth = True

    if use_synth:
        print("FALLBACK synthetic abstraction-mode demo (no Kaggle data yet)", flush=True)
        rc = run([py, "scripts/eda_report.py", "--synthetic"])
        steps.append({"step": "eda_synthetic", "rc": rc})
        rc = run(
            [
                py,
                "-m",
                "rwt.train",
                "--synthetic",
                "--out",
                str(ROOT / "artifacts" / "late_synth"),
                "--folds",
                "3",
            ]
        )
        steps.append({"step": "train_synthetic", "rc": rc})
    else:
        rc = run(
            [
                py,
                "scripts/eda_report.py",
                "--data",
                str(data),
                "--max-wells",
                "12",
            ]
        )
        steps.append({"step": "eda_real", "rc": rc})
        rc = run(
            [
                py,
                "-m",
                "rwt.train",
                "--data",
                str(data),
                "--out",
                str(ROOT / "artifacts" / "late_real"),
                "--folds",
                "5",
            ]
        )
        steps.append({"step": "train_real", "rc": rc})

    cert = {
        "artifact": "late_entry_certificate",
        "competition": "rogii-wellbore-geology-prediction",
        "thesis": "TSS abstraction modes on wellbore TVT (GO_MEASURE)",
        "lineage": "https://github.com/ZuluYokohama/topology-sees-sequence",
        "steps": steps,
        "submit": "Human: upload submissions/*.csv after schema align — no auto-submit",
        "ok": all(s["rc"] == 0 for s in steps if s["step"].startswith("train") or s["step"].startswith("eda")),
    }
    out = ROOT / "artifacts" / "late_entry_certificate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cert, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(cert, indent=2))
    return 0 if cert["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
