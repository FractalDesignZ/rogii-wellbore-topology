#!/usr/bin/env python3
"""Watch full_train until cv_report.json exists or train log fails. Prints one DONE/FAILED line."""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# argv: [progress_or_log_path] optional; default parallel out dir
_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts" / "full_train_parallel"
if _arg.is_dir():
    OUT = _arg
    LOG = OUT / "progress.log"
elif _arg.name == "progress.json":
    OUT = _arg.parent
    LOG = OUT / "progress.log"
else:
    # legacy: treat as train log; report next to full_train_parallel preferred
    LOG = _arg
    OUT = ROOT / "artifacts" / "full_train_parallel"
REPORT = OUT / "cv_report.json"
FAIL_RE = re.compile(r"Traceback|SystemExit|Error:|FAIL model")


def main() -> int:
    print(f"watching REPORT={REPORT}", flush=True)
    while True:
        if REPORT.is_file():
            try:
                j = json.loads(REPORT.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"FAILED report_read {e}", flush=True)
                return 1
            print(
                "DONE "
                f"oof={j.get('oof_rmse_selected')} "
                f"residual={j.get('use_residual')} "
                f"prior={j.get('final_prior_type')} "
                f"wells={j.get('n_wells')} "
                f"stations={j.get('n_stations')}",
                flush=True,
            )
            return 0
        if LOG is not None and LOG.is_file():
            try:
                tail = LOG.read_text(encoding="utf-8", errors="replace")[-4000:]
            except OSError:
                tail = ""
            if FAIL_RE.search(tail) and "assembled" not in tail[-200:]:
                # only fail if traceback-like and not mid-assemble noise
                if "Traceback" in tail or "SystemExit" in tail:
                    line = tail.strip().splitlines()[-1] if tail.strip() else "unknown"
                    print(f"FAILED {line}", flush=True)
                    return 1
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
