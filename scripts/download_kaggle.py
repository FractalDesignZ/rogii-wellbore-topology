#!/usr/bin/env python3
"""
Download ROGII competition data into data/raw.

Requires:
  pip install kaggle
  %USERPROFILE%\\.kaggle\\kaggle.json  (or KAGGLE_USERNAME + KAGGLE_KEY)
  Competition rules accepted in the browser.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMP = "rogii-wellbore-geology-prediction"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--no-unzip", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.is_file() and not (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    ):
        print(
            "NEED_INFO: missing Kaggle credentials.\n"
            f"  Place API token at: {kaggle_json}\n"
            "  Or set KAGGLE_USERNAME and KAGGLE_KEY.\n"
            f"  Accept rules: https://www.kaggle.com/competitions/{COMP}/rules",
            file=sys.stderr,
        )
        return 2

    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "competitions",
        "download",
        "-c",
        COMP,
        "-p",
        str(out),
    ]
    print("RUN", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("FAIL download — accept rules? pip install kaggle?", file=sys.stderr)
        return r.returncode

    if not args.no_unzip:
        for z in out.glob("*.zip"):
            print("UNZIP", z.name, flush=True)
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(out)
    print("OK data root:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
