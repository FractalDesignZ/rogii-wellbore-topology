#!/usr/bin/env python3
"""Checkout-compatible wrapper for the installed rwt-smoke command."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rwt.cli import smoke_main

if __name__ == "__main__":
    raise SystemExit(smoke_main())
