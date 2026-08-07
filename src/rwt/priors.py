"""Shared residual prior column contract (train ↔ submit)."""
from __future__ import annotations

import pandas as pd

# Residual base prior: strongest public heel prior (CERTIFY beat-table intent).
DEFAULT_PRIOR_COL = "prior_last_tvt"
PRIOR_FALLBACKS = ("prior_last_tvt", "prior_mid", "prior_flat_tvt")


def resolve_prior_column(
    df: pd.DataFrame,
    preferred: str = DEFAULT_PRIOR_COL,
) -> str:
    if preferred in df.columns:
        return preferred
    for c in PRIOR_FALLBACKS:
        if c in df.columns:
            return c
    raise KeyError(f"No residual prior column in {list(df.columns)[:24]}")
