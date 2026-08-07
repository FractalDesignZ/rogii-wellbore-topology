"""
Load ROGII-style well tables. Column names are normalized to a small schema.

When real Kaggle files arrive, extend `discover_layout` and parsers — do not
hard-fail on one organizer schema version.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

# Canonical column aliases → internal names
ALIASES = {
    "MD": ["MD", "md", "MeasuredDepth", "measured_depth"],
    "X": ["X", "x", "Easting", "easting"],
    "Y": ["Y", "y", "Northing", "northing"],
    "Z": ["Z", "z", "TVD", "tvd", "SS", "ss"],
    "GR": ["GR", "gr", "GammaRay", "gamma_ray", "SGR"],
    "TVT": ["TVT", "tvt", "TrueVerticalThickness"],
    "TVT_input": ["TVT_input", "tvt_input", "TVT_INPUT"],
    "well_id": ["well_id", "WellId", "WELL_ID", "well", "Well"],
    "is_predict": ["is_predict", "predict_mask", "mask", "in_prediction_zone"],
}


def _remap_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lower = {c.lower(): c for c in out.columns}
    rename = {}
    for canon, alts in ALIASES.items():
        for a in alts:
            if a in out.columns:
                rename[a] = canon
                break
            if a.lower() in lower:
                rename[lower[a.lower()]] = canon
                break
    return out.rename(columns=rename)


def discover_layout(root: Path) -> dict[str, Any]:
    root = Path(root)
    train = root / "train"
    test = root / "test"
    csvs = list(root.rglob("*.csv"))
    return {
        "root": str(root),
        "has_train_dir": train.is_dir(),
        "has_test_dir": test.is_dir(),
        "n_csv": len(csvs),
        "sample_csv": [str(p.relative_to(root)) for p in csvs[:20]],
    }


def read_well_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _remap_columns(df)
    if "well_id" not in df.columns:
        df["well_id"] = path.stem
    return df


def iter_well_frames(root: Path, split: str = "train") -> Iterator[tuple[str, pd.DataFrame]]:
    """
    Yield (well_id, df). Supports:
      root/train/*.csv
      root/train/wells/*.csv
      single multi-well CSV with well_id column
    """
    root = Path(root)
    base = root / split if (root / split).is_dir() else root
    csvs = sorted(base.rglob("*.csv"))
    multi = []
    for p in csvs:
        if "sample_submission" in p.name.lower():
            continue
        if "typewell" in p.name.lower() or "type_well" in p.name.lower():
            continue
        df = read_well_csv(p)
        if "well_id" in df.columns and df["well_id"].nunique() > 1:
            multi.append(df)
            continue
        wid = str(df["well_id"].iloc[0]) if "well_id" in df.columns else p.stem
        yield wid, df
    for df in multi:
        for wid, g in df.groupby("well_id"):
            yield str(wid), g.reset_index(drop=True)


def load_typewell(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _remap_columns(df)


def synthetic_well(
    n: int = 200,
    ps_frac: float = 0.65,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic lateral + typewell for smoke tests without Kaggle data."""
    rng = np.random.default_rng(seed)
    md = np.arange(n, dtype=float)
    # build then lateral
    z = np.concatenate(
        [
            np.linspace(0, -800, n // 3),
            np.linspace(-800, -820, n - n // 3),
        ]
    )
    x = np.linspace(0, 4000, n) + rng.normal(0, 2, n)
    y = np.linspace(0, 500, n) + rng.normal(0, 2, n)
    # TVT slowly drifts with some steering noise in lateral
    tvt = 100 + 0.02 * md + np.cumsum(rng.normal(0, 0.05, n))
    # GR correlated with TVT mod formation
    tw_tvt = np.linspace(50, 200, 300)
    tw_gr = 40 + 30 * np.sin(tw_tvt / 15) + rng.normal(0, 2, 300)
    # map lateral GR from nearest typewell TVT
    gr = np.interp(tvt, tw_tvt, tw_gr) + rng.normal(0, 3, n)
    ps = int(n * ps_frac)
    is_pred = np.zeros(n, dtype=bool)
    is_pred[ps:] = True
    lateral = pd.DataFrame(
        {
            "well_id": "SYN01",
            "MD": md,
            "X": x,
            "Y": y,
            "Z": z,
            "GR": gr,
            "TVT": np.where(is_pred, np.nan, tvt),
            "TVT_input": np.where(is_pred, np.nan, tvt),
            "is_predict": is_pred,
            "TVT_true": tvt,  # for smoke eval only
        }
    )
    typewell = pd.DataFrame({"TVT": tw_tvt, "GR": tw_gr})
    return lateral, typewell
