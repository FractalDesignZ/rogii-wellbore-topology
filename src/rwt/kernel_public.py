"""Open-core public baseline kernel (AGPLv3). Safe to publish."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rwt.geometry import trajectory_geometry_features


class PublicKernel:
    def kernel_id(self) -> str:
        return "public.baseline.v1"

    def is_private(self) -> bool:
        return False

    def trajectory_features(
        self,
        md: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> dict[str, Any]:
        return trajectory_geometry_features(md, x, y, z)

    def augment_station_frame(
        self,
        stations: pd.DataFrame,
        lateral: pd.DataFrame,
        typewell: pd.DataFrame | None,
    ) -> pd.DataFrame:
        # Public layer does not add proprietary columns
        out = stations.copy()
        out["kernel_id"] = self.kernel_id()
        out["kernel_private"] = False
        return out
