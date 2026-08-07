"""
Kernel boundary — public interface vs proprietary implementation.

Public CI always uses PublicKernel. PrivateKernel loads only when
RWT_KERNEL=private and private_kernel is importable.
"""
from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class GeometryKernel(Protocol):
    def kernel_id(self) -> str: ...

    def is_private(self) -> bool: ...

    def trajectory_features(
        self,
        md: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> dict[str, Any]: ...

    def augment_station_frame(
        self,
        stations: pd.DataFrame,
        lateral: pd.DataFrame,
        typewell: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Add optional proprietary columns; public kernel returns unchanged."""
        ...


def resolve_kernel() -> GeometryKernel:
    mode = (os.environ.get("RWT_KERNEL") or "public").strip().lower()
    if mode in ("private", "closed", "proprietary"):
        try:
            # private_kernel package living beside repo or on PYTHONPATH
            import private_kernel  # type: ignore

            k = private_kernel.get_kernel()
            if not k.is_private():
                raise RuntimeError("private_kernel.get_kernel must report is_private=True")
            return k
        except Exception as e:
            # fail closed to public — never crash competition path
            from rwt.kernel_public import PublicKernel

            pk = PublicKernel()
            pk._load_note = f"private_kernel unavailable: {e}"  # type: ignore[attr-defined]
            return pk
    from rwt.kernel_public import PublicKernel

    return PublicKernel()
