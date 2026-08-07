# private_kernel — PROPRIETARY (do not publish)

This directory is **gitignored** in the public tree. Keep a local (or private-repo) copy of advanced implementations here.

## Expected layout (local only)

```text
private_kernel/
  __init__.py          # exposes KernelImpl
  holonomy_advanced.py # optional richer transport / invariants
  sheaf_features.py    # optional λ₁ / coboundary style features
  residual_map.py      # optional proprietary residual transforms
```

## Contract

Implement `rwt.kernel_api.GeometryKernel` (see `src/rwt/kernel_api.py`):

- `trajectory_features(md, x, y, z) -> dict`
- `station_features(lateral_df, typewell_df, split) -> pd.DataFrame`  # optional columns
- `kernel_id() -> str`
- `is_private() -> True`

Load with:

```bash
set RWT_KERNEL=private
# package on PYTHONPATH or install private wheel
```

## Law

- Never push this tree to the public GitHub remote.  
- Competition: export only predictions via `scripts/export_competition_bundle.py`.  
