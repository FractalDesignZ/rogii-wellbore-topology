# private_kernel — PROPRIETARY (do not publish)

This directory is **gitignored** in the public tree. Keep a local (or private-repo) copy of advanced implementations here.

## Expected layout (local only)

```text
private_kernel/
  __init__.py          # exposes KernelImpl
  kernel_impl.py       # PrivateKernel: beam + sheaf station columns
  beam_match.py        # beam stalk v2 knobs + v6 heel GR affine / adaptive sigma
  lattice_match.py     # band-lattice Viterbi + soft conf-gate (v6 lat_floor)
  sheaf_features.py    # path Laplacian / energy features
  residual_map.py      # optional proprietary residual transforms
```

## Pilot seals (40w comparable cohort)

| Prior | mean RMSE (ft) | Note |
| --- | ---: | --- |
| last_known | 10.788 | floor |
| beam_blend v2 | 10.756 | pre-affine seal; reverse_pass/dense reseed **rejected** (→11.08) |
| conf_gate hard v3 | 10.749 | lattice only when conf > beam + 0.08 |
| conf_gate soft v4.1 | 10.747 | soft weight + agree fuse; micro-beat only |
| beam_blend v6 | **10.729** | heel affine + adaptive gr_sigma |
| pure lattice v6 | **10.686** | best raw stalk post-affine (not deploy alone) |
| **conf_gate v6 (deploy)** | **10.705** | −0.083 vs last; −0.042 vs soft v4.1 |
| oracle stalk | ~10.59 | label leak; diag only |

Target hunt: conf_gate ≤ 9.0 then Gate3 residual / full GKF promote if pure moves ≥0.3 ft.

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
