# private_kernel — PROPRIETARY (do not publish)

This directory is **gitignored** in the public tree. Keep a local (or private-repo) copy of advanced implementations here.

## Expected layout (local only)

```text
private_kernel/
  __init__.py          # MUST expose get_kernel() → kernel with truthy is_private()
  kernel_impl.py       # PrivateKernel: beam + sheaf station columns
  beam_match.py        # beam stalk v2 knobs + v6 heel GR affine / adaptive sigma
  lattice_match.py     # band-lattice Viterbi + conf_gate (measure pins sealed knobs)
  sheaf_features.py    # path Laplacian / energy features
  residual_map.py      # optional proprietary residual transforms
```

### Loader contract (`src/rwt/kernel_api.resolve_kernel`)

When `RWT_KERNEL=private`, the public resolver does:

1. `import private_kernel`
2. `k = private_kernel.get_kernel()`  — **required entry point** (not just KernelImpl class)
3. require truthy `k.is_private()` (`if not k.is_private():` → fail-closed to `PublicKernel`)

If import fails or the contract is violated, the public path continues with `PublicKernel` (fail-closed). Implement `get_kernel()` in `__init__.py`.

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

Sealed measure knobs (explicit in `scripts/measure_private_beam_pilot.py`):  
`margin=0.08`, `lat_floor=0.18`, `agree_ft=2.0`, `soft_temp=0.05`, `pure_lat_boost=0`, `n_nodes=97`.  
Rejected on 40w: denser `n_nodes=129` (pure lat 10.801); beam-led margin0.10 (conf_gate 10.710);
GR emission gate (emit 10.738 vs conf_gate 10.706).

Public residual path (when `RWT_KERNEL=private`): `augment_station_frame` emits
`prior_beam_tvt/conf/blend`; `rwt.features` adds deltas and residual X includes them.
16w pilot: residual lost pure stack (FALLBACK); 8w smoke ACCEPT was cohort-specific.

Target hunt: conf_gate ≤ 9.0 then Gate3 residual / full GKF promote if pure moves ≥0.3 ft.

## Contract

Implement `rwt.kernel_api.GeometryKernel` (see `src/rwt/kernel_api.py`):

- `get_kernel() -> GeometryKernel`  # package entry; resolver calls this
- `trajectory_features(md, x, y, z) -> dict`
- `station_features(lateral_df, typewell_df, split) -> pd.DataFrame`  # optional columns
- `kernel_id() -> str`
- `is_private() -> bool`  # must be truthy or resolver fails closed

Load with:

```bash
set RWT_KERNEL=private
# package on PYTHONPATH or install private wheel
```

## Law

- Never push this tree to the public GitHub remote.  
- Competition: export only predictions via `scripts/export_competition_bundle.py`.  
