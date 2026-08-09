# private_kernel — PROPRIETARY (do not publish)

This directory is **gitignored** in the public tree (`private_kernel/*` except this README).
Keep a local (or private-repo) copy of advanced implementations here.

## Expected layout (local only)

```text
private_kernel/
  __init__.py              # get_kernel() → PrivateLatticeKernel
  kernel_impl.py           # GeometryKernel: inject prior_lattice_tvt at seal
  lattice_solver.py        # Viterbi TVT lattice (emission + transition DP)
  beam_match.py            # beam_prior_for_well / beam_blend_last (pilot API)
  holonomy_advanced.py     # optional richer transport
  sheaf_features.py        # optional sheaf extras
  residual_map.py          # optional proprietary residual maps
```

## Contract

Implement `rwt.kernel_api.GeometryKernel` (see `src/rwt/kernel_api.py`):

- `trajectory_features(md, x, y, z) -> dict`
- `augment_station_frame(stations, lateral, typewell) -> DataFrame`
- `kernel_id() -> str`
- `is_private() -> True`

Load with:

```bash
set RWT_KERNEL=private
# repo root on PYTHONPATH so private_kernel imports
python scripts/measure_private_lattice_pilot.py --data data/raw --max-wells 40
# or legacy beam pilot name:
python scripts/measure_private_beam_pilot.py --max-wells 40
```

Public CI / clones without this package **fail closed** to `PublicKernel`.

## Lattice engine (v1 scaffold)

Discrete **banded** Viterbi over typewell TVT states along MD:

- **Emission:** window GR NCC vs typewell at candidate TVT
- **Transition:** soft MD-slope penalty (public ±0.008 law) + last±18 clamp; **banded** neighbors (O(N·S·B), not full S²)
- **Heel pin:** known TVT_input/TVT locked
- **Injected columns:** `prior_lattice_tvt`, `lattice_conf`, `prior_lattice_blend` (conf-blend ↔ last), soft-pull of `prior_best` toward **blend** (conf-weighted ≤0.35). Never overwrites `prior_stack`.

**Adaptive is the game:** pure lattice mean can lose to last_known; conf-blend and soft-pull are the MEASURE axes. Residual may use `--prior-col prior_lattice_blend` or default `prior_best` (post-pull) only if Gate3 OOF beats public floors — never force-OPEN.

**Exp D:** Gate3 floors include `prior_lattice` / `prior_lattice_blend` when private kernel is live.

**Conf calibration:** `lattice_conf` from path emission NCC (not collapsed Viterbi margins).  
`lattice_conf_eff = conf × exp(−|lat−stack|/10)` so conf-blend / soft-pull refuse lattice when it fights public stack.

**dA stalk (v2):** identity `TVT ≈ A−Z+c` ⇒ integrate `dA−dZ`.  
`da_stalk.da_prior_for_well` → `prior_da_tvt`, heel/pad dA prior, **bounded** GR refine only.  
Soft-pull `prior_best` toward dA (preferred) or lattice blend; never overwrite `prior_stack`.  
Gate3 floor key: `prior_da` → column `prior_da_tvt`.

## Law

- Never push proprietary sources to the public GitHub remote.
- Competition: export only predictions via `scripts/export_competition_bundle.py`.
- RESTRICT: measure GroupKFold OOF + Gate3; no Kaggle submit from measure drivers.
