# Lineage: topology-sees-sequence → wellbore late entry

Source: https://github.com/ZuluYokohama/topology-sees-sequence  

## Portable operators (used)

| TSS module | Operator | Wellbore adaptation |
|------------|----------|---------------------|
| `holonomy_extract._rot_between` | Minimal SO(3) map u→v | Discrete frame transport along open MD polyline |
| `holonomy_extract.bishop_holonomy` | Closed-curve Bishop angle | **Open** path: cumulative twist + curvature series (no false closure) |
| `middle_out_v2` | Slice 0 = E_ref prior | Slice 0 = known section prior; toe = residual predict zone |
| `NOTES_EREF_PRIOR` | Prior, not forced energy | Geometric prior on residual TVT; trees own the score |
| `derive.se3_log` / screw | SE(3) residual | Optional later: local SE(3) between survey stations |

## Explicit non-ports

| TSS claim | Why not port |
|-----------|----------------|
| Ramachandran basins / Pro/Gly classes | Not a GR alphabet |
| Rosetta total score P2 | Invalid even in TSS; irrelevant here |
| Closed holonomy = 0 by definition | Wells are **open** curves |
| AF structure yield | Different domain |

## Operating call

Same as TSS post-correction: **geometry first as prior and features**, do not force a topological score as the competition objective. Leaderboard is **TVT RMSE**.
