# Manifold overlay — integrate, fully dimensionalize, proceed

**Law:** `RESTRICT → MEASURE → AUDIT → CERT`. Residue never forced. No Kaggle submit from measure drivers.  
**Hunt:** GroupKFold selected OOF TVT RMSE **&lt; 5.0 ft**.  
**Status:** NEED_IMPROVE (public selected ~14.7–15.2; private conf_gate pilot ~10.7).

This note **integrates** the competition geometry with a single construct language,  
**fully dimensionalizes** every atomic station observation, and states what  
**proceed** means under max-op / Gate3 (not force-OPEN).

---

## 1. Integrate (conformed construct)

Three “sources” join on **station keys** `(well_id, station, MD)` via shared dimensions:

| Source | Manifold role | Conformed keys / fields |
| --- | --- | --- |
| **Typewell** | Formation / signature manifold | `TVT` axis, `GR(TVT)` interpolated grid |
| **Lateral (test)** | Open test manifold (SE(3) path + sensors) | `MD`, `X,Y,Z`, `GR`, heel `TVT_input` |
| **Neighbors / lease** | Spatial overlay chart | `neighbor_tvt`, `transfer_ok`, offsets |

**Registration (index):** map each lateral station → coordinate on the typewell TVT grid  
(NCC, PF, beam, lattice). That is not “predict TVT from a bag of columns” first —  
it is **indexing the test manifold against the interpolated formation grid**.

**Multi-dim overlay:** geometry, GR match, neighbor, and confidence are fields on the  
same stations; residual may only correct **ΔTVT** after pure floors are honest (Gate3).

```
typewell grid  GR(TVT)  ──index──►  TVT*(MD) on lateral
lateral path   SE(3), GR, heel TVT  ──overlay──►  priors + conf
neighbors      chart transfer       ──conform──►  neighbor_* features
residual       ΔTVT                 ──only if──►  OOF ≤ all pure floors
```

---

## 2. Fully dimensionalize (atomic station fact)

Every low-level observation is one **station fact**. Dimensions attach **before**  
residual learning (Kimball sense: dimensionalize at extract / assemble).

### 2.1 Continuous dimensions (continua — not categories)

| Dimension | Continuum | Code / prior |
| --- | --- | --- |
| **TVT** | formation index (ft) | target; clamp last±18 |
| **MD** | along-hole (ft) | `relative_md` |
| **GR level / shape** | API / ΔGR | rolls, NCC, PF likelihood |
| **SE(3) strain** | κ, τ, e_ref, DLS | `geometry`, dip-frame |
| **Dip / structure** | heel plane, local dip | `prior_dip_*`, `prior_stack` |
| **Neighbor chart** | distance, ok, offset | `spatial` attach |
| **Stalk conf** | [0,1] trust of index | `pf_confidence`, `ncc_track_conf`, private conf |
| **Disagreement** | stalk spread (ft) | bag std, lattice margin |

### 2.2 Discrete / identity dimensions

| Dimension | Values | Role |
| --- | --- | --- |
| `well_id` | group key | GroupKFold — no station leakage |
| `is_known` / `is_predict` | heel vs toe | middle-out mask |
| `kernel_id` | public / private | IP fence |
| `prior_name` | last, stack, ncc, beam, conf_gate, … | pure floor table |

### 2.3 What “fully” does *not* mean here

- Not closed-curve holonomy forced to zero (wells are open).  
- Not peptide Ramachandran / AF yield (LINEAGE_TSS non-ports).  
- Not form-surface leak columns (ANCC…BUDA).  
- Not NPU score authority without a Gate3-green residual seal.

Saturation (grounded-theory sense): keep adding **properties of the index/overlay**  
until pure-floor OOF stops moving (~0.05 ft) — then CERT plateau or new lever class.

---

## 3. Proceed (exit criteria)

Aligned with max-op **Proceed** and construction-style “proceed ≠ certified accuracy”:

| Gate | Meaning |
| --- | --- |
| **Public path sealed** | GroupKFold + Gate3 + provenance on public kernel; residual optional |
| **Proceed to private** | Crown gap → `private_kernel` beam/DTW/lattice (gitignored offline) |
| **Proceed residual** | Only if pure floor moves ≥ ~0.3 ft same cohort **and** Gate3 ACCEPT |
| **Proceed CERT sub-5** | `oof_rmse_selected < 5.0` + `measure_vs_5ft.json` + provenance |
| **Do not proceed** | Force-OPEN residual, denser lattice without measure win, Kaggle under RESTRICT |

“Proceed” authorizes the **next experiment class**, not a claim that finite TVT error is zero.

---

## 4. Current seals (dimensional readout)

| Cohort | Selected / deploy | Best pure | Gate3 | beats_5ft |
| --- | ---: | ---: | --- | --- |
| 773w public (old) | ~14.67 | stack ~15.11 | ACCEPT residual | no |
| 773w stack-base residual | ~15.20 | stack ~15.76 | ACCEPT (no-promote vs 14.67) | no |
| 64w stack residual | **12.27** | stack 12.40 | ACCEPT | no |
| 128w stack residual | **13.72** | stack 14.47 | ACCEPT | no |
| Private conf_gate ~40w | **~10.70** | last ~10.79 | pilot only | no |

Residual shaves decimals; **index/overlay** owns the multi-ft gap to crown/hunt.

---

## 5. Proceed order (implementation)

1. **Dimensionalize emission** on private grid: likelihood = GR + geometry + conf (joint), not GR-only.  
   - **v9 landed:** `lattice_prior_for_well(..., multi_field=True)` — GR level + dGR/dMD shape vs typewell dGR/dTVT·process + e_ref stickiness (`private_kernel/lattice_match.py`).  
   - **v10 landed:** GR **heatmap field** — graduate typewell onto regular TVT chart (GR / dGR / d²GR channels); index lateral GR strip seed-locally (`private_kernel/gr_heatmap.py`). The field is the layered box; the lateral is the fork.
2. **Integrate** private deploy prior as optional residual feature/floor when pure improves ≥0.3 ft.  
3. **MEASURE** pilot (≤64w / ≥40w private) then promote full GKF only on real pure move.  
4. **AUDIT / CERT** Gate3 + `beats_5ft`; STOP on 3× plateau or sub-5 seal.  
5. Public residual retune alone is **not** the proceed path.

---

## 6. Operator one-liner

> **Fully dimensionalize** each station on the formation×test×neighbor chart;  
> **integrate** via registration to the typewell grid;  
> **proceed** only to private stalk / Gate3 residual / CERT under measure law — residue never forced.
