# ALL CONTEXT — operator construct (two-step locked)

**RESTRICT:** GroupKFold-by-well OOF + Gate3 + seals. No Kaggle submit.  
**Data root:** `data/raw/train` ≡ Documents `rogii-wellbore-geology-prediction-original/train` (~773 laterals + typewells).

This is the single page that merges every thread of merit: MWD survey math, MD-as-index, XYZ:GR cloud, GR@TVT book, pad manifold, adaptive residual, Wave A, synthetic falsifiers vs real matrix.

---

## 0. The two-step (say it out loud)

```text
STEP 1 — GEOMETRY
  Surveys + surface depth  →  ordered path in space (X,Y,Z)
  MD is only the path parameter / merge key (not a free physical axis)

STEP 2 — FORMATION
  LWD paints GR on that path
  Typewell is the book G_tw : TVT → GR
  Geosteering places the path on the book → TVT
  Residual polish only if Gate3 + trust (conf / GR-agree)
```

**Score** lives almost entirely in **Step 2 placement (TVT on toe)**.  
**Step 1** is given (and mostly trusted) as columns X,Y,Z,MD.

---

## 1. Industry stack (MWD/LWD) → our columns

| Field practice | What it produces | ROGII column |
| --- | --- | --- |
| Accel / mag survey | Inc, az, toolface | (consumed upstream; we get XYZ) |
| Survey calc (trig / min-curve) | Trajectory points | **X, Y, Z** |
| Surface depth tracking | Bit/hole depth | **MD** (index / arc proxy) |
| Clock-merge LWD memory | Depth-referenced logs | **GR** on path (gappy) |
| Geosteering / correlation | Strat position | **TVT**, **TVT_input** (heel) |
| Typewell / offset GR | Formation fingerprint | typewell **TVT, GR** |
| Named tops (train study) | Marker surfaces | **ANCC…BUDA** (train only) |

MWD law that matches the insight: **depth is not measured in the FE tool**; path is geometry; FE is sensor-on-path; MD glues time-based memory to the path.

### 1.1 deep-research-11 seal (Lesson-1 MWD/LWD — Partial)

Sources: OIL Direction & Inclination, LWD Ops, Formation Evaluation, Nuclear Theory (GR strand only) + contest task/CSVs. Full report in session workflow scratch.

| Lesson-1 concept | Contest mapping | Our status |
| --- | --- | --- |
| MD / Inc / Az → min-curve XYZ (TVD, HD, VS) | Columns already give **MD, X, Y, Z** | Step 1 given; **do not** rebuild from raw toolface/accels/mags |
| Dogleg severity | Path features from XYZ segments | Live: `inclination_deg`, `curvature_dls` |
| Natural GR = lithology fingerprint | Only FE channel in matrix | Step 2 observation; book inverse |
| GR multi-scan → memory; ≥2 pts/ft quality | Gappy GR, NaNs OK | Wave A nulls + gap tags |
| Log depth = surface time–depth merge | MD merges GR to path | MD is index, not free physics axis |
| Resistivity / mud-pulse / density–neutron / HSE | No channels; not in RMSE(dTVT) | **STOP** as features |
| TVD in D&I texts | **≠** contest **TVT** (strat thickness book) | Do not equate TVD≡TVT |

**OPEN (re-confirmed):** geometry from XYZ; GR@TVT book inverse; null hygiene; DLS from path.  
**STOP (re-confirmed):** raw survey reconstruction; resistivity/telemetry/nuclear-env features; tops in residual X; treating Lesson-1 “correlation” (depth-time) as typewell placement (that job is ours: GR-match / PF / lattice).

---

## 2. The real matrix (not synthetic)

Each horizontal row = one point on the walk:

```text
(X, Y, Z) : GR          measured cloud
MD                      path parameter (obscuring if treated as independent)
TVT_input               known page on heel only
TVT                     full page numbers on TRAIN; withheld on TEST toe
ANCC…BUDA               train-only surfaces — NEVER residual features
```

Typewell (separate file per well):

```text
G_tw(TVT) = GR_book     the formation book
```

**Multidimensional slices** = many well paths through one play.  
**Not** a filled voxel cube. Sparse lines only.

Train/test asymmetry (the only real hardness):

| Channel | Train | Test |
| --- | --- | --- |
| MD, X, Y, Z, GR | yes | yes |
| Heel TVT_input | yes | yes |
| Toe TVT | **yes** | **no** |
| Tops | **yes** | **no** |
| Typewell | yes | yes |

---

## 3. Math object (clean)

**Measured path cloud**

\[
\mathcal{C} = \big\{\big(X_s, Y_s, Z_s, \mathrm{GR}_s\big)\big\}_{s\in\mathrm{path}}
\quad\text{(ordered by MD / arc-length)}
\]

**Formation book**

\[
G_{\mathrm{tw}} : \mathrm{TVT} \mapsto \mathrm{GR}
\]

**Placement (inverse)**

\[
\widehat{\mathrm{TVT}}_s
  = \arg\min_{t(\cdot)}
  \rho\big(\mathrm{GR}_s,\, G_{\mathrm{tw}}(t_s)\big)
  + \mathrm{Reg}_{\mathrm{path}}
  + \mathrm{Reg}_{\mathrm{heel}}
  + \mathrm{Reg}_{\mathrm{pad}}
\]

**Serve**

\[
\widehat{\mathrm{TVT}}_s
  = \mathrm{prior}_M(s)
  + \mathbf{1}_{\mathrm{fire}}(s)\,\Delta_{\mathrm{res}}(s)
\]

Clamp \(|\Delta|\le 18\). Gate3: residual only if OOF beats pure floors.  
Adaptive: fire only if conf (GR-agree / long-lived) earns it.

---

## 4. Process map (every module has a home)

```text
STEP 1  GEOMETRY
  load XYZ, MD
  build path metrics (arc, dls, e_ref, relative MD/Z)
  null-decode GR (-999.25 → NaN)
        │
        ▼
STEP 2a BOOK + INVERSE
  typewell G_tw
  PF / NCC / private lattice → stalk TVT prior
  GR mismatch @ hypothesized t
        │
STEP 2b PAD
  neighbor IDW / sheaf-lite TVT transfer (XY)
        │
STEP 2c STRUCTURE
  flat C−Z + heel dip plane
        │
        ▼
FUSE  prior_stack / prior_best (±18)
        │
GATE3 + ADAPTIVE
  residual LGBM Δ only if floors beaten and fire mask on
        │
        ▼
  TVT_hat on toe → RMSE
```

| WP | Name | Status |
| --- | --- | --- |
| 0 | Hygiene / nulls / no tops | Wave A live |
| 1 | Book GR@TVT | live |
| 2 | Stalk inverse | PF/NCC live; lattice private |
| 3 | Pad transfer | multi-neighbor live |
| 4 | Dip / flat | live |
| 5 | Fuse priors | live |
| 6 | Residual | Wave A 12w **10.741** |
| 7 | Adaptive thr×life | live; **exp C: conf = pf × GR-agree** |
| 8 | Pad synth falsifier | live (`measure_manifold_gr_synth`) |
| — | Real field slices | **773w** `artifacts/field_slices_full` |

---

## 5. Empirical anchors (do not gaslight)

| Seal | Result |
| --- | ---: |
| Wave A 12w residual OOF | **10.741** (vs set-19 11.045) |
| Adaptive serve 12w | **False** (pf 11.10; exp C product 11.98; none ≤ residual 10.74) |
| Best pure floor 12w | prior_best ~12.64 |
| 773w residual (prior seal) | ~15.4 |
| Crown / hunt | **6–7 / 5** |
| Private lattice pure mean | loses last_known; conf-blend / well pick only |
| WP8 synth book_ok | GR MAE@true < @last |
| WP8 pad transfer | neighbor ≤ last on test mean |

**Read:** residual polish helped a bit. Crown gap = **better Step 2 placement on real paths**, not deeper trees.

---

## 6. Ontology (point cloud / time / render)

| Metaphor | Engineering |
| --- | --- |
| Consciousness / process time | MD order / arc along the hole |
| Point cloud | (X,Y,Z,GR) samples |
| Render | recover TVT (and pad field) from cloud + book |
| Book / world model | typewell \(G_{\mathrm{tw}}\) |
| Trust | conf / GR corroboration / adaptive |
| Hallucination control | RESTRICT, Gate3, null law, no tops |

**MD obscures** when treated as an independent feature axis.  
**Un-obscured:** geometry path + GR paint + place on book.

---

## 7. Visuals (real + construct)

| Folder | What |
| --- | --- |
| `artifacts/field_slices_full/` | **773-well real field slices** A–H + dashboard |
| `artifacts/field_slices/` | pilot slice set |
| `artifacts/geometry_gr_visuals_real_train/` | Real train: path first, GR heatmap |
| `artifacts/geometry_gr_visuals/` | Synth pad (lab only) |
| `artifacts/manifold_gr_synth/visuals/` | GR@TVT book + pad field |
| `artifacts/wave_a_12w/` | Honest residual seal |

Full-pad slice (MEASURE only — not a score claim):

```powershell
python scripts/slice_real_field.py --data $env:USERPROFILE\Documents\rogii-wellbore-geology-prediction-original\train --max-wells 0 --out artifacts/field_slices_full
```

Re-render path/GR heatmaps:

```powershell
python scripts/render_geometry_gr_heatmap.py --stations artifacts/geometry_gr_visuals_real_train/stations_preview.csv --out artifacts/geometry_gr_visuals_real_train
```

---

## 8. OPEN | STOP (all threads)

**OPEN**
- Treat matrix as \((X,Y,Z):\mathrm{GR}\) path cloud + typewell book
- Real train for all serious MEASURE / visuals
- Gate3 residual when OOF improves (Wave A held)
- Stalk / pad / GR-corroboration conf as crown path (exp C: conf live; adaptive still loses 12w)
- LAS null → NaN; short-gap only with tags
- Lesson-1 hygiene + path kinematics as already encoded

**STOP**
- MD as primary physics (use as order/arc only)
- Tops as residual features
- Synthetic replacing real field work
- Dual residual / soft-nb-into-stack / force pure lattice mean
- Kaggle submit under RESTRICT
- Claiming crown without better placement OOF
- Raw MWD sensor rebuild; resistivity/telemetry/RHOB–NPHI features
- Equating industry TVD with contest TVT

---

## 9. One block (tattoo)

```text
STEP 1  Build the hole in space: (X,Y,Z) ordered by MD
STEP 2  Paint GR on the hole; place the hole on G_tw(TVT)
        Heel pins the page; pad transfers pages; residual edits only if honest
```

**That’s the entire contest in two steps.**  
Everything else is implementation detail under RESTRICT.

---

## Related

- [`TASK_GR_TVT_CONSTRUCT.md`](TASK_GR_TVT_CONSTRUCT.md) — WP decompose  
- [`MANIFOLD_GR_CORROBORATION.md`](MANIFOLD_GR_CORROBORATION.md) — pad falsifier  
- [`SCORE_DROP_SYNTHESIS.md`](SCORE_DROP_SYNTHESIS.md) — PAC/SRM/TDA  
- [`CROWN_PATH.md`](CROWN_PATH.md) — cumulative axes + rejects  
