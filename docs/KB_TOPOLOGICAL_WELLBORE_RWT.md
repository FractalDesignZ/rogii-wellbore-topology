# Knowledge Base: Topological Wellbore Navigation & `rwt` Pipeline

Noise-free synthesis of lineage docs, architectural notes, and measured cohort results.
Optimized for NotebookLM / RAG ingestion. Claims below match **live open-core `src/rwt`**
as of full-cohort train seal (773 wells).

**Public clamp law (empirical):** no prior may run unbounded off last-known; residual ships
only if OOF beats the **best pure floor** (stack / blend / last_known), not merely its residual base.

---

## PART 1: Theoretical Abstractions & Epistemology

### Executive Synthesis: Volume II (The Topological Toolkit)

Volume II establishes the epistemological and verification machinery of the framework—answering
*how we know* and *how we verify* structural integrity across complex, non-stationary systems.
It bridges the foundational ontology of Volume I with the empirical methods implemented across
the software stack (`JTopo`, `rwt`, and `SHEAF-STACK`).

**Axiomatic Codebase Mapping Matrix**

| Axiom Block | Mathematical Primitive | Codebase Implementation |
| --- | --- | --- |
| **4.1: Topological Invariance** | Betti numbers ($\beta_0, \beta_1$), Homology | `JTopo` & `scripts/` |
| **4.2: Filtration** | Vietoris-Rips & k-NN filtrations | `scripts/render_tda_network_graphics.py` |
| **4.3: Persistence** | Birth/Death diagrams, barcodes | Single-linkage $H_0$ Kruskal merges on well centroids |
| **4.4: Mapper** | Nerve graph & cluster summaries | Spatial wellbore networks |
| **5.1: Reverse Engineering** | Bottom-up structural recovery | `src/rwt/abstraction.py` |
| **5.2: Term Series (CTS)** | $X \to \mathcal{M} \to \mathcal{M}_v \to H_k \to Q$ | 7-Mode Pipeline ($M0 \to M6$) |
| **5.3: Quale Equivalence** | $Q(\Phi) \equiv$ Persistence Diagram | Integrated Information ($\Phi$) framing |
| **6.1: Topological Verification** | $d(T, T_0) < \varepsilon$ (Wasserstein) | Cryptographic SHA-256 seals (`provenance`) |
| **6.2: Continuity Guards** | Open-set preservation & residual gate | `src/rwt/gates.py` ($G1, G2, G3$) |

**Key Theoretical Contributions in Volume II:**

1. **Epistemological Grounding of TDA (Block 4):** Justifies filtering high-dimensional sensor noise.
   Long-lived features (high persistence) are treated as structural signal; short-lived features as topological noise.
2. **The Canon of Emergence (Block 5):** Axiom 5.2 formalizes a multi-stage configurational transform
   from raw substrate facts ($X$) toward irreducible semantic meaning ($Q$). Axiom 5.3 frames semantic
   meaning as a robust topological signature (persistence / section-like invariants).
3. **Verification & Identity Conservation (Block 6):** Replaces heuristic “looks better” checks with
   sealed certificates and operational gates. **Gate 3 (Residual Acceptance)** rejects a residual head
   if OOF RMSE fails the best pure prior floor and/or last-known floor.

**Bridge to Volume III (Signal Processing):**
Extends machinery toward multi-channel time series and cross-domain signal mapping by projecting
signals into a universal factor space (pressure / time / modality / frequency framing in lineage docs).

### Academic Foundation: Ring Closure as Holonomy

*(Cyclic Peptide Backbones as Flux-Threaded Sheaves — lineage isomorphism, not a ROGII deliverable)*

* **Closability criterion:** Strain-free closure when the backbone frame returns to itself; otherwise not for every conformation simultaneously.
* **Flux correspondence:** Backbone sheaf Laplacian unitarily related to a ring tight-binding Hamiltonian with flux $\pm\theta$.
* **Sequence sets cost, not feasibility:** Residue stiffness rescales the spectrum; Betti numbers fixed—topology is blind to sequence; sequence sets energy *scale*.

---

## PART 2: Translation to Directional Drilling

Both systems are continuous 3D paths. A cyclic peptide is a microscopic closed molecular chain;
a drill string / wellbore is a macroscopic kinematic chain. Both live in $SE(3)$.

**Isomorphic Core Claims**

* **Reference-connection strain $\rightarrow$ BHA joint / pipe strain:** residual $SE(3)$ screw relative to relaxed state $\leftrightarrow$ tortuosity, drag, mechanical stress.
* **Bishop holonomy $\rightarrow$ wellbore geometric walk:** parallel transport of the survey curve measures cumulative toolface walk without coordinate distortion.
* **$\psi \approx \varphi$ closure locus $\rightarrow$ neutral bit-tilt equilibrium:** pure geometric regimes with zero net DLS change.

**Mathematical Operator Isomorphism Chart**

| Academic Notation | Technical Operator | Rig-Floor Vernacular |
| --- | --- | --- |
| $\mathbf{g} \in SE(3)$ | Special Euclidean Group 3D | 3D position and orientation of the bit / BHA |
| $\nabla_T \mathbf{X} = 0$ | Parallel transport | Toolface steady relative to the hole curve |
| $\oint_{\gamma} \mathbf{\Omega}$ | Holonomy / monodromy | Total toolface walk after a run |
| $\kappa = \Vert \gamma'' \Vert$ | Curve curvature | Dogleg severity (DLS), deg/100 ft |
| $\tau$ | Curve torsion | Spiraling / micro-tortuosity |
| $\mathcal{F} \xrightarrow{\pi} \mathcal{M}$ | Fiber bundle / sheaf framing | BHA / formation coupling across depth |
| $\mathbf{T} = \mathbb{C} : \mathbf{E}$ | Hooke-type response | Stretch, twist, buckling limits under load |
| $\nabla_{\dot{\gamma}} \mathbf{v}$ | Covariant acceleration framing | Bit walk / drop across lithology contrasts |

---

## PART 3: The `rwt` System Architecture (ROGII Wellbore Geology)

**Task:** Predict $TVT$ (True Vertical Thickness) on the masked evaluation zone (**toe**) of horizontal wells
(1-ft station targets, RMSE metric).

### The Three Fatal ML Traps (and Solutions)

1. **Structural leakage (`ANCC`…`BUDA` mirage):** Formation tops are train-only. They must never enter residual $X$.
   Live: dropped from `FEATURE_COLS_DEFAULT`; `feature_matrix` **raises** if re-injected.
2. **Target overfit (global $Z$ / absolute $TVT$):** Absolute TVT memorizes maps and fails local structure.
   **Fix:** geometric / stalk prior $TVT_{\mathrm{prior}}$, residual $\Delta TVT = TVT - TVT_{\mathrm{prior}}$.
3. **Validation leakage ($M0$ RESTRICT):** Station-level KFold leaks adjacent 1-ft structure.
   **Fix:** **GroupKFold by `well_id` only.**

### The 7-Mode Abstraction Pipeline (M0–M6)

| Mode | Name | Role in Wellbore Prediction | Lineage / Primitive |
| --- | --- | --- | --- |
| **M0** | RESTRICT | TVT RMSE; GroupKFold-by-well; no station leak | RESTRICT law |
| **M1** | GEOM | Tangents, curvature, torsion, $E_{\mathrm{ref}}$, dip frame | Open-curve / $SE(3)$ |
| **M2** | PRIOR | Heel-locked priors (last-known, flat, local MD, dip, PF, stack) | Slice-0 / middle-out |
| **M3** | MATCH | Multi-scale GR rolling stats + sparse NCC to typewell | NCC / GR stalk |
| **M4** | MIDDLE_OUT | Known heel (`TVT_input`) vs masked toe | `middle_out_v2` |
| **M5** | RESIDUAL | LightGBM residual on pure prior | $Y - \mathrm{prior}$ |
| **M6** | CERTIFY | Baseline beat-table, Gate3, SHA-256 seal | Provenance; open-core |

### The Three-Layer Geosteering Stack

$$\mathrm{TVT}(s) = \mathrm{Prior}_{\mathrm{flat}}(s) + \mathrm{Trend}_{\mathrm{dip}}(s) + \mathrm{Residual}_{\mathrm{lithology}}(s)$$

1. **Layer 1 (Flat prior):** Heel constant $C = \mathrm{mean}(TVT_{\mathrm{input}} + Z)$ on known heel only;
   flat field $C - Z$ (neutralizes bulk depth coupling).
2. **Layer 2 (Dip plane):** Ridge($X,Y$) on **near-PS lateral heel only** (not full build), strong
   regularization, MD-decay of dip toward toe. On real laterals this alone often **loses** to last-known
   unless blended/clamped—do not treat as the crown prior by itself.
3. **Layer 3 (Residual):** **LightGBM** residual on toe labels only; features = relative kinematics,
   GR rolls / NCC, PF confidence, neighbor transfer—not absolute $X,Y$ lease map, not form tops.

**Default residual base (live):** `prior_stack` (clamped multi-source blend).  
**Serve residual only if Gate 3 accepts.**

### Local / PS prior (blueprint M2 — corrected law)

Horizontal laterals make raw $\Delta TVT / \Delta Z$ over heel **ill-conditioned** (build section projected
into toe → 50–150 ft disasters). Live `prior_local_dip`:

* Anchor at last-known PS TVT.
* Soft **MD slope only**, clipped to $\lvert a \rvert \le 0.008$ (≤ 8 ft TVT per 1000 ft MD).
* **No Z-dip term** on laterals.
* Heel stations locked to observed `TVT_input` / `TVT`.

### Functional Engines

* **Kinematics engine:** $SE(3)$-adjacent invariants (curvature, torsion, $E_{\mathrm{ref}}$, relative MD/Z,
  inclination, 3D dogleg proxy) to expose physically hard doglegs.
* **Spatial sheaf-lite engine:** k-NN proximity on lateral XY centroids ($d \lesssim 600$ ft), azimuth gate,
  nearest-XY TVT transfer from neighbors (public; private sheaf Laplacians stay offline).
* **Particle-filter (PF) tracker:** Heel-locked particles match lateral GR to typewell GR along TVT;
  multi-config bag + confidence blend with last-known; **hard clamp** $\lvert \Delta \rvert \le 18$ ft vs last-known.
* **Stack blend:** last-known floor + soft dip nudge + PF only where confident; same clamp law.

### Gates (operational continuity)

| Gate | Name | Contract |
| --- | --- | --- |
| **G1** | IP_FENCE | Public vs private kernel; fail-closed to public |
| **G2** | MIDDLE_OUT_MASK | Non-empty heel/toe when `TVT_input` masks; recorded per well |
| **G3** | RESIDUAL_ACCEPTANCE | Residual OOF ≤ residual base **and** ≤ last-known **and** ≤ **best pure floor** among scored priors (stack, blend, bag, local-dip, etc.) |

On G3 reject: serve pure prior with **minimum OOF** among floors (never ship residual that loses the floor).

### IP Boundary & Governance

* **Open-core** `src/rwt/` — AGPLv3: features, priors, PF-lite, train, submit, gates, seals.
* **Private** `private_kernel/*` — heavier sheaf / stalk machinery; gitignored; not required for public train/submit.
* Bundle exporter / seals keep private paths out of competition digests.

### Ops (compute)

* Multi-process **disk-backed** well assemble (`RWT_WORKERS`); no pickling full `SpatialIndex` through pipes.
* BLAS threads forced to 1 per worker under multi-process; LightGBM `num_threads` for residual folds.
* Live progress: `progress.json` / `progress.log`; dashboard bar-only poll + pop-out details
  (`scripts/dashboard_server.py`, `.venv-dash`).
* **Honest hardware:** public stack is **CPU** (NumPy/OpenBLAS NEON or AVX + LightGBM). No open-core GPU/NPU path.

---

## PART 4: Project Progression & Empirical Findings

### 1. In-sample vs out-of-sample

Heel reconstruction (labels present) can look sub-foot. **Blind toe** is a different problem.
Competition metric is toe-only; GroupKFold-by-well is non-negotiable.

### 2. Floors vs crown (measured)

| Reference | Approx RMSE | Notes |
| --- | --- | --- |
| Competition crown (LB) | ~6–7 | Advanced typewell stalk / multi-hypothesis trackers |
| Last-known (subset runs) | ~11.8–13 | Early 12–40 well slices |
| Last-known (**full 773 wells**) | **~15.91** | Public floor on full toe OOF |
| Clamped stack / PF blend (full) | **~15.11** | Best pure public floor on full cohort |
| Unclamped residual / mid priors | up to ~100–1000 | Poison if dual-transport / Z-dip / lease azimuth leak into $X$ |

### 3. Full-cohort certified result (public open-core)

Artifact: `artifacts/full_train_parallel/cv_report.json` (GroupKFold 5-fold, 773 wells, ~3.78M toe stations).

| Head | OOF RMSE |
| --- | ---: |
| **Residual on `prior_stack` (Gate3 ACCEPT)** | **14.54** |
| Best pure floor (`prior_stack`) | 15.11 |
| `prior_pf_blend` | 15.11 |
| `last_known` | 15.91 |
| `pf_bag` | 15.51 |
| `particle_filter` alone | 19.14 |

* Residual beats **all** pure baselines on this seal.
* Residual gain vs best pure floor is **small** (~0.5–0.6 ft)—the floor is the story, not the tree.

### 4. Parallel / SIMD scale

* Abstraction: `rwt.simd` contiguous float64, vectorized NCC helpers; OpenBLAS uses NEON/ASIMD (ARM) or AVX (x86).
* **Dominant wall-time win:** multi-process well assemble (e.g. 7 workers), not SIMD alone.
* Full-basin public train order: tens of minutes on 8-core ARM with workers=7 (assemble + residual).

### 5. Residue (not forced OPEN)

* Gap **~14.5 → ~6–7** remains **RESIDUE**: better typewell stalk (public PF process model, private beam/DTW-class match), neighbor TVT quality, residual feature honesty.
* **No Kaggle submit under current RESTRICT.** Public LB score is not pursued as glory.
  Measure is in-repo only (GroupKFold OOF, Gate3, provenance seals). A CSV export would only
  stamp mid-teens public score; it does **not** close the ~2× gap to LB ~6–7 and is **not** authorized.

---

## PART 5: Canonical Contracts (for agents / RAG)

### Residual train law

1. Rows: `is_predict` ∧ finite `TVT` ∧ finite selected prior.
2. Features: relative / signal / prior columns only—no form tops, no absolute $X,Y$ as residual $X$.
3. Target: $y - \mathrm{prior}$.
4. CV: GroupKFold by well.
5. Gate3: accept residual iff OOF ≤ base prior, ≤ last-known, ≤ best pure floor.
6. Seal report with public tree SHA-256; `use_residual` + `prior_column` stored on model for strict serve.

### Serve law

1. Stored `prior_column` resolved **strict** (no silent swap).
2. Non-finite prior or residual preds → hard fail (no silent fill).
3. If `use_residual` false or empty models → pure prior only.

### Submission schema (dormant — do not upload)

* If ever authorized: columns `id`, `tvt` with `id = {WELLNAME}_{row_index}`.
* **Current law: do not build/upload competition CSVs for glory.** `scripts` / `rwt.submit` stay for schema tests only.

---

## PART 6: Suggested RAG Chunk Boundaries

1. Theory / Volume II axioms  
2. $SE(3)$ drilling isomorphism  
3. Competition law + three traps  
4. Architecture (7 modes + three layers + engines + gates)  
5. Empirical ledger (floors, full-cohort 14.54, residue to LB)  
6. Ops (parallel, dashboard, IP boundary, serve/submit contracts)

---

## Provenance

* Lineage: topology-sees-sequence / Volume II CTS framing; open-curve holonomy; middle-out slice-0.
* Repo open-core: `src/rwt/*`, `docs/*`, `scripts/*`.
* Full-cohort seal: public kernel `public.baseline.v1`, residual Gate3 ACCEPT, OOF selected **14.54** (773 wells).
* This document is a **technical prior-art / ops ledger**, not a patent claim.
