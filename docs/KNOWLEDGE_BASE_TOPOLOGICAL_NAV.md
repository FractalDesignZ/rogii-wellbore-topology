# KNOWLEDGE BASE: Topological Wellbore Navigation & `rwt` Pipeline

> Clean synthesis for RAG / NotebookLM. Operator law: `restrict → measure → audit → OPEN | STOP`. Residue never forced.

---

## PART 1: Theoretical Abstractions & Epistemology

### Executive Synthesis: Volume II (The Topological Toolkit)

Volume II establishes *how we know* and *how we verify* structural integrity across non-stationary systems. It bridges Volume I ontology with implementations in `JTopo`, `rwt`, and `SHEAF-STACK`.

**Axiomatic Codebase Mapping Matrix**

| Axiom Block | Mathematical Primitive | Codebase Implementation |
| --- | --- | --- |
| **4.1: Topological Invariance** | Betti numbers ($\beta_0, \beta_1$), Homology | `JTopo` & `scripts/` |
| **4.2: Filtration** | Vietoris-Rips & k-NN filtrations | `render_tda_network_graphics.py` |
| **4.3: Persistence** | Birth/Death diagrams, barcodes | Single-linkage $H_0$ Kruskal merges on well centroids |
| **4.4: Mapper** | Nerve graph & cluster summaries | Spatial wellbore networks |
| **5.1: Reverse Engineering** | Bottom-up structural recovery | `src/rwt/abstraction.py` |
| **5.2: Term Series (CTS)** | $X \to \mathcal{M} \to \mathcal{M}_v \to H_k \to Q$ | 7-Mode Pipeline ($M0 \to M6$) |
| **5.3: Quale Equivalence** | $Q(\Phi) \equiv$ Persistence Diagram | Integrated Information ($\Phi$) |
| **6.1: Topological Verification** | $d(T, T_0) < \varepsilon$ (Wasserstein) | Cryptographic SHA-256 seals |
| **6.2: Continuity Guards** | Open set preservation & G3 gate | `src/rwt/gates.py` ($G1, G2, G3$) |

**Key Theoretical Contributions**

1. **TDA epistemology (Block 4):** Long-lived features = signal; short-lived = noise.
2. **Canon of Emergence (Block 5):** Configurational sequence compresses substrate $X$ into irreducible meaning $Q$; quale as robust topological signature.
3. **Verification (Block 6):** Distance on persistence diagrams; **Gate 3** rejects residual heads that regress OOF RMSE vs pure prior floors.

**Volume III bridge:** Multi-channel series in Universal Tensor Space $U = P \times T \times M \times F$.

### Ring Closure as Holonomy (peptide lineage)

- **Closability:** Strain-free closure iff frame returns to itself.
- **Flux correspondence:** Backbone sheaf Laplacian ↔ tight-binding ring with flux $\pm\theta$.
- **Sequence sets cost, not feasibility:** Sequence rescales spectrum; Betti structure fixed. Wells are **open** curves — no false closed holonomy.

---

## PART 2: Translation to Directional Drilling

Peptide backbone and drillstring are continuous 3D paths under $SE(3)$.

| Academic | Operator | Rig-floor meaning |
| --- | --- | --- |
| $\mathbf{g} \in SE(3)$ | Pose | Bit position + orientation |
| $\nabla_T \mathbf{X} = 0$ | Parallel transport | Steady toolface along hole curve |
| $\oint_{\gamma} \mathbf{\Omega}$ | Holonomy | Accumulated toolface walk |
| $\kappa = \|\gamma''\|$ | Curvature | DLS (°/100 ft) |
| $\tau$ | Torsion | Spiraling / micro-tortuosity |
| $\mathcal{F} \xrightarrow{\pi} \mathcal{M}$ | Bundle / sheaf | BHA behavior over formation |
| $\mathbf{T} = \mathbb{C}:\mathbf{E}$ | Hooke | Stretch / twist / bucking |
| $\nabla_{\dot{\gamma}} \mathbf{v}$ | Covariant accel | Walk/drop at lithology change |

**Isomorphic claims**

- Reference-connection strain → BHA joint strain / tortuosity / drag  
- Bishop transport → geometric walk without coordinate distortion  
- Neutral tilt locus → constant-trajectory bit-tilt equilibrium  

---

## PART 3: `rwt` System Architecture (ROGII)

**Task:** Predict TVT on masked toe (1-ft stations). Metric: **RMSE**.

### Three fatal ML traps

1. **Structural leak (`ANCC`…`BUDA`):** Train-only surfaces — drop before features.  
2. **Global $Z$ overfit:** Predict **residual** $\Delta TVT = TVT - TVT_{\mathrm{prior}}$, not absolute TVT alone.  
3. **Station KFold leak:** **GroupKFold by well_id** only ($M0$ RESTRICT).

### Modes M0–M6

| Mode | Name | Role |
| --- | --- | --- |
| M0 | RESTRICT | TVT RMSE; GroupKFold-by-well |
| M1 | GEOM | Tangents, curvature, open Bishop transport |
| M2 | PRIOR | Heel-only structural prior |
| M3 | MATCH | Multi-scale GR↔typewell NCC |
| M4 | MIDDLE_OUT | Known heel vs masked toe |
| M5 | RESIDUAL | Trees on $(y - \mathrm{prior})$ |
| M6 | CERTIFY | Beat-table + provenance seal |

### Three-layer stack

$$
\mathrm{TVT}(s) = \mathrm{Prior}_{\mathrm{flat}}(s) + \mathrm{Trend}_{\mathrm{dip}}(s) + \mathrm{Residual}_{\mathrm{lithology}}(s)
$$

1. Flat prior from heel $C = \mathrm{mean}(TVT_{\mathrm{input}}+Z)$  
2. Dip plane Ridge on heel residuals → toe  
3. LGBM residual on toe labels (GR, curvature, NCC, spatial, PF)  

### Engines

- **Kinematics:** $\kappa$, $\tau$, $E_{\mathrm{ref}}$  
- **Spatial sheaf:** k-NN ($d < 600$ ft), azimuth-gated transfer  
- **Particle filter:** Heel-locked GR↔typewell track + confidence blend  

### IP

- Public: `src/rwt/` AGPLv3  
- Private: `private_kernel/` (beam/DTW/heavy sheaf) — export bundle strips it  

### Coupled harness (meta)

```text
supagen / prime-session: restrict → measure → audit → OPEN|STOP
rwt domain engine:       M0–M6 + G1/G2/G3
```

Cosine / LMS never production OPEN. Gate 3 owns residual acceptance.

---

## PART 4: Empirical floors & crown

| Class | Approx RMSE (ft) | Note |
| --- | ---: | --- |
| Heel in-sample reconstruct | 0.3–1.2 | Not the competition problem |
| Last-known heel baseline | 11.8–13.0 | Floor class |
| Public residual + G3 (reported full cohort) | **14.54** | Beat last_known 15.91 on that run |
| This sandbox pilot (48–64 wells, 2026-08) | **~11.1–11.4** | prior_stack / residual G3 |
| Leaderboard crown | **~6–7** | Advanced PF / Viterbi class |
| Stated hunt target | **5** | Aspirational stretch under G3 |

Unclamped residual can explode (~125). Gate 3 prevents shipping that.

**Path from ~11–14 → ~6–7 / 5:** heavier matching (beam / DTW / multi-PF) in **private** kernel; public path keeps floors + certified residual only.

### Hardware roles (this kit)

| Unit | Owns |
| --- | --- |
| CPU 24c + OpenBLAS | Geometry, PF, NCC, LGBM, GroupKFold |
| 32GB DDR5 | Station matrices + Q8 fibers |
| GPU | LMS SCOUT/PRESERVE, optional neural residual |
| NPU | Optional Job2 measure fabric — not LB authority |

---

## Operator one-pager

1. **RESTRICT:** beat 5 ft; no force-OPEN; GroupKFold-by-well.  
2. **MEASURE:** `python scripts/train_max_precision.py` (or full cohort driver).  
3. **AUDIT:** selected OOF vs floors vs 5 ft; G3 pass/fail.  
4. **CERT:** `artifacts/*/measure_vs_5ft.json` + provenance seal.  

Residue never forced. Law holds.
