# Manifold + GR corroboration — unified synthesis

**RESTRICT:** GroupKFold OOF + Gate3 + seals only. No Kaggle submit.  
**Sources:** deep-research-8 (shared geology manifold), deep-research-9 claims (synthetic GR↔TVT, train/test asymmetry — 18/18 verified), Wave A seal `artifacts/wave_a_12w`, live `src/rwt/*`.

**Parent construct:** [`TASK_GR_TVT_CONSTRUCT.md`](TASK_GR_TVT_CONSTRUCT.md) — task decompose/align under GR@TVT.

---

## 1. The weird thing you noticed (and yes — get it)

Competition structure is intentional asymmetry:

| Channel | **Train map set** | **Test / new map set** |
| --- | --- | --- |
| Geometry MD, XYZ | yes | yes |
| Lateral **GR** | yes | yes |
| Heel **TVT_input** (pre-PS) | yes | yes |
| Full-station **TVT** answers | **yes** | **withheld** |
| Formation tops ANCC…BUDA | **yes (train-only)** | **withheld** |
| Typewell TVT+GR | yes | yes |

They are **giving answers on the training pad** (full TVT + tops), then a **new map** with the same sensors (GR, XYZ, heel TVT_input) but **no TVT labels**. The game is not “invent geology from nothing” — it is:

> Learn the **shared formation manifold** and the **GR↔stratigraphy map** from labeled wells, then **place** unlabeled laterals on that same manifold using GR + heel + neighbors.

That is why pure residual trees plateau: the crown is **placement on geology**, not more ΔTVT regression tricks.

---

## 2. Simple concept (research-8) — one manifold, three interpolants

```text
                    shared geology M
              (continuous layer / TVT field)
               /            |            \
        XY pad graph    dip plane     typewell stalk
        (neighbors)     (flat+dip)    (GR as emission)
               \            |            /
                 prior_stack / prior_best  (±18)
                            |
              adaptive residual (conf × lifetime) only if Gate3
```

\[
\widehat{\mathrm{TVT}}(s)
  = \mathrm{prior}_M(s)
  + \mathbf{1}_{\mathrm{fire}}(s)\cdot\Delta_{\mathrm{res}}(s)
\]

| Interpolant | What it does | Live |
| --- | --- | --- |
| **Spatial sheaf-lite** | Transfer TVT layer position well↔well (k-NN XY, heel offset, IDW) | `spatial.py` |
| **Dip field** | \(\mathrm{TVT} \approx C-Z +\) Ridge\((X,Y)\) plane on heel residual | `priors.py` |
| **Typewell stalk** | MD → TVT via GR match (PF / NCC / private Viterbi) | `particle_track`, `gr_match`, private lattice |

Named engines fuse into **one** clamped stalk. Extra residual heads / soft-nb-into-stack / deeper-only LGBM = STOP (measure).

---

## 3. GR corroboration (research-9) — the falsifier

If the manifold is real, **gamma must agree with known TVT**:

At true (or hypothesized) stratigraphic position \(t\):

\[
\mathrm{GR}_{\mathrm{lateral}}(s)
  \;\approx\;
  a\cdot \mathrm{GR}_{\mathrm{typewell}}(t) + b
  \quad\text{(heel-calibrated affine)}
\]

Already encoded / tested:

| Check | Where |
| --- | --- |
| Build GR by interp(typewell @ true TVT)+noise | `synthetic_well` |
| Mask toe TVT; keep offline `TVT_true` | same + middle-out |
| Toe RMSE vs `TVT_true` | `smoke_synthetic.py` |
| PF GR×ΔGR path beats last_known on synthetic toe | unit tests |
| At true TVT, mismatch near zero after heel affine | `test_gr_mismatch_at_true_tvt_near_zero` |
| Never use ANCC…BUDA as residual features | `FORM_SURFACE_LEAK_COLS` |

**Corroboration loop (score-aligned):**

```text
1. Propose TVT_hat on predict zone (prior / residual / lattice).
2. Query GR_tw = typewell_gr_at_tvt(TVT_hat) [heel-calibrated].
3. Score GR consistency: ||GR_lat − GR_tw|| (level + ΔGR).
4. If GR agrees → raise conf / fire residual / trust lattice.
5. If GR disagrees → fall back to last/stack (adaptive refuse).
```

Wave A already supports the refuse side: LAS nulls→NaN, gap features, conf×lifetime mask. Lifetime did not win 12w OOF; **GR-consistency conf** is the next natural conf channel.

---

## 4. Synthetic interpolated dataset protocol (mirror competition)

Goal: multi-well pad where **geology is continuous**, GR is generated from the manifold, and train/test asymmetry matches Kaggle.

### 4.1 Build pad manifold \(M\)

1. Define a **shared typewell** \(G_{\mathrm{tw}}(\mathrm{TVT})\) (formation fingerprint).  
2. Define a **smooth TVT field** on pad XY (or XY + MD): e.g. planar dip + mild corrugation.  
3. Place \(N\) laterals with realistic MD, XYZ paths through that field → **true TVT(s)** per station.  
4. Set \(\mathrm{GR}(s) = G_{\mathrm{tw}}(\mathrm{TVT}_{\mathrm{true}}(s)) + \epsilon\) (+ optional washouts / LAS nulls).  
5. Heel: expose `TVT_input` = true TVT; toe: NaN TVT, keep `TVT_true` offline only.

### 4.2 Train vs “new map” split

| Wells | Role | Labels |
| --- | --- | --- |
| Pad A (train) | full TVT on all stations + optional fake tops | like competition train |
| Pad B (test) | same GR/XYZ/heel TVT_input; **no TVT** | like competition test |
| Eval | RMSE vs hidden `TVT_true` on B toes | offline only |

Never put `TVT_true` or tops into serve features.

### 4.3 What “GR corroborates knowns” means (metrics)

| Metric | Pass idea |
| --- | --- |
| **Placement RMSE** | \(\mathrm{RMSE}(\widehat{\mathrm{TVT}}, \mathrm{TVT}_{\mathrm{true}})\) on synthetic toe |
| **GR recovery** | At \(\widehat{\mathrm{TVT}}\), mismatch vs lateral GR low iff placement good |
| **Identity check** | At \(\mathrm{TVT}_{\mathrm{true}}\), mismatch ≈ noise floor (affine) |
| **Neighbor transfer** | Hold out one well’s toe; reconstruct from pad neighbors only |
| **Null stress** | Inject -999.25 / gaps; adaptive must not fire residual there |

If neighbor+dip+stalk **fail** GR recovery while residual “wins” OOF, residual is fitting noise — STOP force-OPEN.

### 4.4 Relation to live code

| Piece | Status |
| --- | --- |
| Single-well `synthetic_well` + smoke | **live** |
| Multi-well pad field generator | **not yet** (next scaffold) |
| GR@TVT interp + heel affine | **live** |
| Middle-out heel/toe mask | **live** |
| Adaptive refuse low conf | **live** (Wave A; lifetime optional) |
| Private lattice MAP on stalk | **local** |

---

## 5. Wave A MEASURE (empirical anchor)

`artifacts/wave_a_12w` (12 wells, full assemble + residual CV):

| Metric | Value |
| --- | ---: |
| residual OOF | **10.741** |
| set-19 residual | 11.045 |
| Δ | **−0.30** |
| best pure floor | prior_best 12.637 |
| Gate3 | ACCEPT |
| adaptive OOF best | 11.098 (thr=0.15, L=0) |
| adaptive_serve | **False** (residual better) |

Read: null hygiene + gap features **helped residual**; dual lifetime fire **not selected**. Still far from crown 6–7 → **stalk/manifold**, not thr polish.

---

## 6. Unified law (OPEN|STOP)

```text
OPEN  public residual when Gate3 + features improve OOF (Wave A did).
OPEN  synthetic pad manifold for GR↔TVT corroboration (falsifies stalk).
OPEN  adaptive fire only where conf (and optionally GR-agree) earns it.
STOP  form tops as features; TVT_true at serve; pure lattice mean force-OPEN.
STOP  dual residual / soft-nb-into-stack / SMOTE stations.
STOP  treating -999.25 as numeric GR.
```

**Crown path rewritten in one sentence:**  
*Interpolate continuous layer position on a shared geological manifold; use gamma only to corroborate placement; residual only where corroboration is long-lived and Gate3 accepts.*

---

## 7. Next concrete MEASURE (when coding)

1. **`scripts/measure_manifold_gr_synth.py`** — multi-well pad synthetic; report placement RMSE + GR mismatch@true vs @predicted.  
2. **Neighbor-only transfer A/B** on synthetic pad (no residual) — does sheaf-lite recover TVT?  
3. **GR-agreement conf** channel → feed adaptive mask (compare to pf_confidence-only).  
4. Hold Wave A residual as public floor until stalk beats it under Gate3.

Law remains: restrict → measure → audit → OPEN|STOP.
