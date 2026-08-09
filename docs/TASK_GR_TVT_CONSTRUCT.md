# Task construct: GR@TVT — decompose, align, measure

**Law:** RESTRICT → measure (GroupKFold-by-well OOF + Gate3 + seals) → audit → OPEN|STOP.  
**No Kaggle submit under RESTRICT.**  
**Canonical object:** geology as \(\mathrm{GR} = G_{\mathrm{tw}}(\mathrm{TVT})\); laterals as paths that must be **placed** on that book.

This document synthesizes research of merit (deep-research 7–9, KB/crown seals, Wave A MEASURE, PAC/SRM/HMM/TDA licenses) into one **task-aligned construct**.

---

## 0. One sentence

> **Coordinates + sensors place the bit on a stratigraphic axis; the formation is a GR@TVT tensor; labels teach placement and small residual; the new map keeps GR and drops TVT.**

---

## 1. Competition asymmetry (what they give vs withhold)

| Channel | Train pad | Test / new map | Role in construct |
| --- | --- | --- | --- |
| MD, X, Y, Z | yes | yes | Path through space; neighbor graph; dip |
| Lateral **GR** | yes | yes | Observation of formation book |
| Heel **TVT_input** (pre-PS) | yes | yes | Anchor / origin on stalk |
| Full-station **TVT** | **yes** | **no** | Stratigraphic coordinate (answer) |
| Tops ANCC…BUDA | **yes** | **no** | Train-only; **never residual features** |
| Typewell TVT + GR | yes | yes | The book \(G_{\mathrm{tw}}(\mathrm{TVT})\) |

**Get it:** answers on one map → learn book + placement; new map with same GR → **invert** book. Not invent geology from XYZ alone.

Score: \(\mathrm{RMSE}(\widehat{\mathrm{TVT}}, \mathrm{TVT})\) on 1-ft predict-zone stations (toe where `TVT_input` missing).

---

## 2. Core tensors (practical decomposition)

### 2.1 Formation book (always known)

\[
G_{\mathrm{tw}} : \mathrm{TVT} \mapsto \mathrm{GR}
\quad\text{(typewell curve / grid)}
\]

**Tensor form:** 1-D table `(tw_tvt[j], tw_gr[j])` or dense vector `GR_tw[t_bins]`.  
**Ops:** `interp(t)`, heel affine \(a\cdot G_{\mathrm{tw}}+b\), window NCC / likelihood along \(t\).

### 2.2 Lateral path (train and test)

| Field | Shape | Train | Test |
| --- | --- | --- | --- |
| `MD[s], XYZ[s]` | \(S\) | known | known |
| `GR[s]` | \(S\) | known (LAS null → NaN) | known |
| `TVT_input[s]` | \(S\) | heel known, toe NaN | same |
| `TVT[s]` | \(S\) | **label** | **absent** |
| `TVT_true[s]` | \(S\) | synthetic offline only | never serve |

### 2.3 Pad manifold (multi-well)

Shared continuous field of layer position:

\[
t : (X,Y) \text{ or } (X,Y,\mathrm{MD}) \mapsto \mathrm{TVT}
\]

**Live (sheaf-lite):** k-NN on lateral XY, heel offset, IDW TVT transfer.  
**Not live:** joint multi-well sheaf Laplacian / global surface fit (private residue).

### 2.4 Hierarchical predictor (serve)

\[
\widehat{\mathrm{TVT}}(s)
  = \underbrace{\mathrm{prior}_M(s)}_{\text{placement on GR@TVT + pad + dip}}
  + \mathbf{1}_{\mathrm{fire}}(s)\cdot
    \underbrace{\Delta_{\mathrm{res}}(s)}_{\text{Gate3 tree polish}}
\]

Clamp law (Tikhonov-style hard set): \(|\Delta|\le 18\) vs base / last as published.

---

## 3. Task decomposition (aligned work packages)

```text
WP0  HYGIENE     nulls, masks, no form leak
WP1  BOOK        GR@TVT typewell tensor + heel affine
WP2  INVERSE     place lateral on book (PF / NCC / lattice)
WP3  PAD         well↔well TVT transfer (neighbors)
WP4  STRUCTURE   flat + dip plane prior
WP5  FUSE        prior_stack / prior_best (±18)
WP6  RESIDUAL    LGBM Δ on prior_best, Gate3 only
WP7  ADAPTIVE    fire residual only where conf earns it
WP8  FALSIFY     synthetic pad: GR corroborates knowns?
```

| WP | Job | Live encoding | Research license |
| --- | --- | --- | --- |
| **0** | Decode LAS −999.25 → NaN; gap features; no tops in X | Wave A `load`/`curves` | oilfield null law; form-leak trap |
| **1** | Maintain \(G_{\mathrm{tw}}\); GR at hypothesized \(t\) | `gr_match.typewell_gr_at_tvt` | synthetic GR=interp(tw@true t) |
| **2** | Invert \(\mathrm{GR}_{\mathrm{lat}}\mapsto t\) with path reg | PF (public MC), NCC, private Viterbi | HMM/MAP/state-space |
| **3** | Interpolate \(t\) across pad | multi-neighbor spatial | sheaf-lite / diffusion on XY |
| **4** | Soft structural prior | \(C-Z\) + Ridge heel dip | three-layer prior contract |
| **5** | One stalk ensemble | stack / best blends | crown path; engines are arms not heads |
| **6** | Polish when stalk incomplete | deep 4-seed LGBM, Gate3 | ERM–SRM / RF bag; PAC unit = well |
| **7** | Refuse residual on weak placement | station conf thr (± lifetime) | Bayes-risk proxy; zigzag long-lived |
| **8** | Prove book+inverse, not just residual OOF | smoke + pad synth (next) | train/test asymmetry mirror |

---

## 4. Inverse problem (the actual task)

Given typewell book and lateral observations:

\[
\widehat t(s)
  = \arg\min_{t(\cdot)}
  \underbrace{\sum_s \rho\big(\mathrm{GR}(s),\, G_{\mathrm{tw}}(t(s))\big)}_{\text{GR corroboration}}
  + \underbrace{\mathrm{Reg}_{\mathrm{path}}(t)}_{\text{MD slope, }\pm 18}
  + \underbrace{\mathrm{Reg}_{\mathrm{heel}}(t)}_{\text{TVT_input lock}}
  + \underbrace{\mathrm{Reg}_{\mathrm{pad}}(t)}_{\text{neighbor TVT}}
\]

| Solver | Role | Status |
| --- | --- | --- |
| Particle filter | MC approx of stalk | public live |
| Multi-scale NCC | local lag / peak match | public live |
| Banded Viterbi lattice | exact MAP on discrete \(t\) | private local |
| Neighbor IDW | pad term only | public live |
| LGBM residual | not inverse; \(\Delta\) after prior | public live, Gate3 |

**GR corroboration identity:** at true \(t^\*\), heel-calibrated mismatch \(\| \mathrm{GR}_{\mathrm{lat}} - G_{\mathrm{tw}}(t^\*) \|\) ≈ noise floor (unit tests).  
If mismatch high at \(\hat t\) → placement wrong → **low conf → no residual fire**.

---

## 5. Math licenses mapped onto the construct

| Source idea | Maps to WP | Use / refuse |
| --- | --- | --- |
| PAC / GroupKFold-by-well | 6–7 measure | stations ≠ i.i.d.; well is unit |
| ERM–SRM / Gate3 | 6 | residual class only if OOF ≤ all pure floors |
| Tikhonov / hard clamp ±18 | 2,5,6 | published band |
| HMM / MAP / state-space | 2 | stalk inverse |
| Bayes risk | 7 | conf ≈ local risk of residual |
| Zigzag short- vs long-lived | 7 | lifetime optional; short spikes refuse residual |
| RF / trees bag | 6 | 4-seed LGBM bag; importance prune A/B |
| kNN / diffusion on XY | 3 | neighbor transfer |
| Sheaf / section over base \(M\) | 3–5 | framing; public = sheaf-lite only |

**Refuses (STOP):** SMOTE stations; dual residual default; soft-nb into stack; tops as features; −999.25 as numeric GR; pure lattice mean force-OPEN; neural residual without Gate3.

---

## 6. Empirical anchors (findings of merit)

### Wave A 12w (`artifacts/wave_a_12w`)

| Metric | Value | Read |
| --- | ---: | --- |
| residual OOF | **10.741** | OPEN vs set-19 **11.045** (−0.30) |
| best pure floor | prior_best 12.637 | Gate3 ACCEPT |
| adaptive OOF | 11.098 (thr=0.15, L=0) | residual better → adaptive_serve **False** |
| lifetime | L*=0 wins | dual life not selected this cohort |

**Merit:** hygiene + gap features help residual slightly. **Not merit for crown:** thr/lifetime polish alone.

### Cohort floors (order of magnitude)

| | 12w | 773w-ish |
| --- | ---: | ---: |
| residual | ~10.7–11.0 | ~15.4 |
| pure stack / best | ~12.1–12.6 | ~15.8 |
| last_known | ~13.4 | ~15.9 |
| crown / hunt | **6–7 / 5** | same target |

Gap to crown = **WP2/3 stalk + pad**, not WP6 depth.

### Synthetic / smoke (research-9 verified)

- Build: true \(t\) → GR = \(G_{\mathrm{tw}}(t)+\epsilon\); mask toe; keep offline `TVT_true`.  
- Eval: toe RMSE vs truth; PF must beat last_known; mismatch@true \(t\) small.  
- Mirror competition: no tops in X; no `TVT_true` at serve.

---

## 7. Alignment checklist (every change)

Before CODE, ask:

1. **Does this improve placement on GR@TVT** (WP1–5) or only fit \(\Delta\) (WP6)?  
2. **Train/serve parity?** Same mask, medians, prior column, adaptive thr/L.  
3. **Form leak?** No tops, no full-path TVT in serve features, no station-i.i.d. CV.  
4. **Null law?** Sentinels → NaN; gaps abstract or short-fill only.  
5. **Gate3?** Residual OPEN only if OOF beats pure floors.  
6. **Adaptive?** Fire only if conf (and preferably GR-agree) earns it; else pure prior.  
7. **Falsifier?** Synthetic: does GR corroborate \(\hat t\) when we claim accuracy?

If a change fails (1) and only chases (6) after residual already near Bayes floor → likely STOP.

---

## 8. Recommended MEASURE sequence (aligned)

| Order | Experiment | WP | Success |
| --- | --- | --- | --- |
| A | Keep Wave A residual as public floor | 0,6 | sealed 10.74 12w |
| B | Pad synthetic GR@TVT + neighbor-only RMSE | 1,3,8 | GR mismatch@true low; transfer beats last_known |
| C | GR-agreement as conf channel → adaptive | 1,7 | **CODE live + MEASURE:** residual **10.741** holds; adaptive still loses (best channel `pf` 11.10; product 11.98; gr_only 11.35; geomean 11.20). OOF picks channel; `adaptive_serve=false`. Seal `artifacts/exp_c_gr_agree_12w` |
| D | Private lattice conf-blend prior under Gate3 | 2,5,6 | **12w OPEN residual 10.471**; **40w pure stack 11.090** (all residual bases REJECT after Gate3 floor fix). Residual on pf_blend was false OPEN when stack floor dropped for sparse NaN. Seals: `exp_d_lattice_*`, `base_ab_gate3_fix.json` |
| E | Residual on improved prior_best | 6 | selected OOF drops vs Wave A |
| F | 40w / 773w seal only after B–E OPEN | all | no surprise form leak |

**Exp C encoding:** `|GR − G_tw(prior_best)| → exp(−|mm|/20)` = `gr_agreement_conf`; product with `pf_confidence` = `adaptive_conf` (mask channel). Residual X unchanged (Wave A floor preserved). Gate3 still requires residual OOF ≤ floors before adaptive can select.

---

## 9. Diagram — full construct

```text
┌─────────────────────────────────────────────────────────────┐
│  FORMATION BOOK   G_tw : TVT → GR     (typewell tensor)     │
└────────────────────────────▲────────────────────────────────┘
                             │ emit / match
┌────────────────────────────┴────────────────────────────────┐
│  LATERAL PATH   GR[s], MD, XYZ, heel TVT_input               │
│  TRAIN only: TVT[s] labels   TEST: TVT withheld              │
└───────┬──────────────────┬──────────────────┬───────────────┘
        │ WP2 inverse      │ WP3 pad          │ WP4 structure
        ▼                  ▼                  ▼
   PF / NCC / lattice   neighbor TVT      flat + dip
        \                  |                  /
         \                 |                 /
          ▼                ▼                ▼
              WP5  prior_stack / prior_best  (±18)
                          │
              WP7  fire?  conf · GR-agree · lifetime
                          │
              WP6  + Δ_res  iff Gate3 ACCEPT
                          │
                          ▼
                    TVT_hat  →  RMSE score
```

---

## 10. Bottom line

| Concept | Statement |
| --- | --- |
| **Geology** | GR@TVT book |
| **Lateral** | Path reading the book |
| **Train** | Book + true page numbers (TVT) |
| **Test** | Book + GR reading; page numbers hidden |
| **Prior** | Invert GR→page + pad + dip |
| **Residual** | Small page correction if SRM allows |
| **Adaptive** | Don’t “correct” when the reading is bad |
| **Crown** | Better inverse on the book, not deeper trees alone |

**OPEN** residual polish that beats floors (Wave A).  
**OPEN** synthetic pad + GR corroboration as the stalk falsifier.  
**STOP** anything that treats the task as generic tabular regression of TVT from XYZ without the GR@TVT book.

---

## Related docs

- [`MANIFOLD_GR_CORROBORATION.md`](MANIFOLD_GR_CORROBORATION.md) — pad manifold + synthetic protocol  
- [`SCORE_DROP_SYNTHESIS.md`](SCORE_DROP_SYNTHESIS.md) — PAC/SRM/TDA → levers  
- [`CROWN_PATH.md`](CROWN_PATH.md) — cumulative public axes + rejected A/B  
- [`MAXOP_CR_REVIEW_LEDGER.md`](MAXOP_CR_REVIEW_LEDGER.md) — set history  

**Primary API metaphor for the team:** *tensorize the formation as GR@TVT; the rest is placement, fuse, and gated residual.*

**Full stack (MWD + two-step + seals):** [`ALL_CONTEXT_OPERATOR.md`](ALL_CONTEXT_OPERATOR.md)
