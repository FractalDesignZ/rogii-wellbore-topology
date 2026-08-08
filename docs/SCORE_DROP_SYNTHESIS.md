# Score-drop synthesis — adaptive crown × KB mathematics

**RESTRICT:** measure = GroupKFold OOF + Gate3 + seals. No Kaggle submit.  
**Spine:** adaptive residual serve (station-conf, OOF-tuned threshold).  
**Target:** move residual / selected OOF from ~11 (12w) / ~15.4 (773w) toward crown **6–7** / hunt **5**.

This note synthesizes our live RWT mathematics with concepts from:

| Source | Role in synthesis |
| --- | --- |
| Petersen *Mathematics of Machine Learning* (+ twin notes) | PAC / Rademacher / ERM–SRM / CV / Tikhonov / kernels / diffusion |
| soulmachine ML cheat sheet | Bayes/MAP, HMM, state-space, MC, boosting, GP |
| *Persistent Topological Features in LLMs* (Gardinazzi et al.) | Zigzag persistence, kNN filtration, short- vs long-lived features, phase pruning |
| Classifier Series — Decision Trees (Alam) | Recursive partition, impurity, interpretability, prune |
| Random Forest — Predictive Maintenance (Alam) | Tree ensembles, feature importance, imbalance, multi-sensor residual risk |

---

## 1. Where the score actually lives

Competition score ≈ **predict-zone TVT RMSE**. Our stack is hierarchical:

```text
TVT_hat(s) = prior(s)  +  1_{adaptive fire}(s) · Δ_residual(s)
             └─ stalk MAP/ensemble ─┘   └─ PAC-gated tree bag ─┘
```

| Layer | Live encoding | Approx floor (honest) |
| --- | --- | ---: |
| last_known | forward-fill heel | 13.4 (12w) / 15.9 (773w) |
| pure `prior_stack` | last + dip + PF bag + high-conf NCC | 12.15 / 15.76 |
| residual on `prior_best` | deep LGBM 4-seed, Gate3 | 11.05 / **15.38** |
| adaptive residual | station-conf mask, thr OOF-tuned | ≤ residual when thr earns it |
| crown / hunt | private lattice + beam (residue) | **6–7 / 5** |

**Diagnosis (Bayes error + model class):**  
Public residual has already eaten most of the **tabular tree** margin over pure prior (~1–1.5 ft on 12w; ~0.4 on 773w). Gap to 6–7 is **not** more leaf depth — it is **stalk fidelity** (HMM/lattice MAP + topology of conf) and **adaptive refusal** of residual where the stalk is short-lived noise.

Petersen: if Bayes error of the residual class given current features is large, ERM on LGBM cannot close the crown gap. Gardinazzi: long-lived structure is middle-phase; short-lived is rearrange noise — **do not fire residual on short-lived**.

---

## 2. Unified program (one law)

### 2.1 Structural risk (Petersen L8–L9 + Gate3)

- **ERM** alone on residual → overfits well idiosyncrasy (rejected A/Bs: Huber, deep leaves, drop-base-from-X).  
- **SRM / Gate3:** accept residual head only if OOF RMSE ≤ every pure floor (stack, last_known, …).  
- **Tikhonov-style hard constraint:** `|ΔTVT| ≤ 18` (published clamp); residual Δ clipped vs base.  
- **GroupKFold-by-well** = honest risk under dependence (stations ≠ i.i.d.; well is the PAC unit).

### 2.2 Adaptive selector (name of the game)

Map decision-tree **partition** + RF **risk gate** onto serve:

```text
fire(s) ⇔ finite(pf_confidence[s]) ≥ τ*
τ* = argmin_τ  RMSE( apply_mask(residual, prior; conf ≥ τ) )   on OOF
adaptive_serve ⇔ RMSE_adaptive < RMSE_global_residual
```

This is **model selection** between hypothesis classes  
H₀ = pure prior, H₁ = residual everywhere, H_τ = residual on high-conf stations.  
Same mask at train and serve (`adaptive_residual_mask`) — no form leak.

**Well-level fire** (residual beats prior on that well) is **diagnostic only** unless serialized: it needs y_true (Petersen: cannot use labels at test). Conf is the **serveable** proxy for local Bayes risk.

### 2.3 Prior as MAP on a chain (cheat sheet HMM / state-space)

| Concept | RWT encoding |
| --- | --- |
| Hidden state | TVT along typewell stalk |
| Observation | lateral GR (+ ΔGR shape) |
| Transition | MD-slope ±0.008, last±18 |
| Emission | GR likelihood / window NCC |
| PF (public) | particle Monte Carlo, adaptive process noise |
| Lattice (private) | banded Viterbi MAP |
| Conf | path margin / particle concentration |

Pure lattice mean **lost** pilot (15.98 vs last 12.17) but **won** isolated wells — classic **mixture of regimes**. Adaptive blend (conf × lattice → last) is the correct SRM step, not pure MAP force-OPEN.

### 2.4 Zigzag persistence on the lateral (Gardinazzi → geosteering)

Reinterpret MD sequence as “layers” and stations as points in (MD, TVT_hyp, GR) space:

| Zigzag idea | Score-drop use |
| --- | --- |
| kNN filtration | spatial multi-neighbor index (already live) |
| short-lived 0/1-holes | GR dropouts, alias tracks, conf spikes → **do not fire residual** |
| long-lived features | stable typewell match / formation walk → **trust residual / lattice** |
| inter-layer persistence Z_p | conf that **survives** along MD (rolling min conf, lifetime) as feature + thr input |
| phase prune (middle plateau) | dual residual off (set-17): do not keep diluted heads |

**Concrete features to MEASURE (public-safe):**

1. `pf_conf_lifetime` — length of consecutive stations with conf ≥ thr (long-lived).  
2. `pf_conf_birth_rate` — fraction of local conf “births” (short-lived churn).  
3. `prior_disagreement_persist` — rolling persistence of stack vs last vs NCC disagreement.  
4. `e_ref_persist` — screw-norm stability along MD (geometry phase).

Feed (1–4) into residual X **and** into adaptive mask as second thr dimension:

```text
fire(s) ⇔ conf(s) ≥ τ*  AND  lifetime(s) ≥ L*
```

(Tune τ*, L* on OOF only — same SRM law.)

### 2.5 Tree ensembles (Decision Trees + RF maintenance)

| RF/DT practice | RWT mapping | Status |
| --- | --- | --- |
| Bag of trees | 4-seed LGBM bag | live |
| Feature importance | diagnose stalk vs residual feats | measure tool |
| Imbalance (rare failure) | hard wells rare; adaptive fire fraction small | live (2/12 hist) |
| SMOTE-style oversample | **avoid** on stations (form leak / GroupKFold break) | reject |
| Max depth / leaves | deep profile held; aggressive leaves rejected | sealed |
| Multi-sensor fusion | GR + geometry + neighbor + PF conf | live |

**Do not** re-open dual residual or soft-nb-into-stack (measure killed them). Use RF **importance** to prune collinear residual features on next 12w A/B (Petersen: reduce hypothesis complexity → better gen bound).

### 2.6 Diffusion / spectral (Petersen L19–L20)

Neighbor graph of wells ≈ discrete Laplace on XY:

- Live: multi-neighbor weighted TVT transfer.  
- Residue: diffusion map coords as residual features (spectral embedding of train wells) — **only if** OOF beats set-19; watch form leak (must recompute fold-local).

---

## 3. Score-drop levers ranked by expected ΔRMSE

| Priority | Lever | Math license | Expected | Path |
| ---: | --- | --- | --- | --- |
| **P0** | OOF-tuned adaptive thr + lifetime mask | SRM / zigzag long-lived | 0.05–0.3 on 12w; larger if residual poison stations | public |
| **P0** | Conf-blend private lattice (not pure MAP) | adaptive MAP + RF risk gate | large on bad wells; mean only if trust gate | private |
| **P1** | Persistence / lifetime features in residual X | zigzag descriptors | 0.1–0.4 if conf alone is weak | public |
| **P1** | Feature importance prune + re-seal | VC / Rademacher shrink | 0–0.2; often hold | public |
| **P2** | Banded Viterbi + better emission (multi-scale) | HMM MAP | private stalk → 6–9 band | private |
| **P2** | Fold-local spectral neighbor coords | diffusion maps | uncertain | public A/B |
| **STOP** | Deeper LGBM only / dual residual / soft-nb stack | failed SRM | reject | — |
| **STOP** | Neural residual without Gate3 | failed floors | reject | — |

Public path alone is unlikely to hit **6–7**. Honest split:

- Public adaptive + persistence → push **11 → ~10.5–10.8 (12w)** and **15.4 → ~15.0-ish (773w)** if lucky.  
- Crown **6–7** needs **private** stalk (lattice/beam) **adaptively** blended, then residual on that prior under Gate3.

---

## 4. Operational MEASURE plan (adaptive first)

### Wave A — public adaptive (CPU-light)

1. Seal 12w with current OOF thr sweep (`artifacts/adaptive_thr_12w`).  
2. Add lifetime features; re-seal; Gate3 + adaptive_selected.  
3. Feature-importance prune (drop bottom quartile of residual feats if OOF ≤ hold).  
4. Compare `oof_rmse_selected` vs set-19 **11.045 / adaptive 11.040**.

### Wave B — private adaptive lattice (local only)

1. Pilot with conf-blend + well-adaptive (already in `measure_private_lattice_pilot.py`).  
2. Soft-pull `prior_best` only when `lattice_conf` long-lived (kernel_impl).  
3. Residual train with `RWT_KERNEL=private` only if pure lattice / blend floors beat public stack on OOF.  
4. Never force-OPEN pure lattice mean (pilot: mean loses, tails win).

### Wave C — topology residue (optional)

1. Cheap 1-D persistence on (MD, ΔTVT_prior) per well → barcode length features.  
2. Zigzag between PF particle cloud snapshots at MD windows (if cost allows).  
3. Prune “middle plateau” dual heads (already off).

Law at every step: **restrict → measure → audit → OPEN|STOP**. Residue never forced.

---

## 5. Compact math cheat sheet for this repo

```text
Gate3:     R_res ≤ min_k R(prior_k)          # SRM accept residual class
Adaptive:  τ* = argmin_τ R( residual ⊙ 1_{c≥τ} + prior ⊙ 1_{c<τ} )
Serve:     y = apply_adaptive_residual(res, prior, mask(c, τ*))
Clamp:     |Δ| ≤ 18  (Tikhonov hard set)
PAC unit:  well (GroupKFold)
MAP stalk: argmax path  Σ log emit + log trans   (PF approx / Viterbi exact)
Persist:   long-lived conf → fire; short-lived → prior
Ensemble:  seed bag mean of trees (boosting/RF idea without label SMOTE)
```

---

## 6. What we refuse (KB-informed)

| Temptation | Why refuse |
| --- | --- |
| SMOTE / station oversample | breaks GroupKFold PAC unit; form leak |
| Force dual residual | dilutes best head (set-16/17) |
| Soft-nb into stack | polluted pure floor (set-11) |
| Pure lattice mean as default prior | pilot mean 15.98 > last 12.17 |
| Neural residual without floors | Gate3 REJECT history |
| Report well_fire as serve OOF | train≠serve (fixed set-20 parity) |

---

## 7. Wave A landed (code) + next MEASURE

**Landed in tree (Wave A):**

1. **LAS null decode** at load (`-999.25` etc → NaN; GR absurd floor) — `rwt.load`.  
2. **Gap abstraction** — `gr_is_null`, `gr_gap_run`, `gr_valid_frac_win`, `gr_was_imputed`; short-gap fill for rolls only — `rwt.curves`.  
3. **Dual adaptive fire** — OOF joint sweep thr × min conf-run lifetime; blob `adaptive_min_lifetime`; train≡serve mask — `rwt.gates` / train / submit.

**Next MEASURE:**

1. 12w seal `artifacts/wave_a_12w` — compare `oof_rmse_selected` vs set-19 **11.045**.  
2. Private lattice conf-blend pilot re-run.  
3. MAXOP ledger set-22 OPEN|STOP from those measures.

**Bottom line:** Adaptive is not a garnish — it is the **selector between ERM residual and MAP prior**, licensed by PAC/SRM and sharpened by zigzag “long-lived only.” Trees polish the residual; topology decides **whether** to fire; lattice (private) rebuilds the stalk. That is the only coherent path to drop the loco score toward crown without lying to Gate3.
