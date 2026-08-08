# Max-op CodeRabbit review ledger (public path)

Law: `restrict → measure → audit → OPEN|STOP`. Residue never forced.  
RESTRICT: no Kaggle submit from measure drivers.

## Landed on `main`

| Set | PR | Results → fixes |
|-----|-----|-----------------|
| **1** | #7 | OOF `row_i` map; Gate3 OOF mask; index shuffle; ONNX/private fail-closed |
| **2** | #7 | Floor subsets; empty OOF `ok=False`; structure-only ONNX; unit packs |
| **3** | #7 | G3 selected uses best-floor OOF; safe thread env parse; GPU `--workers` |
| **4** | #8 | Floors fully finite on residual OOF; private ImportError fail-closed; fold clamp; LGBM audit meta |
| **5** | #9 | nanmedian impute; `n_splits` clamp in API; `--prior-col`; unit impute |

## Open / in flight

| Set | PR | Results → fixes |
|-----|-----|-----------------|
| **1** (crown) | #11 | seed-local NCC band (alias 21.8→16.3); stack-led prior_best weights (12w) |
| **2** (crown) | #11 | (1) heel conf no bleed into toe; (2) clamp_origin freeze + walk last on known; (3) prior_best heel lock TVT_input-first; (4) TVT-sort typewell + expand empty seed band |
| **3** (crown) | #11 | residual default `prior_stack`; toe-weighted LGBM + deeper default; median feature impute; high-conf NCC soft in stack (bag arm kept). 12w residual **11.27** / pure stack **12.33** |
| **4** (crown) | #11 | dual seed bag; residual Δ clip ±18; prior_disagreement; soft neighbor→stack. Huber rejected. CR P1: fold-local medians + persist serve medians + serve refit. 12w residual **11.08** honest |
| **5** (crown) | #11 | neighbor soft **before** prior_best; reject last-reclamp / trust-shrink / collinear feats. Score hold 11.08. |
| **6** (crown CR) | #11 | heel-safe soft nb; MAX_TVT_DEV; serve zero-fill; **prior_disagreement recompute after neighbor** (OPEN→fixed) |
| **7** (crown CR) | #11 | `compute_prior_disagreement` helper; deterministic post-neighbor test (no skip); reject bag-std/3-seed/NCC-dense A/B. Score hold **11.08** |
| **8** (crown) | #11 | GR mismatch@stack/last/ncc (heel-cal typewell); `_prior_stack_pre_neighbor` col (no attrs concat break). 12w residual **11.06** |
| **9** (crown) | #11 | shared `fit_heel_gr_affine` once per well; reject mismatch-delta feat + maxp-12w. Score hold **11.06** |
| **10** (crown CR) | #11 | **P1 spatial parity**: finalize/seal shared; single kernel after final spatial |
| **11** (crown) | #11 | `STACK_SOFT_NB_WEIGHT=0` (soft-nb into residual base hurt stack 12.33→12.62 / res 11.06→11.48); neighbors remain in prior_best. Residual hold **11.06**. Tests: default-off stack==pre_nb; trusted nb shifts prior_best; weight 0.12 moves stack. |
| **12** (crown CR) | #11 | fold_best_iters only (no fold boosters); seal-phase timing; finalize guard/reuse nb; CR set-11 test polish |
| **13** (crown CR) | #11 | `select_n_refit` + tests; residual MD-smooth hook train/serve (**w=21 rejected 11.09**; default 0); `fold_best_iters` + `residual_smooth_window` in report/blob; CountingKernel `super()`; tautology test cleanup. Soft-nb P2 closed. |
| **14** (crown) | #11 | residual default base **`prior_best`** (12w OOF **10.95** vs stack-base 11.06); pure floor still prior_stack 12.33; PRIOR_FALLBACKS reordered. |
| **15** (crown) | #11 | dual residual heads best+stack train/serve; weights **0.75/0.25** → **10.948** (equal-mean 10.985 rejected); blob `residual_heads` + `dual_residual_weights`; tests dual serve. |
| **16** (crown CR) | #11 | named dual weights by base (not position); thread-pool seal (CR serial seal); deep LGBM under dual → **10.836**; reject `prior_delta_best_stack` feat 10.955; tests weight-by-name. |
| **17** (crown) | #11 | dual residual **off** (dilutes best head); single deep `prior_best` **10.830**; `CROWN_DEEP_RESIDUAL=True`; n_refit floor 400 deep; dual infrastructure retained opt-in. |
| **18** (crown) | #11 | deep **4-seed bag** → **10.825**; n_refit p90 when deep floor≥400; reject drop-base-from-X 10.944 and LGBM lr/leaves/l2 A/B. |
| **19** (merge) | #11 | merge all-suggestions into crown single-seal path; research architecture map in CROWN_PATH. |
| **19b** (main) | landed | Full 12w reassemble: residual **11.045** / adaptive **11.040** (2 fire); pure stack **12.148** (↑ vs 12.33); selected prefers adaptive when better; `feat/all-suggestions-integration` content fully on main. |

## CodeRabbit

Fair-usage rate limits often blocked full inline review. Local max-op review substituted; unit tests green at merge.

**Open P2 (set-17 follow-ups):** GR mismatch missing-`prior_stack` edge; residual
smoothing without `well_id`; dual-serving distinct-base stress; deep `n_refit`
integration via full `train_lgbm_residual` path (constants unit-tested).

## Sealed MEASURE (this kit)

| Cohort | Selected OOF (ft) | Notes |
|--------|------------------:|-------|
| 48-well default | ~11.12 | pure prior_stack G3 reject residual |
| 64-well max-prec | ~11.35 | residual G3 accept |
| **773-well max-prec** | **~14.67** | residual G3 accept vs stack 15.11 / last_known 15.91 |
| CUDA TCN 48-well | 13.78 residual | **G3 REJECT** → pure 11.12 |
| Hunt / crown | **5 / ~6–7** | NEED_IMPROVE — private beam/DTW |

## Proceed

Public path sealed. Crown gap → private_kernel beam/DTW (gitignored), not more public LGBM alone.

## Loop (sub-5' TVT RMSE scheduled fire)

Track consecutive cycles for plateau stop (`abs(Δ selected_oof) < 0.05` ×3) and delta vs prior.

| Cycle | Date (UTC) | Branch / PR | Lever | selected_oof_ft | best_floor_ft | beam_blend_ft | last_known_ft | Δ selected | Gate3 | beats_5ft | Note |
|------:|------------|-------------|-------|----------------:|--------------:|--------------:|--------------:|-----------:|-------|-----------|------|
| 0 | seal | main | baseline seals | **14.67** (773w) | **15.11** stack | 11.924 | 10.788 | — | ACCEPT residual | false | beam lost to last_known |
| 1 | 2026-08-07 | `feat/private-beam-conf-clamp` / #14 | private beam stalk v2 | 14.67 (unchanged; no full GKF) | 15.11 (public) | **10.756** | 10.788 | 0.00 selected | n/a pilot | false | **beam blend beats last_known** (−0.032); pure beam 10.775; clamp_origin freeze + GR-rel conf + NCC reseed offline |
| 2 | 2026-08-08 | `feat/private-beam-conf-clamp` / #14 | CR fix: comparable last-known cohort | 14.67 | 15.11 | 10.756 (prior pilot) | 10.788 | 0.00 selected | n/a | false | measure script: finite last∩y∩blend∩beam mask; win-rate/delta same cohort |
| 3 | 2026-08-08 | `feat/private-beam-conf-clamp` / #14 | CR fix: drop redundant int(len) | 14.67 | 15.11 | 10.756 | 10.788 | 0.00 selected | n/a | false | Ruff RUF046; 3× abs(Δselected)<0.05 + CR cleared → **coherence plateau STOP** |
| 4 | 2026-08-08 | `feat/private-beam-conf-clamp` / #14 | re-arm: lattice Viterbi + conf-gate; CR md pipes | 14.67 | 15.11 | **10.756** | 10.788 | 0.00 selected | n/a pilot | false | conf_gate **10.749** (−0.007 vs beam, −0.040 vs last); pure lattice 10.804 reject; no GKF (gap≪0.3); plateau continues |
| 5 | 2026-08-08 | `feat/private-beam-conf-clamp` / #14 | CR fix: lattice mask + ledger honesty | 14.67 | 15.11 | **10.756** | 10.788 | 0.00 selected | n/a pilot | false | conf_gate **10.749** (Δcg ~0); pure lattice **10.903** (honest shared mask); CR cleared; no GKF; continue (CR fixed this cycle) |
| 6 | 2026-08-08 | `feat/private-beam-conf-clamp` / #14 | heel GR affine + soft conf_gate; reverse REJECT | 14.67 | 15.11 | **10.729** | 10.788 | 0.00 selected | n/a pilot | false | conf_gate **10.705** (Δcg **−0.044**); reverse_pass →11.056 REJECT; pure lat 10.686; no GKF; plateau reset (Δcg≥0.02) |
| 7 | 2026-08-08 | `feat/private-beam-conf-clamp` / #14 | CR + denser/beam-led REJECT; 64w G3 FALLBACK | 14.67 | 15.11 | **10.729** | 10.788 | 0.00 selected | FALLBACK 64w pure 12.40 | false | denser129 pure lat **10.801 REJECT**; v8 conf_gate **10.710 REJECT**; keep v6 **10.705**; CR header/loader/params; plateau no-inc (CR fixed) |
| 8 | 2026-08-08 | `feat/private-beam-conf-clamp` / #14 | CR + emit_gate REJECT; all_stack REGRESS | 14.67 | 15.11 | **10.729** | 10.788 | 0.00 selected | all_stack 15.20 ACCEPT; promotion rejected | false | emit_gate **10.738 REJECT** vs conf_gate **10.706**; keep deploy 10.705; CR B007+min_stations; plateau no-inc (CR) |
| 9 | 2026-08-08 | `feat/private-beam-conf-clamp` / #14 | CR + private beam residual feats; 16w G3 FALLBACK | 14.67 | 15.11 | **10.729** | 10.788 | 0.00 selected | 16w FALLBACK pure 11.95 | false | beam residual X wired; 8w ACCEPT 10.58 cohort; 16w res 12.02 loses stack; CR emit_conf+Gate3 wording; plateau no-inc (CR) |
| 10 | 2026-08-08 | `feat/private-beam-conf-clamp` / #14 | CR private output guard + cert kernel | 14.67 | 15.11 | **10.729** | 10.788 | 0.00 selected | 16w FALLBACK pure 11.95 hold | false | strip prior_beam from stations.csv; export strip; cert kernel_*; nan delta tests; plateau no-inc (CR) |
| 11 | 2026-08-08 | `feat/private-beam-conf-clamp` / #14 | multi-field lattice floor + pure-lat conf_gate; heatmap REJECT | 14.67 | 15.11 | **10.729** | 10.788 | 0.00 selected | n/a pilot | false | pure lat multi-field **10.544**; conf_gate **10.699** (delta_cg -0.007 KEEP); heatmap blend **15.98 REJECT**; plateau_B=1 |
| 12 | 2026-08-08 | `feat/private-beam-conf-clamp` / #14 | CR heatmap cohort + lattice residual feats | 14.67 | 15.11 | **10.729** | 10.788 | 0.00 selected | 8w ACCEPT 10.50 smoke | false | conf_gate **10.699** hold; heatmap out of deploy mask; write_stations_csv helper; 8w lattice residual ACCEPT 10.50; plateau no-inc (CR) |
| 13 | 2026-08-08 | `feat/private-beam-conf-clamp` / #14 | CR private export + 16w lattice residual FALLBACK | 14.67 | 15.11 | **10.729** | 10.788 | 0.00 selected | 16w FALLBACK pure 11.95 | false | conf_gate **10.699** hold; write_private_full default off; pred schema guard; 16w res 12.97 loses stack; 8w smoke not promoted; plateau no-inc (CR) |
| 14 | 2026-08-08 | `feat/private-beam-conf-clamp` / #14 | CR casefold export + lattice residual base REJECT | 14.67 | 15.11 | **10.729** | 10.788 | 0.00 selected | 8w lattice-base ACCEPT 10.88 worse | false | casefold priv strip; lattice_tvt base 10.88 / blend 10.87 ACCEPT but lose to prior_best-base 10.50; keep features not base; conf_gate 10.699; plateau no-inc (CR) |
| 15 | 2026-08-08 | `feat/private-beam-conf-clamp` / #14 | dual residual best+lattice KEEP micro | 14.67 | 15.11 | **10.729** | 10.788 | 0.00 selected | 8w dual ACCEPT 10.476 | false | dual --prior-col prior_best --dual-partner prior_lattice_tvt OOF **10.476** (vs single 10.502, Δ−0.026); conf_gate 10.699; not promote 8w; plateau no-inc if CR quiet |

### Cycle-1 detail (private pilot, 40 wells)

- **Prior pilot seal:** blend 11.924 / last_known 10.788 / `beats_last_known_mean=false`
- **v2 MEASURE:** blend **10.756** / pure beam **10.775** / last_known 10.788 / `beats_last_known_mean=true` / frac wells beat 0.475 / mean_conf 0.464
- **Δ blend vs prior pilot:** −1.168 ft (clamp_origin was the main pure-beam fix; relative-GR conf enables blend)
- **Not yet:** GroupKFold residual stack with beam prior as floor; hunt 5 ft still NEED_IMPROVE
- **Next lever:** beam residual feature into Gate3-honest residual head **or** lattice/DTW denser stalk; full 773w only if pure floor moves ≥0.3 ft on same cohort (pilot pure-beam gap vs last_known is 0.013; beam_blend gap is 0.032 — both below threshold)

### Cycle-2 detail (CR on PR #14)

- CodeRabbit **Major**: non-finite `last[m]` made `r_last=NaN` but `beats_last_known=False` → win-rate counted losses; aggregates mixed cohorts.
- Fix: require finite `last` (and beam) on predict mask before scoring; aggregates use shared comparable cohort only.
- Selected OOF unchanged (no GKF this cycle). Next: lattice/DTW stalk or Gate3 beam residual feature.

### Cycle-3 detail (CR polish + plateau)

- CodeRabbit Trivial: remove redundant `int(len(C))` / `int(len(R))` (RUF046).
- Major from cycle-1 review already resolved in cycle-2 (line outdated).
- Selected OOF still 14.67 (no full GKF; pilot pure gap to last_known 0.03 < 0.3 promote).
- **STOP — coherence plateau:** three consecutive cycles with `abs(Δ selected_oof) < 0.05` (all 0.00) and open CR fixes cleared. Sub-5 not sealed; resume only on new lever / user re-arm.

### Stop criteria check

- beats_5ft: **false** (selected OOF still ~14.67; pilot conf_gate **10.699**; 8w dual residual smoke **10.476**)
- Plateau B: cycle-15 lever improved 8w residual |Δ|≥0.02 vs single-base smoke → **plateau reset 0/3**; **continue**
- No open CR this cycle (casefold already landed cycle-14)

### Cycle-4 detail (re-arm lattice denser stalk)

- **Lever:** private band-lattice Viterbi (`lattice_match.py`, gitignored) + label-free conf-gate vs beam blend.
- **40w MEASURE (shared comparable cohort):**

| Prior | mean RMSE (ft) | vs last_known | vs beam_blend |
| --- | ---: | ---: | ---: |
| last_known | 10.788 | — | — |
| beam_blend (v2 seal) | **10.756** | −0.032 | — |
| lattice_blend | 10.804 | +0.015 | +0.047 **reject pure** |
| fuse conf-weight | 10.796 | +0.008 | +0.040 reject |
| **conf_gate (deploy)** | **10.749** | **−0.040** | **−0.007** |
| oracle stalk (diag only) | 10.608 | −0.180 | −0.149 (label leak; not deploy) |

- Selected OOF unchanged 14.67 (no full GKF; pure/selected move ≪ 0.3 ft promote threshold).
- CR: escape/remove raw `abs(Δselected)` pipes in loop table (cycle-3 cell broke markdown columns).
- conf_gate micro-beat not enough to promote; CR follow-ups deferred to cycle-5.

### Cycle-5 detail (CR integrity + mask)

- **Lever:** CodeRabbit cycle-4 review (≤4 fixes):
  1. `measure_private_beam_pilot.py`: include raw `prior_lattice_tvt` in shared comparable mask (Major).
  2. Ledger: keep `beam_blend_ft=10.756` in column; conf_gate in Note.
  3. Ledger: pure-beam gap vs last_known = 0.013 (blend gap 0.032) wording.
  4. Ledger: mark Markdown-pipe / prior CR as Cleared.
- **40w MEASURE (post-mask, artifacts/private_beam_pilot_v3):**

| Prior | mean RMSE (ft) | vs last_known | vs beam_blend |
| --- | ---: | ---: | ---: |
| last_known | 10.788 | — | — |
| beam_blend | **10.756** | −0.032 | — |
| pure lattice (honest) | **10.903** | +0.114 | +0.146 reject |
| lattice_blend | 10.804 | +0.015 | +0.047 reject |
| fuse | 10.796 | +0.008 | +0.040 reject |
| **conf_gate (deploy)** | **10.749** | **−0.040** | **−0.007** |
| oracle stalk (diag) | 10.608 | −0.180 | −0.149 |

- Δ conf_gate vs cycle-4 seal: ~0.000 (still 10.749); Δ selected_oof: 0.00
- No full GKF (pure/selected move ≪ 0.3 ft). **continue** (CR fixed; do not plateau-stop).
- **Next lever:** conf_gate/beam process (reseed/band/weights) toward ≤9.0 **or** Gate3 residual with beam conf_gate prior (private measure).

### Cycle-6 detail (heel GR affine KEEP + reverse REJECT)

- **Lever A (rejected):** beam `reverse_pass=True`, reversion 0.04, ncc_reseed_every=120 → **conf_gate 11.056 / beam 11.076**. Restored v2 stalk knobs.
- **Lever B (kept):** heel GR affine (`fit_heel_gr_affine`) + adaptive gr_sigma on beam/lattice; soft conf_gate (logistic + agree@2ft).
- **40w MEASURE (artifacts/private_beam_pilot_v6):**

| Prior | mean RMSE (ft) | vs last_known | vs prior seal |
| --- | ---: | ---: | ---: |
| last_known | 10.788 | — | — |
| beam_blend v2 seal | 10.756 | −0.032 | — |
| **beam_blend v6 affine** | **10.729** | **−0.060** | −0.027 vs v2 |
| pure beam v6 | 10.730 | −0.058 | — |
| pure lattice v6 | **10.686** | −0.102 | beats pure beam |
| lattice_blend v6 | 10.760 | −0.028 | — |
| fuse v6 | 10.769 | −0.020 | — |
| conf_gate v3 hard seal | 10.749 | −0.040 | — |
| **conf_gate v6 (deploy)** | **10.705** | **−0.083** | **Δcg −0.044** |
| reverse_pass trial (rej.) | 11.056 | +0.267 | — |
| oracle stalk (diag) | 10.594 | −0.195 | label leak |

- Δ conf_gate vs cycle-5: **−0.044** (≥0.02 → plateau reset); Δ selected_oof: 0.00; no open CR
- No full GKF (pure move 0.027–0.044 ≪ 0.3 promote). **continue**
- Note: pure lattice (10.686) best raw stalk; conf_gate still preferred deploy (label-free floor protect via blends).
- **Next lever:** denser lattice on affine GR / lattice-led conf_gate weights / Gate3 residual from conf_gate prior; do not re-enable reverse_pass without per-well guard.

### Cycle-7 detail (CR integrity + denser/beam-led REJECT + 64w Gate3)

- **CR (≤4 from cycle-6 review):**
  1. `docs/CROWN_PATH.md`: table header `blend RMSE` → `mean RMSE (ft)`.
  2. `private_kernel/README.md`: document `get_kernel()` + `is_private()==True` loader contract.
  3. `measure_private_beam_pilot.py`: `n_wells_scanned` + `n_wells_eligible` (not mislabeled raw).
  4. Explicit `conf_gate_params` / `lattice_n_nodes` in measure summary for seal reproducibility.
- **Lever A (REJECT):** denser lattice `n_nodes=129` → pure lattice **10.801** (vs v6 **10.686**).
- **Lever B (REJECT):** beam-led conf_gate margin0.10/lat_floor0.20 → conf_gate **10.710** (vs seal **10.705**). Artifacts: `private_beam_pilot_v8_40w`.
- **Deploy hold:** conf_gate **10.705** / beam_blend **10.729** / last_known 10.788 (v6 seal).
- **Public 64w GroupKFold** (`artifacts/max_prec_64_now`): residual OOF **12.547** vs best floor prior_stack **12.398** → Gate3 **FALLBACK_PURE_PRIOR**; selected **12.398**; beats_5ft false. Honest residual reject.
- Δ selected_oof (773w seal): 0.00; Δ deploy conf_gate: 0.00; no 773w promote (gap ≪ 0.3). Concurrent full-stack train may still be running offline — not promoted from this pilot gap.
- **Next lever:** Gate3 residual features from private conf_gate prior (export columns only) / multi-scale GR NCC stalk; avoid denser lattice without per-well adaptive nodes.

### Cycle-8 detail (CR + GR emission REJECT + all_stack audit)

- **CR (≤4 from cycle-7 review):**
  1. Drop unused `enumerate` index in measure loop (Ruff B007).
  2. Record `min_comparable_stations=20` in summary + cohort string.
- **Lever (REJECT):** `gr_emission_gate` — station-wise pick beam_blend vs lat_blend by typewell GR emission (heel-affine TW GR). 40w (`artifacts/private_beam_pilot_v9_emit`):

| Prior | mean RMSE (ft) | vs conf_gate |
| --- | ---: | ---: |
| last_known | 10.788 | — |
| beam_blend v6 | 10.729 | — |
| conf_gate deploy | **10.706** | — |
| **emit_gate trial** | **10.738** | **+0.032 REJECT** |
| pure lattice | 10.686 | — |
| oracle (diag) | 10.592 | — |

- **Deploy hold:** conf_gate ~**10.705–10.706** (v6 seal).
- **Public full 773w `max_prec_all_stack`** (prior_col=prior_stack residual base): selected residual **15.200** vs floor stack **15.760** (Gate3 ACCEPT) but **worse than sealed max_prec_all 14.67** → **do not promote**; seal remains 14.67 / best floor 15.11.
- Δ selected_oof (promoted seal): 0.00; Δ deploy conf_gate: ~0; no 773w promote from pilot.
- **Next lever:** private conf_gate / beam columns as residual **features** (not residual base alone) on ≤64w; or soft-neighbor / multi-scale NCC stalk denser only when conf drops.

### Cycle-9 detail (CR + private beam residual features)

- **CR (≤4 from cycle-8 review):**
  1. Ledger cycle-8 Gate3 cell: `ACCEPT; promotion rejected` (not conflated with promotion).
  2. measure: drop unused `emit_conf` (Ruff RUF059).
  3. private_kernel README: `is_private()` contract is **truthy** (matches `if not k.is_private()`).
- **Lever:** residual **features** from private beam (`prior_beam_tvt/conf/blend` + deltas vs stack/last/best) in `FEATURE_COLS_DEFAULT`; floors include beam pure; `train_max_precision --kernel private` + repo root on `sys.path`.
- **MEASURE:**
  - 8w private (smoke): residual **ACCEPT** OOF **10.576** vs floor stack **11.393** (beam feats present).
  - 16w private (`artifacts/max_prec_16_beam_feat`): residual OOF **12.018** loses best floor prior_stack **11.949** → Gate3 **FALLBACK_PURE_PRIOR**; selected **11.949**; beam pure floor ~12.86–12.97 (worse than stack).
- Deploy pilot conf_gate hold **10.705**; 773w seal **14.67** unchanged (no ≥0.3 promote).
- **Next lever:** conf_gate export as residual feature/floor; adaptive residual mask on beam conf; or ≤64w only when residual beats pure on multi-seed. Do not force residue.

### Cycle-10 detail (CR private output guard + cert provenance)

- **CR (≤4 from cycle-9 review):**
  1. README: private beam residual columns only when kernel_private resolution succeeds; fail-closed PublicKernel omits them.
  2. **Major:** do not publish private stalks via --out/stations.csv — write stripped stations.csv + optional local-only stations_private_full.csv; export bundle also strips prior_beam*.
  3. measure_vs_5ft.json cert includes kernel_id, kernel_private, kernel_note, RWT_KERNEL.
  4. Unit test non-finite operands → NaN deltas in attach_private_residual_deltas.
- **MEASURE hold:** 16w private residual still FALLBACK pure stack **11.95** (res ~12.02); 8w smoke ACCEPT not promoted; pilot conf_gate **10.705**; 773w seal **14.67**.
- A/B residual base default on 16w reaffirmed FALLBACK (residue never forced).
- **Next lever:** multi-seed ≤32–64w private only if residual beats pure; conf_gate column as residual feature/floor; adaptive mask on prior_beam_conf.

### Cycle-11 detail (multi-field lattice + pure-lat conf_gate; heatmap REJECT)

- **Lever A (KEEP floor):** multi-field lattice emission (GR level + shape + e_ref) → pure lattice **10.544** vs v6 pure **10.686** (Δ **−0.142**). Artifacts: private_beam_pilot_v9_mf.
- **Lever B (KEEP deploy micro):** conf_gate with pure_lattice + pure_lat_boost=0.45, margin 0.06, lat_floor 0.15 → conf_gate **10.6985** vs seal 10.7054 (Δcg **−0.007**). Artifacts: private_beam_pilot_v11_purelat.
- **Lever C (REJECT):** GR heatmap field index → pure **17.9** / blend **16.0** (far worse than last_known). Do not deploy.
- **40w comparable cohort (v11):**

| Prior | mean RMSE (ft) | vs last_known |
| --- | ---: | ---: |
| last_known | 10.788 | — |
| beam_blend v6 | 10.729 | −0.060 |
| pure multi-field lattice | **10.544** | **−0.244** |
| lattice_blend | 10.750 | −0.038 |
| conf_gate v6 seal | 10.705 | −0.083 |
| **conf_gate v11 pure-lat** | **10.699** | **−0.090** |
| heatmap blend | 15.978 | +5.19 reject |
| oracle (diag) | 10.464 | label leak |

- Selected OOF 14.67 unchanged (pure/deploy move ≪ 0.3 promote). Residue never forced.
- **Next lever:** export multi-field lattice + conf as residual features/floors on ≤32–64w private; conf_gate further lattice-led only if conf calibrated; kill heatmap until template match beats last_known.

### Cycle-12 detail (CR + multi-field lattice residual features)

- **CR (≤4 from cycle-11 review):**
  1. Ledger: delete truncated duplicate Cycle-11 table row.
  2. **Major:** deploy cohort mask no longer requires finite heatmap (diag-only); separate heatmap support mask for heat RMSEs.
  3. Shared write_stations_csv / public_station_columns in src/rwt/train.py used by train CLIs.
  4. Export strip extends to prior_lattice* / prior_heatmap*.
- **Lever:** private kernel exports multi-field prior_lattice_tvt/conf/blend + deltas in residual X and floors.
- **MEASURE:**
  - 40w pilot remeasure (private_beam_pilot_v12_cohort): conf_gate **10.699** hold; pure multi-field lattice **10.544**; heatmap still REJECT (~16 blend) on separate mask.
  - 8w private residual smoke (max_prec_8_lattice_feat): residual **ACCEPT** OOF **10.502** vs floor stack **11.393** (beam-only smoke was 10.576) — cohort-only; not promoted.
- Selected 773w seal **14.67** unchanged. Residue never forced.
- **Next lever:** ≤16–32w multi-seed private lattice residual if 8w holds; adaptive mask on lattice conf; do not promote smoke.

### Cycle-13 detail (CR private export + 16w lattice residual)

- **CR (≤4 from cycle-12 review):**
  1. measure: delta_heatmap_vs_conf_gate uses heatmap cohort H for both terms.
  2. **Major:** write_private_full defaults **False**; train_max_precision requires --write-private-stations opt-in.
  3. export_competition_bundle: validate --pred schema; reject proprietary column prefixes before write.
  4. train CLI write_stations_csv(..., write_private_full=False).
- **Lever MEASURE:** 16w private multi-field lattice residual features (max_prec_16_lattice_feat): residual OOF **12.968** loses best floor prior_stack **11.949** → Gate3 **FALLBACK_PURE_PRIOR**; selected **11.949**. 8w smoke ACCEPT 10.50 not promoted (cohort-only).
- Pilot conf_gate hold **10.699**; pure multi-field lattice **10.544**; 773w seal **14.67**. Residue never forced.
- **Next lever:** pure multi-field lattice as residual **base** (not only feature) on ≤16w if Gate3 honest; or conf-adaptive residual mask; avoid full 773w promote.

### Cycle-14 detail (CR casefold + lattice residual base)

- **CR:** export/features proprietary-prefix checks use `str.casefold()` so mixed-case headers (e.g. `Prior_Beam_TVT`) are rejected; same for `public_station_columns`.
- **Lever:** residual **base** = multi-field lattice (not only residual X features).
  - 8w `prior_lattice_tvt` base: residual **ACCEPT** OOF **10.882** vs stack floor **11.393** but **worse** than prior_best-base smoke **10.502** → **REJECT as default base**.
  - 8w `prior_lattice_blend` base: residual **ACCEPT** OOF **~10.865** (log) — still worse than prior_best base.
  - 16w lattice-as-feature remains FALLBACK pure 11.95 (cycle-13).
- Keep lattice as residual **features/floors**; residual base stays `prior_best` / public defaults.
- Pilot conf_gate **10.699**; pure multi-field lattice floor **10.544**; 773w seal **14.67**. Residue never forced.
- **Next lever:** conf-adaptive residual mask on lattice conf; or soft-ensemble residual heads (best + lattice) only if OOF improves on ≥16w multi-seed.

### Cycle-15 detail (dual residual prior_best + lattice)

- **Lever:** CLI `--dual-partner prior_lattice_tvt` with primary `prior_best` (weights 0.75/0.25). API: `train_lgbm_residual(..., dual_partner=...)`.
- **MEASURE 8w private** (`artifacts/max_prec_8_dual_lat`): residual **ACCEPT** OOF **10.476** vs floor stack **11.393** (Δ **−0.026** vs single prior_best lattice-feat smoke **10.502**). Gate3 honest ACCEPT.
- Not promoted: 8w cohort only; 16w lattice-feature residual still FALLBACK pure 11.95 (cycle-13).
- Pilot conf_gate **10.699**; pure multi-field lattice **10.544**; 773w seal **14.67**. Residue never forced.
- **Next lever:** dual residual on 16w multi-seed; if ACCEPT and improves ≥0.3 vs pure, consider wider pilot. Adaptive lattice-conf mask secondary.

