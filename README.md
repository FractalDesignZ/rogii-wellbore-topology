# rogii-wellbore-topology

An open-core research harness for [ROGII Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction), with abstraction-mode compute from [`topology-sees-sequence`](https://github.com/ZuluYokohama/topology-sees-sequence) and an optional boundary to a separately installed Network Engine.

**Original work is a project requirement.** External solutions provide context for typical approaches and published results. Candidate designs, features and weights require traceable first-party origins. [Repository rule](AGENTS.md)

**Validation status.** Earlier spatial experiments selected full training `TVT` before splitting wells into folds. Their historical scores require remeasurement. The corrected default uses observed `TVT_input` and supports unseen target geometry. The current development comparisons below isolate supervised donors by training fold. The optional native Network Engine checks measure graph structure.

**Development measurements.** Cross-fitted member correction followed by affine gamma-ray alignment reached **10.030446 ft RMSE** on 619 development wells and 3,034,427 original toe stations. A Julia recurrence ordered by measured evidence reached **10.027222 ft** on the same rows, with a paired interval crossing zero against the affine baseline. These offline comparisons use the complete recorded lateral. The 154 confirmation wells are sealed. [Measurements, controls and provenance](docs/MASTER_ITERATION.md#combining-member-correction-with-observed-gamma-ray)

**Run the shared nodal comparison.** Follow a reading from its original CSV row through a revised edge weight to four predictions on the same stations. The controller records the Python AST, source dependencies, worker runs and artifact reads, then recomputes each model's error from the raw depth labels. [Run and verify](docs/NODAL_HARNESS.md) · [Typed observations and neighbor axes](docs/NODAL_HARNESS.md#typed-observations-and-neighbor-axes) · [Relational methods](docs/RELATIONAL_METHODS.md) · [Jovin source and native checks](docs/JOVIN_ENROLLMENT.md)

**Reference-to-source graph.** The six supplied mathematical and ML PDFs are cross-referenced to process decisions, Python definitions and checks. The verifier binds source and AST identities and keeps the proposed persistence tracker and nested selector explicit. [Process and control graph](docs/METHOD_REFERENCE_GRAPH.md)

| Abstraction mode | Role |
|------------------|------|
| **M0 RESTRICT** | TVT RMSE · GroupKFold-by-well · no production OPEN |
| **M1 GEOM** | Open-curve tangents, curvature, Bishop-style transport (TSS lineage) |
| **M2 PRIOR** | Heel-only flat-formation / last-TVT priors (E_ref-style: prior, not forced) |
| **M3 MATCH** | Multi-scale GR ↔ typewell NCC |
| **M4 MIDDLE_OUT** | Known section vs masked toe |
| **M5 RESIDUAL** | Trees learn correction to prior only |
| **M6 CERTIFY** | CV + baselines + artifact trail |

Full thesis: [`docs/ABSTRACTION_MODES.md`](docs/ABSTRACTION_MODES.md) · entry blurb: [`docs/KAGGLE_ENTRY.md`](docs/KAGGLE_ENTRY.md)

**NotebookLM / RAG knowledge base (truth-audited):**  
[`docs/KB_TOPOLOGICAL_WELLBORE_RWT.md`](docs/KB_TOPOLOGICAL_WELLBORE_RWT.md) — Volume II map, three traps, three-layer stack, Gate3, and historical full-cohort ledger (requires remeasurement under the corrected spatial policy).

## Competition law (short)

- **Target:** TVT (stratigraphic position), not Z alone  
- **Metric:** RMSE(dTVT)  
- **CV:** GroupKFold by well  
- **Trap:** global TVT↔Z ≠ within-lateral slope  
- **Clamp / Gate3:** residual only if it beats best pure prior floor *and* last_known  

## Layout

```text
src/rwt/           # executable modes + train/submit
scripts/           # download, probe, eda, late-entry, dashboard, parallel train
tests/
docs/              # KB, ABSTRACTION_MODES, LINEAGE_TSS, PROTOCOL, KAGGLE_ENTRY
data/raw/          # Kaggle download (gitignored)
artifacts/         # CV + certificates (gitignored contents ok to regenerate)
```

## Network Engine integration

The first integration exports raw `MD`, `Z`, and `GR`, observation masks, declared depth units, and provenance for offline relational diagnostics. It preserves acquisition order and missing support, converts depth to metres, and uses MD as the analysis axis. The container's sample ordinal is not physical time. Target labels, priors, and private/model features are excluded.

```bash
# Public preparation path; no Network Engine installation required
python scripts/measure_network_dynamics.py --synthetic

# One raw lateral; declare its actual depth unit
python scripts/measure_network_dynamics.py --csv path/to/lateral.csv --well-id WELL_ID --depth-unit ft

# Optional native slice diagnostics with a separately licensed local engine
python scripts/measure_network_dynamics.py --synthetic --run-local
```

The synthetic fixture is practice-only and measured in metres. Each run creates a fresh artifact directory with input hashes and a provenance seal. Native runs add diagnostic outputs and a sealed receipt; legacy windowed dynamics stays disabled. Diagnostics do not yet change TVT predictions or Gate3 decisions.

The public neighbor graph now reads observed `TVT_input` only by default. Full donor labels require explicit opt-in and fold-local donor visibility before GroupKFold use. Integration details, current native export limitations, and the validation path to dynamic prior gating: [`docs/NETWORK_ENGINE_INTEGRATION.md`](docs/NETWORK_ENGINE_INTEGRATION.md).

### Master iteration and image relationships

The public relation search overlays multiple maps on each observed transition and traces which candidate depths survive. Run `python scripts/demo_relation_search.py --phase-stack` to see complementary periodic readings resolve one of 30 depths. With a local Network Engine installation, add `--native-lift` to turn admitted transitions into relation nodes. The 12-well pilot records which maps helped resolve a depth and which exhausted the available paths. [Shared-graph search and measurements](docs/RELATION_SEARCH.md)

Use `rwt.offset_path.relation_cost` to weight evidence separately for each candidate crossing. `minimum_cost_offset_path` selects the least-cost reachable trajectory through those crossings. [Worked example with observation-dependent weights](docs/RELATION_SEARCH.md#weighting-relationships-inside-the-search)

The next iteration uses two linked tracks: rapid model/alignment candidates and relational operator/image candidates, compared on the same fixed development cohort. The controller retains static AST/import evidence, a dependency graph, bounded subprocess receipts, exact selected OOF predictions, and a separate metric audit. Confirmation wells remain outside candidate selection. See [`docs/MASTER_ITERATION.md`](docs/MASTER_ITERATION.md).

The standard BASE92 route is available as an [experimental fit/predict interface](docs/EXPERIMENTAL_BLOCK_MODEL.md). It requires an explicit development/confirmation manifest, stores training medians and a versioned LightGBM text bundle, and retains exact grouped-OOF evidence. It preserves the visible heel and uses no residual output clamp.

```bash
python -m rwt.block_model fit --data-root data/raw/example --manifest cohort.json --out artifacts/block_experiment
python -m rwt.block_model predict --model artifacts/block_experiment/model --lateral target.csv --typewell reference.csv --well-id TARGET --out artifacts/target_predictions.csv
```

Supplied well figures are retained as paired views. The image ingress can extract a declared path-color region and register projected XYZ to pixel coordinates using explicit control points. Full-TVT and formation-top panels are training/evaluator data; supplied PNGs are training-only. Neither a fixed viewing angle nor an image label's unit is silently assumed.

```bash
pip install -e ".[images]"
python scripts/inspect_registered_views.py --image path/to/well.png --lateral path/to/well.csv --depth-unit ft --azimuth-deg -74.5 --path-roi 350 1650 2500 2535 --out artifacts/image_attempt_001
```

The ROI above is an example requiring verification for the supplied image. Without explicit control points the result is unregistered pixel candidates, not recovered geometry or a TVT prediction.

Registration design and the checked development-image correspondence are documented in [`docs/REGISTERED_IMAGE_VIEWS.md`](docs/REGISTERED_IMAGE_VIEWS.md).

## Quick start

```bash
cd rogii-wellbore-topology
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
pip install -e .

# unit tests
python -m unittest discover -s tests -v

# full driver (downloads if credentials exist; else synthetic demo)
python scripts/run_late_entry.py

# or step-by-step
python scripts/download_kaggle.py          # needs kaggle.json + rules accepted
python scripts/probe_data.py
python scripts/eda_report.py --data data/raw

# multi-core full train + live progress bar dashboard
powershell -ExecutionPolicy Bypass -File scripts/launch_parallel_train.ps1
# → http://127.0.0.1:8765/  (bar-only refresh; details in pop-outs)
```

### Kaggle credentials

1. Accept competition rules in the browser.  
2. Place API token at `%USERPROFILE%\.kaggle\kaggle.json` (or set `KAGGLE_USERNAME` / `KAGGLE_KEY`).  
3. `pip install kaggle`  
4. `python scripts/download_kaggle.py`

## Status

All numerical results in this table are historical and require remeasurement under the corrected spatial observation policy.

| Phase | Status |
|-------|--------|
| Abstraction-mode stack + TSS open-curve geometry | **landed** |
| PF stack + clamps + Gate3 best pure floor | **landed** (PR#3–#4) |
| Multi-process assemble + live dashboard | **landed** (PR#5) |
| Historical 773-well OOF (public kernel) | residual **14.54** · stack **15.11** · last_known **15.91** · historical Gate3 ACCEPT; remeasurement required |
| Multi-neighbor spatial + residual interactions + adaptive Gate3 | **landed** (PR#13) — synthetic seal confirmed |
| Spatial observation boundary | Observed `TVT_input` donors by default; unseen targets supported; hidden-label mutation tests |
| Network Engine boundary | Raw measurement export + optional native slice diagnostics; no TVT improvement claim — [`integration guide`](docs/NETWORK_ENGINE_INTEGRATION.md) |
| Task construct (GR@TVT) | geology = typewell book; place laterals; gated residual — [`docs/TASK_GR_TVT_CONSTRUCT.md`](docs/TASK_GR_TVT_CONSTRUCT.md) |
| All-context two-step | geometry path then GR@TVT placement — [`docs/ALL_CONTEXT_OPERATOR.md`](docs/ALL_CONTEXT_OPERATOR.md) |
| Real field slices (773w) | pad XY / section / GR·TVT matrices / heel-toe — `scripts/slice_real_field.py` → `artifacts/field_slices_full` |
| Adaptive residual serve (name of the game) | OOF-picked conf channel (`pf` / `gr_agree` / product) · thr×lifetime · `adaptive_serve` only if beats residual |
| Exp C GR-agree conf (12w) | residual **10.741** holds; adaptive still loses (best pf 11.10) — seal `artifacts/exp_c_gr_agree_12w` |
| Exp D private lattice (local) | **12w residual 10.471** ACCEPT; **40w pure stack 11.09** (residual REJECT all bases) |
| Gate3 floor honesty | sparse-NaN pure floors no longer dropped — residual must beat **all** floors incl. stack |
| Auto residual base | default base = best pure floor on labeled toe (`prior_stack` typical); override with `--prior-col` |
| Adaptive serve fields in CV report | `adaptive_oof_rmse` · `adaptive_conf_channel` · thr · lifetime · channel sweep |
| GitHub remote | `FractalDesignZ/rogii-wellbore-topology` |
| RAG / NotebookLM pack | [`docs/KB_TOPOLOGICAL_WELLBORE_RWT.md`](docs/KB_TOPOLOGICAL_WELLBORE_RWT.md) |
| Kaggle submit | **OFF** — measure in-repo only (OOF / Gate3 / seals); no competition upload under RESTRICT |

## Attribution

- Geometry: `topology-sees-sequence` holonomy transport (open-curve adaptation)  
- Competition: ROGII × Kaggle  
- Domain features (NCC, tortuosity proxies): standard geosteering practice + public discussion  

## IP / open-core

| Layer | Where | License |
|-------|--------|---------|
| **Public interface** | this repo (`src/rwt`, scripts, tests) | **AGPLv3-or-later** (+ dual-license option) |
| **Proprietary kernel** | `private_kernel/` (gitignored) or private repo | Closed — never pushed here |
| **Competition export** | `scripts/export_competition_bundle.py` | Predictions + seal only |

Governance: [`docs/IP_GOVERNANCE.md`](docs/IP_GOVERNANCE.md) · `LICENSE` · `NOTICE`  
Provenance seals: `rwt.provenance` · watermark: `rwt.watermark` · kernel switch: `RWT_KERNEL=public|private`

```bash
# competition isolation — never ships private_kernel
python scripts/export_competition_bundle.py --pred submissions/raw_preds.csv --out submissions/competition_bundle
```

**Not legal advice.** Patents / commercial contracts → IP attorney.

## License

**AGPL-3.0-or-later** for the public tree. Commercial dual-license available from the copyright holder. Attribute TSS lineage.
