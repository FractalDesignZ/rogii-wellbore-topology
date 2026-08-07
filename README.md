# rogii-wellbore-topology

**Late Kaggle entry** for [ROGII Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction) — framed as a demonstration of **abstraction-mode compute** from [`topology-sees-sequence`](https://github.com/ZuluYokohama/topology-sees-sequence), not a generic “dump features into LGBM” bakeoff.

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
[`docs/KB_TOPOLOGICAL_WELLBORE_RWT.md`](docs/KB_TOPOLOGICAL_WELLBORE_RWT.md) — Volume II map, three traps, three-layer stack, Gate3, full-cohort ledger (773w residual OOF **14.54** vs last_known **15.91**).

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

| Phase | Status |
|-------|--------|
| Abstraction-mode stack + TSS open-curve geometry | **landed** |
| PF stack + clamps + Gate3 best pure floor | **landed** (PR#3–#4) |
| Multi-process assemble + live dashboard | **landed** (PR#5) |
| Full 773-well OOF (public kernel) | residual **14.54** · stack **15.11** · last_known **15.91** · Gate3 ACCEPT |
| GitHub remote | `ZuluYokohama/rogii-wellbore-topology` |
| RAG / NotebookLM pack | [`docs/KB_TOPOLOGICAL_WELLBORE_RWT.md`](docs/KB_TOPOLOGICAL_WELLBORE_RWT.md) |
| First scored Kaggle submit | human upload only — OOF mid-teens; submit MEASURES, does not close LB ~6–7 |

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
