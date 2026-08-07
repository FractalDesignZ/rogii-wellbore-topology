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

## Competition law (short)

- **Target:** TVT (stratigraphic position), not Z alone  
- **Metric:** RMSE(dTVT)  
- **CV:** GroupKFold by well  
- **Trap:** global TVT↔Z ≠ within-lateral slope  

## Layout

```text
src/rwt/           # executable modes + train/submit
scripts/           # download, probe, eda, late-entry driver, smoke
tests/
docs/              # ABSTRACTION_MODES, LINEAGE_TSS, PROTOCOL, KAGGLE_ENTRY
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
python -m rwt.train --data data/raw --out artifacts/late_real
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
| GitHub remote | push target `ZuluYokohama/rogii-wellbore-topology` |
| Kaggle data | **NEED_INFO** until token + rules |
| First scored submit | after data + schema align in `submit.py` |

## Attribution

- Geometry: `topology-sees-sequence` holonomy transport (open-curve adaptation)  
- Competition: ROGII × Kaggle  
- Domain features (NCC, tortuosity proxies): standard geosteering practice + public discussion  

## License

Research / competition entry — attribute TSS lineage.
