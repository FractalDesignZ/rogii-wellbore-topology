# Kaggle late-entry write-up (draft)

## Title ideas

- *Middle-out TVT: open-curve holonomy priors from topology-sees-sequence*
- *Abstraction modes for geosteering prediction (geometry → prior → match → residual)*

## Short description (paste into discussion / notebook)

This late entry applies computational discipline from our topology-sees-sequence work (SE(3)/Bishop-style frame transport, middle-out priors, hard domain reduction) to ROGII wellbore TVT prediction.

We do **not** claim peptide geometry predicts shale. We claim that **ordering the compute**:

1. open-curve trajectory geometry,  
2. reference priors from the known heel only,  
3. multi-scale GR↔typewell match,  
4. residual learning under GroupKFold-by-well  

is a clearer, more corrigible stack than a flat feature dump — and still produces a real submission.

Metric remains **RMSE(dTVT)**. Certificate trail: `artifacts/late_entry_certificate.json`, CV reports, baseline tables.

## Links

- Code: this repo  
- Lineage: https://github.com/ZuluYokohama/topology-sees-sequence  
- Competition: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction  
