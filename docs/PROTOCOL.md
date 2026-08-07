# Late-entry protocol (GO_MEASURE style)

## RESTRICT

- Goal: competitive TVT RMSE on masked laterals  
- Non-goals: production OPEN claims; peptide science theatre  
- Success: GroupKFold CV with honest well groups; beat flat/TVT≈−Z baselines  

## PLAN

1. Load train wells + typewells  
2. Middle-out split at prediction start (PS)  
3. Features: open-curve geometry + multi-scale GR↔typewell match + tortuosity  
4. LGBM residual on ΔTVT vs structural prior  
5. Submit only after CV logged  

## MEASURE

- RMSE overall + per-well quantiles  
- Label: never report public LB alone as truth  

## AUDIT

- No station-level leakage across well groups  
- No future GR from mask if competition rules forbid (use only allowed columns)  

## Residue

- Data not yet in tree  
- Offset-well spatial prior deferred to v1  
