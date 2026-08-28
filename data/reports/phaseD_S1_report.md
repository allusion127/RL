# PosValNet ensemble evaluation report

- members: 5  
- torch: 2.11.0+cu128  device: cpu  
- calibration: yes  
- params/member: 1,595,219  

## Acceptance verdicts (plan sec. 4.4)

| Criterion | Value | Verdict |
|---|---|---|
| S1 cyclen R² ≥ 0.98 & ≥ trees | R²=0.0798, tree -0.246 | **FAILED** |
| S1 cbc_max R² ≥ 0.98 & ≥ trees | R²=-18.9332, tree -43.662 | **FAILED** |
| S1 f_r within-case Spearman ≥ trees | CNN=0.8978 tree=0.4595 | **PASS** |
| S1 f_q within-case Spearman ≥ trees | CNN=0.8909 tree=0.3759 | **PASS** |
| S2 interpolation functional | n_val=0 | **FAILED** |
| S4 interpolation functional | n_val=0 | **FAILED** |

## S1  (status=ok, n_val=6185)

| Target | n | MAE | RMSE | R² (ens) | R² (mbr) | Spearman(case) | n_cases | tree R² | tree Sp |
|---|---|---|---|---|---|---|---|---|---|
| f_r | 6182 | 0.0833 | 0.1116 | 0.8624 | 0.8353 | 0.8978±0.054 | 12 | 0.3795 | 0.4595 |
| f_q | 6182 | 0.1193 | 0.1672 | 0.8563 | 0.8269 | 0.8909±0.059 | 12 | 0.3531 | 0.3759 |
| cbc_max | 633 | 199.2526 | 204.4154 | -18.9332 | -19.6915 | 0.6911±0.000 | 1 | -43.6616 | 0.6529 |
| cyclen | 6182 | 6.0694 | 14.7087 | 0.0798 | 0.0684 | 0.7691±0.360 | 12 | -0.2463 | 0.4126 |
| ao_abs | 6182 | 0.0083 | 0.0131 | 0.3210 | 0.2990 | 0.7938±0.244 | 12 | 0.3678 | 0.1888 |
| discharge_burnup | 6180 | 0.2608 | 0.5106 | 0.6321 | 0.6214 | 0.7715±0.345 | 12 | n/a | n/a |
| max_pin_burnup | 6180 | 0.9594 | 1.1571 | 0.6472 | 0.5952 | 0.7082±0.184 | 12 | n/a | n/a |

- cyclen risk-coverage (coverage:MAE, σ-sorted): 0.2:1.60, 0.4:1.59, 0.6:1.59, 0.8:1.59, 1.0:6.07
