# PosValNet ensemble evaluation report

- members: 5  
- torch: 2.11.0+cu128  device: cpu  
- calibration: yes  
- params/member: 1,594,191  

## Acceptance verdicts (plan sec. 4.4)

| Criterion | Value | Verdict |
|---|---|---|
| S0 cyclen R² ≥ 0.98 & ≥ trees | R²=0.6365, tree 0.978 | **FAILED** |
| S1 cyclen R² ≥ 0.98 & ≥ trees | R²=-0.6848, tree -0.246 | **FAILED** |
| S2 cyclen R² ≥ 0.98 & ≥ trees | R²=0.9997, tree 0.936 | **PASS** |
| S4 cyclen R² ≥ 0.98 & ≥ trees | R²=0.9870, tree 0.149 | **PASS** |
| S0 cbc_max R² ≥ 0.98 & ≥ trees | R²=0.9966, tree 0.979 | **PASS** |
| S1 cbc_max R² ≥ 0.98 & ≥ trees | R²=-0.2948, tree -43.662 | **FAILED** |
| S2 cbc_max R² ≥ 0.98 & ≥ trees | R²=0.9900, tree -0.905 | **PASS** |
| S4 cbc_max R² ≥ 0.98 & ≥ trees | R²=0.9986, tree 0.852 | **PASS** |
| S1 f_r within-case Spearman ≥ trees | CNN=0.9067 tree=0.4595 | **PASS** |
| S1 f_q within-case Spearman ≥ trees | CNN=0.9069 tree=0.3759 | **PASS** |
| S2 f_r within-case Spearman ≥ trees | CNN=0.9971 tree=0.7627 | **PASS** |
| S2 f_q within-case Spearman ≥ trees | CNN=0.9974 tree=0.7025 | **PASS** |
| S2 interpolation functional | n_val=7629 | **PASS** |
| S4 interpolation functional | n_val=4420 | **PASS** |

## S0  (status=ok, n_val=3893)

| Target | n | MAE | RMSE | R² (ens) | R² (mbr) | Spearman(case) | n_cases | tree R² | tree Sp |
|---|---|---|---|---|---|---|---|---|---|
| f_r | 3893 | 0.0181 | 0.0446 | 0.9670 | 0.9608 | 0.9791±0.035 | 18 | 0.8115 | 0.9124 |
| f_q | 3893 | 0.0262 | 0.0668 | 0.9648 | 0.9575 | 0.9795±0.033 | 18 | 0.8025 | 0.9091 |
| cbc_max | 2668 | 3.3079 | 7.9957 | 0.9966 | 0.9945 | 0.9747±0.062 | 12 | 0.9794 | 0.9219 |
| cyclen | 3893 | 1.3623 | 7.9700 | 0.6365 | 0.6346 | 0.9139±0.232 | 18 | 0.9783 | 0.8925 |
| ao_abs | 3893 | 0.0016 | 0.0059 | 0.7987 | 0.7933 | 0.9090±0.207 | 18 | 0.8990 | 0.8592 |

- cyclen risk-coverage (coverage:MAE, σ-sorted): 0.2:0.15, 0.4:0.17, 0.6:0.19, 0.8:0.23, 1.0:1.36

## S1  (status=ok, n_val=6185)

| Target | n | MAE | RMSE | R² (ens) | R² (mbr) | Spearman(case) | n_cases | tree R² | tree Sp |
|---|---|---|---|---|---|---|---|---|---|
| f_r | 6182 | 0.0756 | 0.1084 | 0.8702 | 0.8536 | 0.9067±0.051 | 12 | 0.3795 | 0.4595 |
| f_q | 6182 | 0.1080 | 0.1637 | 0.8622 | 0.8416 | 0.9069±0.047 | 12 | 0.3531 | 0.3759 |
| cbc_max | 633 | 37.8119 | 52.0994 | -0.2948 | -0.7338 | 0.6992±0.000 | 1 | -43.6616 | 0.6529 |
| cyclen | 6182 | 7.5117 | 19.9026 | -0.6848 | -0.6902 | 0.8129±0.299 | 12 | -0.2463 | 0.4126 |
| ao_abs | 6182 | 0.0084 | 0.0147 | 0.1549 | 0.1409 | 0.8156±0.243 | 12 | 0.3678 | 0.1888 |

- cyclen risk-coverage (coverage:MAE, σ-sorted): 0.2:1.22, 0.4:1.22, 0.6:1.25, 0.8:1.30, 1.0:7.51

## S2  (status=ok, n_val=7629)

| Target | n | MAE | RMSE | R² (ens) | R² (mbr) | Spearman(case) | n_cases | tree R² | tree Sp |
|---|---|---|---|---|---|---|---|---|---|
| f_r | 7629 | 0.0076 | 0.0117 | 0.9967 | 0.9936 | 0.9971±0.000 | 2 | -0.1122 | 0.7627 |
| f_q | 7629 | 0.0111 | 0.0163 | 0.9971 | 0.9941 | 0.9974±0.000 | 2 | -0.0590 | 0.7025 |
| cbc_max | 4390 | 2.5433 | 3.4577 | 0.9900 | 0.9784 | 0.9940±0.000 | 1 | -0.9052 | 0.8821 |
| cyclen | 7629 | 0.2065 | 0.2908 | 0.9997 | 0.9991 | 0.9974±0.000 | 2 | 0.9355 | 0.5903 |
| ao_abs | 7629 | 0.0004 | 0.0006 | 0.9974 | 0.9934 | 0.9976±0.001 | 2 | 0.4097 | 0.3395 |

- cyclen risk-coverage (coverage:MAE, σ-sorted): 0.2:0.16, 0.4:0.17, 0.6:0.18, 0.8:0.19, 1.0:0.21

## S4  (status=ok, n_val=4420)

| Target | n | MAE | RMSE | R² (ens) | R² (mbr) | Spearman(case) | n_cases | tree R² | tree Sp |
|---|---|---|---|---|---|---|---|---|---|
| f_r | 4420 | 0.0238 | 0.0488 | 0.9705 | 0.9654 | 0.9833±0.026 | 14 | -0.7683 | 0.5644 |
| f_q | 4420 | 0.0337 | 0.0730 | 0.9675 | 0.9624 | 0.9829±0.025 | 14 | -0.5442 | 0.5906 |
| cbc_max | 715 | 3.7179 | 4.6038 | 0.9986 | 0.9964 | 0.9948±0.003 | 9 | 0.8521 | 0.3443 |
| cyclen | 4420 | 0.6316 | 1.1562 | 0.9870 | 0.9839 | 0.9666±0.037 | 14 | 0.1489 | 0.3403 |
| ao_abs | 4420 | 0.0022 | 0.0041 | 0.9209 | 0.9142 | 0.9428±0.078 | 14 | -2.1660 | 0.1166 |

- cyclen risk-coverage (coverage:MAE, σ-sorted): 0.2:0.12, 0.4:0.14, 0.6:0.23, 0.8:0.42, 1.0:0.63
