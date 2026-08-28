# v5 integrated A/B — decision table

| arm | target | within-cell Spearman | calibrated MAE | P@8 | n cells |
|---|---|--:|--:|--:|--:|
| v4_baseline | cyclen | 0.7798 | 5.8463 | 0.6562 | 36 |
| v4_baseline | f_r | 0.9042 | 0.1434 | 0.7535 | 36 |
| v5_full | cyclen | 0.7519 | 5.2058 | 0.5972 | 36 |
| v5_full | f_r | 0.8989 | 0.1450 | 0.7569 | 36 |
| v5_minus_shape | cyclen | 0.7513 | 6.5650 | 0.6215 | 36 |
| v5_minus_shape | f_r | 0.8984 | 0.1414 | 0.7465 | 36 |
| v5_distill | cyclen | 0.7553 | 5.0316 | 0.6424 | 36 |
| v5_distill | f_r | 0.8972 | 0.1410 | 0.7431 | 36 |
| v5_distill_w160 | cyclen | 0.7666 | 5.2154 | 0.6354 | 36 |
| v5_distill_w160 | f_r | 0.9016 | 0.1402 | 0.7535 | 36 |

## legacy high-cyclen tail (vs incumbent champion)

| arm | pass | worst band MAE increase [EFPD] |
|---|---|--:|
| v4_baseline | True | 0.5420 |
| v5_full | True | 0.0863 |
| v5_minus_shape | True | 0.7581 |
| v5_distill | True | 0.1308 |
| v5_distill_w160 | True | 0.1946 |