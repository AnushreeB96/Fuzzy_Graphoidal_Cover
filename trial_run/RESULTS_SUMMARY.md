# Results summary (top-200 network, 2 seeds, B=20)

Objectives frozen: f1=k, f2=max_r R(L(P_r)), f3=0.7*mean u+0.3*Q90(u), f4=-mean confidence.

## Table 1 - main comparison (mean ± SD over seeds)
| Method                  | Paths ↓    | Worst cost ↓   | Confidence ↑    | Uncertainty ↓   | Runtime (s) ↓   | ARI ±5% ↑     | ARI ±10% ↑    | ARI ±20% ↑    |
|:------------------------|:-----------|:---------------|:----------------|:----------------|:----------------|:--------------|:--------------|:--------------|
| Minimum-cardinality     | 87.0 ± 0.0 | 2.975 ± 0.847  | 0.4073 ± 0.0016 | 0.0980 ± 0.0016 | 0.56 ± 0.01     | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.988 ± 0.018 |
| Minimum worst-cost      | 94.0 ± 0.0 | 1.801 ± 0.001  | 0.4064 ± 0.0008 | 0.0979 ± 0.0008 | 0.56 ± 0.01     | 0.768 ± 0.078 | 0.552 ± 0.082 | 0.482 ± 0.010 |
| Proposed Pareto         | 89.0 ± 0.0 | 2.658 ± 0.215  | 0.4089 ± 0.0002 | 0.0973 ± 0.0009 | 0.56 ± 0.01     | 0.585 ± 0.031 | 0.452 ± 0.025 | 0.423 ± 0.039 |
| Robustness-aware Pareto | 89.5 ± 0.7 | 2.565 ± 0.347  | 0.4098 ± 0.0010 | 0.0978 ± 0.0002 | 0.57 ± 0.01     | 0.570 ± 0.153 | 0.429 ± 0.024 | 0.376 ± 0.006 |

## Table 2 - preference sensitivity
| Preference           | lambda (paths,cost,unc,-conf)   | Paths      | Worst cost    | Uncertainty     | Confidence      |
|:---------------------|:--------------------------------|:-----------|:--------------|:----------------|:----------------|
| Compactness-oriented | (0.5, 0.2, 0.15, 0.15)          | 88.5 ± 0.7 | 2.873 ± 0.520 | 0.0974 ± 0.0009 | 0.4092 ± 0.0006 |
| Balanced             | (0.25, 0.25, 0.25, 0.25)        | 89.0 ± 0.0 | 2.658 ± 0.215 | 0.0973 ± 0.0009 | 0.4089 ± 0.0002 |
| Uncertainty-oriented | (0.15, 0.2, 0.45, 0.2)          | 89.5 ± 0.7 | 2.846 ± 0.482 | 0.0971 ± 0.0006 | 0.4094 ± 0.0009 |

## Table 3 - robustness (secondary analysis)
| Method                  | ARI ±5%       | ARI ±10%      | ARI ±20%      | Tie-aware ARI ±5%   | Tie-aware ARI ±10%   | Tie-aware ARI ±20%   | Δk ±10%     | Cost change ±10%   | Equally optimal covers (nominal)   | Non-dominated frac ±10%   |
|:------------------------|:--------------|:--------------|:--------------|:--------------------|:---------------------|:---------------------|:------------|:-------------------|:-----------------------------------|:--------------------------|
| Minimum-cardinality     | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.988 ± 0.018 | 1.000 ± 0.000       | 1.000 ± 0.000        | 1.000 ± 0.000        | 0.00 ± 0.00 | 0.015 ± 0.006      | 45.0 ± 1.4                         | 1.00 ± 0.00               |
| Minimum worst-cost      | 0.768 ± 0.078 | 0.552 ± 0.082 | 0.482 ± 0.010 | 0.781 ± 0.097       | 0.626 ± 0.023        | 0.552 ± 0.109        | 1.80 ± 1.63 | 0.018 ± 0.002      | 2.5 ± 2.1                          | 0.55 ± 0.14               |
| Proposed Pareto         | 0.585 ± 0.031 | 0.452 ± 0.025 | 0.423 ± 0.039 | 0.585 ± 0.031       | 0.452 ± 0.025        | 0.423 ± 0.039        | 1.25 ± 0.49 | 0.135 ± 0.041      | 1.0 ± 0.0                          | 0.60 ± 0.14               |
| Robustness-aware Pareto | 0.570 ± 0.153 | 0.429 ± 0.024 | 0.376 ± 0.006 | 0.570 ± 0.153       | 0.429 ± 0.024        | 0.376 ± 0.006        | 1.35 ± 0.14 | 0.156 ± 0.007      | 1.0 ± 0.0                          | 0.82 ± 0.18               |

## Table 4 - scalability (target selection vs actual size)
| Target selection   |   Actual genes |   Edges |   Candidate budget |   Candidates (distinct) | Budget cap reached   |   Pareto solutions |   Runtime (s) |   CPU mem (MB) |   GPU mem (MB) |   Proposed cover size |   Min-cardinality cover size |
|:-------------------|---------------:|--------:|-------------------:|------------------------:|:---------------------|-------------------:|--------------:|---------------:|---------------:|----------------------:|-----------------------------:|
| Top-50             |             25 |      20 |                500 |                     130 | False                |                 22 |        1.538  |           1470 |          253   |                    13 |                           13 |
| Top-100            |             47 |      45 |               1000 |                     703 | False                |                 35 |        0.4693 |           1472 |          477.6 |                    34 |                           33 |
| Top-200            |            100 |     116 |               1000 |                     999 | False                |                 66 |        1.039  |           1475 |         1005   |                    89 |                           87 |
| Top-500            |            277 |     422 |               5000 |                    5000 | True                 |                247 |        9.812  |           1557 |         2759   |                   329 |                          320 |

Candidate generation is capped by the stated budget; covers were NOT exhaustively enumerated.

## Paired tests (Wilcoxon signed-rank, Holm-adjusted)
| comparison                                 | metric      |   n |   mean_a |   mean_b |   mean_diff |   W |   p |   rank_biserial |   cohen_dz |   p_holm |
|:-------------------------------------------|:------------|----:|---------:|---------:|------------:|----:|----:|----------------:|-----------:|---------:|
| Proposed Pareto vs Minimum-cardinality     | Paths       |   2 | 89       | 87       |   2         |   0 | 0.5 |          1      |   nan      |        1 |
| Proposed Pareto vs Minimum-cardinality     | Worst cost  |   2 |  2.658   |  2.975   |  -0.3175    |   1 | 1   |         -0.3333 |    -0.2988 |        1 |
| Proposed Pareto vs Minimum-cardinality     | Confidence  |   2 |  0.4089  |  0.4073  |   0.00162   |   0 | 0.5 |          1      |     1.131  |        1 |
| Proposed Pareto vs Minimum-cardinality     | Uncertainty |   2 |  0.09735 |  0.09798 |  -0.0006359 |   0 | 0.5 |         -1      |    -0.851  |        1 |
| Proposed Pareto vs Minimum-cardinality     | Runtime     |   2 |  0.5623  |  0.5607  |   0.001587  |   0 | 0.5 |          1      |    18.78   |        1 |
| Proposed Pareto vs Minimum-cardinality     | ARI ±10%    |   2 |  0.4518  |  1       |  -0.5482    |   0 | 0.5 |         -1      |   -21.55   |        1 |
| Proposed Pareto vs Minimum worst-cost      | Paths       |   2 | 89       | 94       |  -5         |   0 | 0.5 |         -1      |   nan      |        1 |
| Proposed Pareto vs Minimum worst-cost      | Worst cost  |   2 |  2.658   |  1.801   |   0.8564    |   0 | 0.5 |          1      |     3.993  |        1 |
| Proposed Pareto vs Minimum worst-cost      | Confidence  |   2 |  0.4089  |  0.4064  |   0.002518  |   0 | 0.5 |          1      |     4.186  |        1 |
| Proposed Pareto vs Minimum worst-cost      | Uncertainty |   2 |  0.09735 |  0.09789 |  -0.0005447 |   0 | 0.5 |         -1      |    -4.309  |        1 |
| Proposed Pareto vs Minimum worst-cost      | Runtime     |   2 |  0.5623  |  0.5607  |   0.001587  |   0 | 0.5 |          1      |    18.78   |        1 |
| Proposed Pareto vs Minimum worst-cost      | ARI ±10%    |   2 |  0.4518  |  0.5518  |  -0.1001    |   0 | 0.5 |         -1      |    -0.9355 |        1 |
| Proposed Pareto vs Robustness-aware Pareto | Paths       |   2 | 89       | 89.5     |  -0.5       |   0 | 1   |         -1      |    -0.7071 |        1 |
| Proposed Pareto vs Robustness-aware Pareto | Worst cost  |   2 |  2.658   |  2.565   |   0.09329   |   0 | 1   |          1      |     0.7071 |        1 |
| Proposed Pareto vs Robustness-aware Pareto | Confidence  |   2 |  0.4089  |  0.4098  |  -0.0008116 |   0 | 1   |         -1      |    -0.7071 |        1 |
| Proposed Pareto vs Robustness-aware Pareto | Uncertainty |   2 |  0.09735 |  0.09783 |  -0.0004805 |   0 | 1   |         -1      |    -0.7071 |        1 |
| Proposed Pareto vs Robustness-aware Pareto | Runtime     |   2 |  0.5623  |  0.5744  |  -0.01208   |   0 | 0.5 |         -1      |   -25.76   |        1 |
| Proposed Pareto vs Robustness-aware Pareto | ARI ±10%    |   2 |  0.4518  |  0.4286  |   0.02313   |   0 | 0.5 |          1      |    15.23   |        1 |

## Reactome comparison
| method              |   multi_gene_paths |   paths_enriched |   paths_with_FDR_lt_0_05 |   fraction_significant |   median_best_FDR |   genes_in_multigene_paths |   cover_level_significant_pathways |
|:--------------------|-------------------:|-----------------:|-------------------------:|-----------------------:|------------------:|---------------------------:|-----------------------------------:|
| Proposed Pareto     |                 14 |               14 |                       14 |                      1 |         0.0003561 |                         42 |                                 15 |
| Minimum-cardinality |                 13 |               13 |                       13 |                      1 |         0.0003242 |                         43 |                                 15 |
| Minimum worst-cost  |                 22 |               22 |                       22 |                      1 |         0.0004862 |                         42 |                                 15 |
