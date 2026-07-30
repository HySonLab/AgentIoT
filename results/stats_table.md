# Consolidated statistics (auto-generated — do not hand-edit)

## Per-system results (mean ± std over seeds; n = #seeds)

|                                |   n |   f1_mean |   f1_std |   prec_mean |   prec_std |   rec_mean |   rec_std |   auc_mean |   auc_std |   dF1_mean |   lat_ms |
|:-------------------------------|----:|----------:|---------:|------------:|-----------:|-----------:|----------:|-----------:|----------:|-----------:|---------:|
| ('boiler_drift', 'baseline1')  |   5 |    0.5319 |   0.0117 |      0.3957 |     0.0228 |     0.8166 |    0.0489 |     0.75   |    0.0095 |     0.0234 |   0.2779 |
| ('boiler_drift', 'baseline2')  |   5 |    0.5464 |   0.0148 |      0.4193 |     0.0256 |     0.7888 |    0.0403 |     0.7502 |    0.0095 |     0.0523 |   0.2666 |
| ('boiler_drift', 'semas')      |   5 |    0.5378 |   0.0377 |      0.4393 |     0.0832 |     0.7367 |    0.1231 |     0.733  |    0.0123 |     0.047  |   0.7798 |
| ('boiler_static', 'baseline1') |   5 |    0.5367 |   0.0185 |      0.4341 |     0.0549 |     0.7345 |    0.1276 |     0.6575 |    0.0169 |     0      |   0.9373 |
| ('boiler_static', 'baseline2') |   5 |    0.5367 |   0.0185 |      0.4341 |     0.0549 |     0.7345 |    0.1276 |     0.6575 |    0.0169 |     0      |   0.9076 |
| ('boiler_static', 'semas')     |   5 |    0.5207 |   0.0595 |      0.4448 |     0.0332 |     0.6494 |    0.1493 |     0.6504 |    0.0251 |     0      |   2.0334 |
| ('wind_static', 'baseline1')   |   5 |    0.0639 |   0.0017 |      0.0337 |     0.001  |     0.6093 |    0.0314 |     0.4895 |    0.0093 |     0      |   3.1684 |
| ('wind_static', 'baseline2')   |   5 |    0.0639 |   0.0017 |      0.0337 |     0.001  |     0.6093 |    0.0314 |     0.4895 |    0.0093 |     0      |   3.2717 |
| ('wind_static', 'semas')       |   5 |    0.0682 |   0.0172 |      0.0596 |     0.0308 |     0.306  |    0.2489 |     0.5264 |    0.004  |     0      |   1.4999 |

## Welch t-tests on seed-level F1 (SEMAS vs baselines)

| Dataset | Comparison | ΔF1 | t | p | Cohen's d | Significant (α=0.05) |
|---|---|---|---|---|---|---|
| boiler_drift | SEMAS vs baseline1 | +0.0059 | 0.33 | 0.7519 | 0.21 | no |
| boiler_drift | SEMAS vs baseline2 | -0.0086 | -0.48 | 0.6536 | -0.30 | no |
| boiler_static | SEMAS vs baseline1 | -0.0160 | -0.57 | 0.5926 | -0.36 | no |
| boiler_static | SEMAS vs baseline2 | -0.0160 | -0.57 | 0.5926 | -0.36 | no |
| wind_static | SEMAS vs baseline1 | +0.0043 | 0.55 | 0.6088 | 0.35 | no |
| wind_static | SEMAS vs baseline2 | +0.0043 | 0.55 | 0.6088 | 0.35 | no |

## Drift-mode per-segment F1 (mean over seeds)

* baseline1: 0.502 -> 0.567 -> 0.526
* baseline2: 0.502 -> 0.585 -> 0.555
* semas: 0.495 -> 0.579 -> 0.542
