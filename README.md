# NSW minimum demand: mixed-data Gradient Boosting

This branch combines the original training workbook with the newly reprocessed supplementary data. Original workbook rows win overlapping timestamps. Both regression targets use **Histogram Gradient Boosting only**, with 300 iterations, learning rate 0.05 and random seed 42. No PV features or linear models are trained in this experiment.

## Run

Install requirements and run from the repository root. Choose a fresh output directory to preserve the saved results:

```bash
python experiments/run_mixed_gradient.py --output experiments/results/mixed_repeat
```

The original workbook is in `data/original/`. Supplementary input, source links, original weather downloads and cleaning audits are in `data/own_reprocessed/`. The executable original aggregation/feature steps are preserved in `experiments/original_feature_steps.json`.

## Method

Supplementary weather uses the 90-minute maximum forward-fill rule, followed by whole-day exclusion if unresolved required weather remains. Original workbook values retain their original processing. Sydney Airport temperature and four-station mean wind are used for supplementary observations; original weather includes its regional weighted wind. Calendar dates remain explicit for demand lags and rolling features. Training uses 951 eligible days before April 2026. Missing weather in original records is handled natively by Gradient Boosting.

Actual same-day weather stands in for forecasts. This is retrospective evaluation, not proof of operational day-ahead accuracy. Test periods were previously inspected, and excluded dates reduce evaluation coverage.

## Results on the same test dates

MAE in MW; lower is better. Comparisons below use identical dates and verified matching demand targets.

| Period | Days | Mixed: daily min | Own-only: daily min | Mixed: midday min | Own-only: midday min |
|---|---:|---:|---:|---:|---:|
| April-June | 75 | 315.98 | 327.89 | 436.67 | 447.15 |
| July | 15 | 266.57 | 292.71 | 357.63 | 329.04 |
| August | 24 | 603.19 | 602.25 | 687.67 | 662.86 |
| July-August | 39 | 473.72 | 483.19 | 560.73 | 534.47 |

Adding the original data improves July daily-minimum predictions, but August daily error is essentially unchanged and midday error is higher in July and August. There is no across-the-board improvement. Source mix and training coverage change together, so this does not isolate a causal explanation.

[Full report](experiments/results/mixed_recleaned_gradient/REPORT.html) · [Same-date comparison](experiments/results/mixed_recleaned_gradient/matched_comparison.csv) · [All available-date metrics](experiments/results/mixed_recleaned_gradient/metrics.csv).

The preceding own-only experiment remains available in [its documentation](OWN_DATA_EXPERIMENT.md) and `experiments/results/own_gradient_90min/`. Older branch contents remain recoverable in Git history and a verified local archive. Main and the PV branch are unchanged.
