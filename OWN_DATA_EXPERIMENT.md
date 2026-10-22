# NSW minimum demand: own-source Gradient Boosting

This local experimental branch now uses only independently collected AEMO demand and IEM weather data. It replaces the prior merged-data workflow. The main branch and PV experiment branch are unchanged. No changes have been pushed.

The two regression targets are the lowest half-hour demand of the day and the lowest half-hour demand between 10:00 and 15:00. Both use the existing scikit-learn Histogram Gradient Boosting model with 300 iterations, learning rate 0.05 and random seed 42. No linear models are trained or selected. No PV features are supplied.

## Data and cleaning

See [the data guide](data/own_reprocessed/README.txt), [exact weather download URLs](data/own_reprocessed/weather_download_sources.json) and [source manifest](data/own_reprocessed/source_manifest.json).

Temperature is Sydney Airport observation. Wind is the equal mean of Sydney Airport, Williamtown, Dubbo and Tamworth, after separately cleaning each station. Missing runs up to 90 minutes use the last real observation. Longer runs remain missing and the affected day is excluded. Demand requires six valid five-minute observations per half-hour and is never filled. Calendar gaps are preserved before deriving lag and rolling features. No median filling is applied.

The 2023, 2025 and September 2025-August 2026 groups contain 973 unique days. Cleaning retains 949 complete days; 811 days have all 26 model inputs, including seven-day history. No 2024 data is supplied by the selected own-source periods. The archived donor workbook and prior merged table are not part of this branch's active data.

## Run

Install requirements, then run from the repository root:

```bash
python experiments/run_own_gradient.py
```

The default result directory already contains this run. To rerun without overwriting it, choose a new directory:

```bash
python experiments/run_own_gradient.py --output experiments/results/own_gradient_repeat
```

Alternatively use [the notebook](notebooks/NSW_Own_Data_Gradient_Boosting.ipynb) from a full checkout. Models use 697 eligible training days before April 2026. Test periods remain April-June (75 eligible days), July (15) and August (24). The original five validation calendar windows are retained but only available dates are evaluated. This is not identical validation coverage to earlier runs.

## Results

[Full results](experiments/results/own_gradient_90min/REPORT.html), [metrics](experiments/results/own_gradient_90min/metrics.csv), [excluded test dates](experiments/results/own_gradient_90min/excluded_test_dates.csv) and [comparison on identical dates](experiments/results/own_gradient_90min/matched_date_comparison.csv).

| Period | Eligible days | Daily-minimum MAE (MW) | Midday-minimum MAE (MW) |
|---|---:|---:|---:|
| April-June | 75 | 327.89 | 447.15 |
| July | 15 | 292.71 | 329.04 |
| August | 24 | 602.25 | 662.86 |
| July-August | 39 | 483.19 | 534.47 |

Scores cover retained dates only. Use the matched-date comparison rather than comparing to earlier full-month scores. Previously trained Ridge predictions are included as a saved reference only; Ridge is not rerun. Observed same-day weather is an idealized input, not an archived forecast. Earlier observed test days feed later demand lags. Test dates have been inspected before, so this is exploratory evaluation rather than a new blind final test.

## Preservation

Before replacement, the complete former branch was exported to a verified local archive outside the repository. Previous results remain recoverable from that archive and the preceding Git commit. Only this branch is changed; no push is performed.
