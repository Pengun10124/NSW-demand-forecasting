# NSW Minimum Load Forecasting

This repository preserves the original implementation and records separate experiments in dataset expansion, fixed-date validation and rooftop photovoltaic (PV) features. Branches retain their inputs, code and results so that changes can be reproduced and compared.

The regression targets are the **daily minimum demand** and the **midday minimum demand between 10:00 and 15:00**. They are not forecasts of the complete 48-point daily demand curve. Reported MAE is the mean absolute error in MW; lower is better. MAE should not be interpreted as a percentage accuracy score.

## Branches

| Branch | Purpose | Changes |
| --- | --- | --- |
| `main` | Original implementation and project overview | Original notebook and workbook from commit `7be74df` remain unchanged; this README is the only addition. |
| `codex/current-results-snapshot` | Preserved experiments before PV integration | Adds supplementary inputs, the merged dataset, fixed-date validation, July–August testing and error diagnostics. Preserves the original notebook and workbook. |
| `codex/pv-model-experiment` | Rooftop PV feature comparison | Builds on the snapshot branch, adds traceable PV inputs and compares the same Ridge model with historical and observed PV features. |

The snapshot and PV branch commits are `d47db5d` and `ee554b2`, respectively. Multiple experiment stages are stored inside the snapshot branch; they are not separate Git branches. Historical filenames are retained for reproducibility.

## Data changes

The original workbook contains 35,040 half-hour records covering July 2024–June 2026. Supplementary inputs cover the full 2023 calendar year and January 2025–August 2026.

The merged dataset has **55,536 records across 1,157 complete days**:

- 17,520 supplementary records from 2023;
- 35,040 original records from July 2024–June 2026;
- 2,976 supplementary records from July–August 2026.

On overlapping interval-start timestamps, the entire original row takes priority, including any missing values. This removes 26,208 overlapping supplementary rows. January–June 2024 remains absent; no synthetic records are inserted. Lag and rolling features restart after this gap, removing seven warm-up days from each continuous segment.

Times use fixed AEST (UTC+10). Interval-end electricity timestamps are converted to interval starts by subtracting 30 minutes. Demand is a half-hour mean. The inputs retain temperature, wind and calendar fields needed by the original model; price, humidity and textual weather descriptions are not model features.

The original wind input is regionally weighted, whereas supplementary wind is an available-station mean. A shared column name does not make these definitions equivalent. Supplementary temperature comes from Sydney Airport observations, and some inherited wind values were forward-filled. Source manifests and row-level provenance record these differences.

## Model and validation changes

The original comparison includes Ridge regression, Random Forest, Gradient Boosting and tuned Gradient Boosting, with yesterday, last-week and seasonal-average baselines. Features describe calendar seasonality, observed weather and recent demand. The fixed-date version retains the model definitions, tuning grid and random seed.

Adding rows to a conventional five-fold time-series split moves the validation dates. An initial merged-data run therefore changed both the training history and the validation windows, selecting tuned Gradient Boosting for daily minimum demand. A subsequent version fixes the original validation windows:

| Fold | Validation start | Validation end |
| --- | --- | --- |
| 1 | 2024-10-23 | 2025-02-04 |
| 2 | 2025-02-05 | 2025-05-20 |
| 3 | 2025-05-21 | 2025-09-02 |
| 4 | 2025-09-03 | 2025-12-16 |
| 5 | 2025-12-17 | 2026-03-31 |

Each fold contains 105 validation days and trains only on earlier available days. The merged version has 990 training days before April 2026, compared with 632 in the original-data experiment. Imputation and scaling are fitted within each training fold. The fixed-date comparison selects **Ridge for both targets**, using validation MAE rather than test results.

### April–June 2026 test

All rows below use the same 91 test days and identical original demand targets.

| Experiment | Daily-minimum model | Daily-minimum MAE | Midday-minimum model | Midday-minimum MAE |
| --- | --- | ---: | --- | ---: |
| Original dataset | Ridge | 316.31 | Ridge | 456.94 |
| Merged dataset, moving validation windows | Tuned Gradient Boosting | 344.45 | Ridge | 433.63 |
| Merged dataset, fixed validation dates | Ridge | **301.04** | Ridge | **433.63** |

Relative to the original-data run, fixed-date merged-data MAE decreases by approximately 4.8% and 5.1%. The initial daily-minimum deterioration should not be attributed to additional data alone: validation windows and the selected model also changed.

Supplementary-only experiments are retained under `experiments/results/our_2025_2026` and `our_baseline_2023`. The former reports April–June 2026 MAEs of 305.89 and 469.76 MW; the latter reports October–December 2023 MAEs of 432.56 and 563.13 MW. Their inputs, training histories and/or test dates differ, so these scores are not controlled measures of the benefit of merging data.

### July–August 2026 test

The selected Ridge models are reconstructed with the same training cutoff, **31 March 2026**, and unchanged parameters. They reproduce the earlier April–June scores before evaluation on 62 later days. No April–August observations are used to fit the model. Earlier observed demand within the test period is available as a lag input for subsequent days, making this a rolling daily evaluation rather than a single two-month-ahead forecast.

| Test period | Daily-minimum MAE | Midday-minimum MAE |
| --- | ---: | ---: |
| July | 264.21 | 317.28 |
| August | 602.46 | 710.78 |
| July–August | 433.34 | 514.03 |

Error diagnostics reconciled all 2,976 half-hour demand values against six-reading averages from the locally archived AEMO five-minute files, finding only rounding differences. Daily minimum demand changed more sharply in August, and several large errors coincided with changes in PV output. Removing wind or extending training through June did not materially resolve the errors. This supports further feature and model investigation, not deletion of the high-error observations. Local reconciliation is not an independent audit of every upstream observation or proof of a particular weather cause.

## Rooftop PV experiment

The PV branch adds NSW1 `MEASUREMENT` values from monthly `ROOFTOP_PV_ACTUAL_FILE01.csv` archives, including the 2024 months absent from the supplementary core tables. `POWER` is joined to interval starts after subtracting 30 minutes from `INTERVAL_DATETIME`. Extra next-month days in some archives are excluded using the owning month. `SATELLITE` records are not blended with `MEASUREMENT` records.

The combined file keeps all 55,536 rows and preserves the original eight non-PV columns. Six PV values remain missing across three incomplete days. Missing values are not interpolated. Incomplete daily PV summaries remain missing and are handled by the training-fitted median imputer. Source quality indicators and hashes are retained under `data/pv`.

Three variants use the same **Ridge model with alpha=10**, training dates, fixed validation windows and test periods, without further parameter tuning:

1. **No PV:** the existing 26 features.
2. **Historical PV only:** adds daily mean, daily maximum and midday mean PV, each lagged one and seven days.
3. **Observed PV plus history:** also adds current-day PV mean, maximum, midday mean and the change in midday mean since yesterday.

### Validation and test MAE

Each cell below is **daily minimum / midday minimum**, in MW.

| Period | No PV | Historical PV only | Observed PV plus history |
| --- | ---: | ---: | ---: |
| Fixed-date cross-validation | 433.27 / 606.40 | 429.99 / 576.43 | 304.85 / 344.39 |
| April–June 2026 | 301.04 / 433.63 | 314.91 / 485.08 | 321.07 / 450.31 |
| July 2026 | 264.21 / 317.28 | 308.20 / 349.58 | 448.78 / 605.27 |
| August 2026 | 602.46 / 710.78 | 614.19 / 736.20 | 480.63 / 676.86 |
| July–August 2026 | 433.34 / 514.03 | 461.20 / 542.89 | 464.70 / 641.06 |

Observed PV improves August MAE but worsens July and the combined July–August MAE. Historical PV alone does not improve these test periods. Validation gains therefore do not establish a consistent out-of-sample improvement. The PV branch is retained as an experiment, not a replacement for the existing model.

**Actual same-day PV is unavailable in advance.** The observed-PV variant is an explanatory scenario; operational forecasting would require a PV forecast. All variants also retain observed weather inputs, which represent idealized weather information rather than actual weather forecasts.

## Reproduction and result locations

The experiment files exist on their respective branches, not on `main`. Install dependencies after switching to an experiment branch:

```bash
git switch codex/current-results-snapshot
python -m pip install -r requirements.txt
python experiments/run_fixed_validation.py
python experiments/test_july_august.py
```

`test_july_august.py` deliberately refuses to overwrite its existing result directory. To repeat that run, first preserve or rename `experiments/results/fixed_validation_july_august_2026`. The other commands write to their designated experiment directories; preserve those directories first if the saved run must remain unchanged. Low-level CPU threading can be limited with `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`.

For the PV comparison:

```bash
git switch codex/pv-model-experiment
python -m pip install -r requirements.txt
python experiments/run_pv_experiment.py
```

The saved joined PV CSV supports reruns without the original desktop archive paths. Rebuilding that CSV from scratch requires the local source archives recorded in its manifest.

| Experiment | Entry point | Saved report |
| --- | --- | --- |
| Fixed-date merged comparison | `notebooks/NSW_Minimum_Load_Combined_Fixed_Validation.ipynb` | `experiments/results/combined_fixed_validation/COMPARISON.html` |
| July–August evaluation | `experiments/test_july_august.py` | `experiments/results/fixed_validation_july_august_2026/REPORT.html` |
| August diagnosis | Saved diagnostic scripts and tables | `experiments/results/august_error_diagnosis/REPORT.html` |
| PV comparison | `notebooks/NSW_Rooftop_PV_Fixed_Validation.ipynb` | `experiments/results/pv_fixed_validation/REPORT.html` |

Recorded runs used pandas 2.2.3 and scikit-learn 1.6.1. CSV metrics, daily predictions, validation windows and metadata are retained alongside reports. Historical diagnostic scripts may contain local desktop paths.

## Interpretation limits

- These test periods have already been examined. Subsequent comparisons are exploratory; final performance claims require an untouched evaluation period.
- Gradient Boosting tuning and reported cross-validation scores use the same folds rather than nested validation.
- The original low-load classification experiments remain available. Their statistical thresholds are based on training-set quantiles, differ between datasets and are not official grid-security limits. Classification thresholds are not re-estimated within each CV fold. The PV comparison evaluates regression only.
- Source-specific weather definitions, missing data and seasonality limit direct comparisons across datasets and periods.
- PV and demand associations do not by themselves establish causal weather effects.

## Sources

- [AEMO NEM data](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem)
- [AEMO NEMWeb](https://nemweb.com.au/)
- [Iowa Environmental Mesonet observation download](https://mesonet.agron.iastate.edu/request/download.phtml?network=AU__ASOS)

Exact input files, hashes and processing provenance are recorded in `data/provenance`, `data/combined` and `data/pv` on the experiment branches. Numerical summaries above come from the saved local experiment metrics, not from published benchmark claims.
