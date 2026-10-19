# NSW demand forecasting: original and additional datasets

The upstream notebook and original workbook are preserved. The original notebook was restored byte-for-byte from Git HEAD.

## Choose a notebook

- `notebooks/NSW_Minimum_Load_Forecasting.ipynb`: original upstream notebook, original GitHub data-loading behaviour and saved outputs.
- `notebooks/NSW_Minimum_Load_Forecasting_Our_Data.ipynb`: added version using our local workbook, four-station mean wind, and June–August 2026 hold-out. Outputs are cleared so they can be regenerated.

## Available data

- `data/NSW_ML_Training_Data_2024_2026.xlsx`: original workbook, unchanged.
- `data/NSW1_Minimum_Load_Inputs.xlsx`: additional workbook. Main sheet covers January 2025–August 2026; separate Baseline 2023 sheet retains the full year.
- Our matching CSVs and `data/USER_GUIDE.html` are included; source and quality records are in `data/provenance/`.

Do not combine 2023 with the later main sheet using positional lags: 2024 is missing. Our wind is an available-station mean, not the original three-region weighted measure. Weather is observed, used as an idealized forecast input. The risk threshold is statistical, not an official safety threshold.

## Run locally

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook
```

For the added notebook, launch from the repository root or `notebooks/`. For Colab, upload the added notebook and new workbook. Opening the upstream Colab URL still opens the original version.

## Dataset trials

`experiments/compare_datasets.py` runs the original model grids against three datasets with a fixed seed. The original and our main inputs are compared on the same April–June 2026 hold-out; training history differs and is recorded. Baseline 2023 uses October–December 2023 and is a separate period experiment, not a like-for-like ranking. Trial outputs are stored under `experiments/results/`. See its `COMPARISON.html` after execution.

The added interactive notebook retains its June–August 2026 split; the comparative experiment uses April–June for the overlapping 2026 period. Do not confuse the two evaluations.

## Combined dataset trial

Run `python experiments/run_combined.py` to merge both workbooks and test the combined inputs. Overlapping timestamps retain the complete original row, including missing values. The merged CSV is `data/combined/NSW1_Combined_Original_Priority.csv`; row provenance and merge rules are beside it. January-June 2024 remains absent; lag features reset per continuous segment. The April-June 2026 hold-out is unchanged, and July-August 2026 is excluded from this test. Results are in `experiments/results/combined_original_priority/COMPARISON.html`. Existing inputs and notebooks are preserved.


## Preserved branches and rooftop PV experiment

- `main`: identical to the fetched teammate `origin/main`, commit `7be74df8f60d3e10c1c2a471bb8e7a9097a08849`.
- `codex/current-results-snapshot`: all pre-PV code, data and experiment results, including previously ignored results.
- `codex/pv-model-experiment`: adds the rooftop PV dataset and fixed-date Ridge comparison.

Open `notebooks/NSW_Rooftop_PV_Fixed_Validation.ipynb` or run `python experiments/run_pv_experiment.py` from the repository root. Results: `experiments/results/pv_fixed_validation/REPORT.html`. The joined dataset is self-contained; source archives are required only to rebuild it from scratch. Observed same-day PV is an explanatory scenario, not an available operational forecast. Historical-only PV is reported separately. Existing model and source columns are unchanged.

The PV experiment improves August under observed-PV assumptions, but does not improve July-August overall. Keep all versions for comparison. Branches and commits are local; nothing has been pushed.
