NSW1 model inputs: own-source reprocessing, 90-minute maximum weather gap

FILES
baseline_2023.csv, baseline_2025.csv and recent_20250901_20260831.csv contain retained full days, 48 rows per day.
model_inputs_all_unique.csv combines the three periods without duplicate timestamps. The 2025 baseline overlaps the recent period during September-December 2025. Use this unique file when combining periods.
daily_model_ready.csv contains the original model's 26 features plus daily targets and auxiliary daily statistics. Select only columns listed in feature_columns.json as inputs. Never pass targets, min_hour, day_mean or eligibility flags as predictors.
daily_features_with_eligibility.csv retains the complete daily calendar and feature eligibility, including the unavailable 2024 period and excluded dates.
fill_audit.csv records station observations, cleaned values, original gap length, fill status and age since the last true observation.
daily_quality.csv and excluded_days.csv record excluded dates and reasons.
all_intervals_with_exclusion_flags.csv retains every requested interval for audit; it is NOT the training table.
source_manifest.json records source paths and SHA256 hashes. weather_download_sources.json contains exact publicly verifiable weather download URLs. raw_weather/ contains the unmodified downloaded responses.

SOURCES
Demand: the existing local AEMO NSW1 monthly PRICE_AND_DEMAND CSV files, TOTALDEMAND. These files were reread; no earlier cleaned table or Thomas workbook supplies data.
AEMO public data portal: https://www.nemweb.com.au/
Weather: Iowa State University Iowa Environmental Mesonet (IEM) METAR archive, not a direct BOM export.
https://mesonet.agron.iastate.edu/request/download.phtml?network=AU__ASOS
https://mesonet.agron.iastate.edu/info/datasets/metar.html
Stations: YSSY Sydney Airport, YWLM Williamtown, YSDU Dubbo, YSTW Tamworth. All are NSW stations. Canberra is excluded.
Holidays: https://www.nsw.gov.au/about-nsw/public-holidays (NSW statewide holidays only, excluding bank and local holidays). Historical 2023/2025 dates follow the previously documented calendar; 2026 includes April 27 additional Anzac holiday.

PROCESSING
1. Fixed AEST (UTC+10) throughout, with no daylight-saving conversion. Electricity timestamp is interval end; Interval_Start_AEST is 30 minutes earlier.
2. Each half-hour demand equals the arithmetic average of all six five-minute TOTALDEMAND readings ending in that interval. Require all six readings. Demand is never forward-filled. In this run no half-hour demand is incomplete.
3. Select weather observations at exact :00/:30 timestamps. Match the observation timestamp to electricity interval END, preserving the earlier alignment convention. These are point observations, not measured half-hour average weather. Off-grid special reports are not substituted.
4. Convert temperature from Fahrenheit to Celsius, and wind from knots to m/s using 1852/3600. Duplicate weather timestamps are accepted only if values agree; conflicting duplicates stop processing.
5. Apply the user-defined gap rule separately to Sydney temperature and each of the four wind series. For an entire missing run of at most three consecutive half-hour slots, copy the last genuine observation, with a maximum age of 90 minutes. Longer runs are left wholly missing, including their first three slots. Missing leading values with no suitable predecessor remain missing. Boundary observations are downloaded for gap classification.
6. Temperature input is Sydney Airport temperature. Wind is the arithmetic mean of the same four stations after station-level cleaning; all four must be present. It is an equal-station mean, NOT Thomas's regional weighted wind, NOT a validated NSW-area weighted mean. Other stations' temperatures are retained in raw downloads but are not model inputs.
7. Exclude an entire interval-start calendar day if any of its 48 intervals has unresolved Sydney temperature, any unresolved wind station, or incomplete demand. This implements the long-gap rule at station level. No additional threshold for cumulative scattered short gaps was invented.
8. No statistical extreme-value clipping or median filling is applied. Observed extremes remain. Source missing markers are treated as missing, not numeric zero. Negative wind/demand or conflicting source values stop processing for review.
9. Rebuild daily targets and the existing 26 features on a full daily calendar. Excluded days remain gaps before lag1, lag7 and rolling7 calculations, so dates cannot slide. Excluded days do not contribute lagged demand; subsequent days with incomplete required history are also ineligible for modelling. No median imputation rescues these incomplete feature rows.
10. Preserve the three named date ranges, but the cleaned versions are no longer complete years after exclusions. 2024 is unavailable in these own-source groups and is not supplied from Thomas's dataset.

MODEL USE
The original notebook assumes continuous dates and a fixed row count. Do not simply upload these CSVs unchanged and ignore failing checks. Use daily_model_ready.csv with feature_columns.json; import_model_data.py provides dated train/test matrices and both targets, avoiding the original row-based gap problem.
Retain the training cutoff at March 31, 2026 for comparison with earlier experiments. Validation windows must have available training/validation rows; folds located in unavailable 2024 cannot be reused blindly. When comparing cleaning approaches, evaluate on matching dates and report excluded dates separately.
Weather is observed same-day weather, as in the existing retrospective setup. It is not an archived forecast and cannot demonstrate operational day-ahead performance.
This delivery prepares data only; no new model accuracy is claimed. No data or code has been pushed to GitHub.

RESULTS
{
  "baseline_2023": {
    "requested_days": 365,
    "retained_days": 358,
    "excluded_days": 7,
    "rows": 17184
  },
  "baseline_2025": {
    "requested_days": 365,
    "retained_days": 353,
    "excluded_days": 12,
    "rows": 16944
  },
  "recent_20250901_20260831": {
    "requested_days": 365,
    "retained_days": 357,
    "excluded_days": 8,
    "rows": 17136
  },
  "unique": {
    "requested_days": 973,
    "retained_days": 949,
    "excluded_days": 24,
    "model_ready_days": 811,
    "feature_count": 26,
    "filled_values": {
      "YSSY_wind_mps_filled": 880,
      "YSSY_temperature_c_filled": 876,
      "YWLM_wind_mps_filled": 1132,
      "YSDU_wind_mps_filled": 313,
      "YSTW_wind_mps_filled": 249
    },
    "excluded_by_month": {
      "2023-01": 4,
      "2023-02": 0,
      "2023-03": 0,
      "2023-04": 0,
      "2023-05": 3,
      "2023-06": 0,
      "2023-07": 0,
      "2023-08": 0,
      "2023-09": 0,
      "2023-10": 0,
      "2023-11": 0,
      "2023-12": 0,
      "2025-01": 3,
      "2025-02": 2,
      "2025-03": 2,
      "2025-04": 1,
      "2025-05": 0,
      "2025-06": 0,
      "2025-07": 1,
      "2025-08": 0,
      "2025-09": 1,
      "2025-10": 1,
      "2025-11": 0,
      "2025-12": 1,
      "2026-01": 0,
      "2026-02": 0,
      "2026-03": 0,
      "2026-04": 1,
      "2026-05": 1,
      "2026-06": 0,
      "2026-07": 2,
      "2026-08": 1
    }
  }
}

EXACT WEATHER DOWNLOAD LINKS
https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=YSSY&station=YWLM&station=YSDU&station=YSTW&network=AU__ASOS&data=tmpf&data=sknt&data=metar&year1=2022&month1=12&day1=31&year2=2024&month2=1&day2=2&tz=Etc%2FGMT-10&format=onlycomma&latlon=yes&elev=yes&missing=M&report_type=3&report_type=4&direct=no

https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=YSSY&station=YWLM&station=YSDU&station=YSTW&network=AU__ASOS&data=tmpf&data=sknt&data=metar&year1=2024&month1=12&day1=31&year2=2026&month2=9&day2=2&tz=Etc%2FGMT-10&format=onlycomma&latlon=yes&elev=yes&missing=M&report_type=3&report_type=4&direct=no