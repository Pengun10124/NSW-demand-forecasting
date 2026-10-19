"""Merge all supplied periods (upstream wins) and run a gap-aware experiment."""
from pathlib import Path
import os, json, time, hashlib, contextlib
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR','/tmp/nsw-comparison-mpl')
os.environ.setdefault('OMP_NUM_THREADS','2')
import pandas as pd
from joblib import parallel_config

REPO=Path(__file__).resolve().parents[1]
DATA=REPO/'data/combined'
OUT=REPO/'experiments/results/combined_original_priority'
DATA.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
FIELDS=['Interval_Start_AEST','TOD_Slot_0_47','Demand_MW','Sydney_Airport_Temp_C','Wind_speed_mps','DayOfWeek_Mon0','Is_Weekend','Is_NSW_Public_Holiday']
their_path=REPO/'data/NSW_ML_Training_Data_2024_2026.xlsx'
our_path=REPO/'data/NSW1_Minimum_Load_Inputs.xlsx'
def read(path,sheet,wind,source):
    d=pd.read_excel(path,sheet_name=sheet).rename(columns={wind:'Wind_speed_mps'})[FIELDS].copy()
    d.Interval_Start_AEST=pd.to_datetime(d.Interval_Start_AEST,errors='raise')
    assert not d.Interval_Start_AEST.duplicated().any()
    d['source']=source
    return d
theirs=read(their_path,'ML Training Data','Wind_NSW_Weighted_mps','original')
ours=pd.concat([read(our_path,'Baseline 2023','Wind_NSW_Mean_mps','our_2023'),read(our_path,'ML Training Data','Wind_NSW_Mean_mps','our_main')],ignore_index=True)
assert not ours.Interval_Start_AEST.duplicated().any()
overlap=ours.Interval_Start_AEST.isin(theirs.Interval_Start_AEST)
# Whole-row precedence: even upstream missing values are retained on overlap.
merged=pd.concat([ours.loc[~overlap],theirs],ignore_index=True).sort_values('Interval_Start_AEST').reset_index(drop=True)
assert not merged.Interval_Start_AEST.duplicated().any()
pd.testing.assert_frame_equal(merged.set_index('Interval_Start_AEST').loc[theirs.Interval_Start_AEST,FIELDS[1:]],theirs.set_index('Interval_Start_AEST')[FIELDS[1:]],check_dtype=False)
ts=merged.Interval_Start_AEST
assert ts.is_monotonic_increasing
assert merged.groupby(ts.dt.normalize()).size().eq(48).all()
assert merged.TOD_Slot_0_47.eq(ts.dt.hour*2+ts.dt.minute//30).all()
assert merged.Demand_MW.notna().all()
segment=ts.diff().ne(pd.Timedelta(minutes=30)).cumsum()
segments=merged.groupby(segment).agg(start=('Interval_Start_AEST','min'),end=('Interval_Start_AEST','max'),rows=('Demand_MW','size'))
merged[FIELDS].to_csv(DATA/'NSW1_Combined_Original_Priority.csv',index=False,date_format='%Y-%m-%d %H:%M:%S')
audit=merged[['Interval_Start_AEST','source']].copy()
audit['wind_definition']=audit.source.map({'original':'Original three-region weighted wind','our_2023':'Available-station mean, inherited forward fills','our_main':'Available-station mean, inherited forward fills'})
audit['source_segment']=segment
audit.to_csv(DATA/'row_provenance.csv',index=False)
manifest={'merged_rows':len(merged),'merged_days':len(merged)//48,'our_rows_discarded_on_overlap':int(overlap.sum()),'retained_by_source':merged.source.value_counts().to_dict(),
 'segments':json.loads(segments.reset_index(drop=True).to_json(orient='records',date_format='iso')),
 'rule':'Whole original row wins on matching interval start; no filling upstream missing values from our rows.',
 'missing_values':merged[FIELDS].isna().sum().to_dict(),
 'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [their_path,our_path]},
 'test_start':'2026-04-01','test_end':'2026-06-30','excluded_future_rows':int((ts>=pd.Timestamp('2026-07-01')).sum())}
(DATA/'merge_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2),flush=True)
nb=json.loads((REPO/'notebooks/NSW_Minimum_Load_Forecasting.ipynb').read_text())
raw=merged.loc[ts<pd.Timestamp('2026-07-01'),FIELDS].copy()
ns={'__name__':'__main__'}
start=time.monotonic()
with (OUT/'run.log').open('w') as log,contextlib.redirect_stdout(log),parallel_config(backend='threading',n_jobs=2):
 for i,c in enumerate(nb['cells']):
    if c['cell_type']!='code':continue
    code=''.join(c['source'])
    if i==2:code=code.replace('RESULTS_DIR = "results"',f'RESULTS_DIR = {str(OUT)!r}')
    if i==4:
        ns['raw']=raw
        print(f'Combined input: {len(raw)} rows through June 2026')
        continue
    if i==6:
        code=code.replace('(ts.diff().dropna() == pd.Timedelta(minutes=30)).all()', '(ts.groupby(ts.diff().ne(pd.Timedelta(minutes=30)).cumsum()).diff().dropna() == pd.Timedelta(minutes=30)).all()')
        code=code.replace('Every interval is 30 minutes','Every interval within each continuous segment is 30 minutes')
        code+='\nprint("Known absent calendar span: January-June 2024; no records fabricated.")\n'
    if i==8:code=code.replace('Wind_NSW_Weighted_mps','Wind_speed_mps')
    if i==10:code=code.replace('Jul 2024 to Jun 2026','Combined 2023 and Jul 2024 to Jun 2026')
    if i==12:
        code=code.replace('feat = build_features(daily)', '''# Reset all shifts and rolling windows at calendar gaps.
segments = daily.index.to_series().diff().ne(pd.Timedelta(days=1)).cumsum()
feat = pd.concat([build_features(part) for _, part in daily.groupby(segments)]).sort_index()''')
        code=code.replace('first week removed','first week of each continuous segment removed')
    exec(compile(code,f'combined_cell_{i}','exec'),ns)
    if i==12:
        assert len(ns['feat'])==len(ns['daily'])-14
        assert pd.Timestamp('2024-07-01') not in ns['feat'].index
        assert pd.Timestamp('2024-07-08') in ns['feat'].index
    if i==14:
        assert len(ns['train'])==990 and len(ns['hold'])==91
        # Hold-out rows/targets must exactly match original-data experiment.
        expected=theirs[(theirs.Interval_Start_AEST>='2026-04-01')&(theirs.Interval_Start_AEST<'2026-07-01')].copy()
        expected['Date']=expected.Interval_Start_AEST.dt.normalize()
        pd.testing.assert_series_equal(ns['hold'].day_min,expected.groupby('Date').Demand_MW.min(),check_names=False)
        pd.testing.assert_series_equal(ns['hold'].mid_min,expected[expected.TOD_Slot_0_47.between(20,29)].groupby('Date').Demand_MW.min(),check_names=False)
    if i==20:print('Combined model search complete',file=__import__('sys').__stdout__,flush=True)
    ns['plt'].close('all')
manifest.update({'train_days':len(ns['train']),'test_days':len(ns['hold']),'lag_warmup_days_removed':14,'thresholds_mw':ns['THRESHOLDS'],'seconds':round(time.monotonic()-start,1),'all_cells_passed':True,'holdout_targets_match_original':True})
(OUT/'run_metadata.json').write_text(json.dumps(manifest,indent=2))
rows=[]
for t,name in ns['TARGETS'].items():
    table=ns['hold_table'];row=table[(table.Target==name)&(table.Model==ns['best_reg'][t])].iloc[0].to_dict()
    rows.append({'Dataset':'Combined (original wins)',**row})
new=pd.DataFrame(rows)
previous=pd.read_csv(REPO/'experiments/results/regression_comparison.csv')
previous=previous[previous.Dataset.isin(['Original data','Our main data'])]
compare=pd.concat([previous[new.columns],new],ignore_index=True)
compare.to_csv(OUT/'comparison_same_holdout.csv',index=False)
body='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Combined data experiment</title><style>body{font:15px/1.6 system-ui;max-width:1250px;margin:40px auto;padding:0 24px;color:#203040}table{border-collapse:collapse;width:100%}th,td{padding:10px;border-bottom:1px solid #ddd;text-align:left}th{background:#eef3f6}</style><h1>Combined data: original rows take priority</h1><p>Both original workbooks and notebooks remain unchanged. The new CSV combines our 2023 and later main data with the original workbook. Matching interval-start timestamps retain the entire original row, including its missing values. No cross-source imputation was applied.</p>'''
body+=f'<p>Total: {len(merged):,} rows ({len(merged)//48:,} complete days). Removed {int(overlap.sum()):,} overlapping rows from our inputs. Missing January–June 2024 remains a gap. Calendar lags reset at each continuous segment; 14 warm-up days are excluded.</p>'
body+='''<p>The model uses 990 training days and the same 91 test days, April–June 2026, as the original-data comparison. All test demand targets match the original data exactly. July–August 2026 is included in the merged file but excluded from this run to avoid training on dates after the test. The original model grids and seed 42 are unchanged. Model selection uses training-period time-series CV.</p><h2>Regression: same test dates and original test targets for the combined run</h2>'''
body+=compare.round(3).to_html(index=False,border=0)
body+='<h2>Combined classification</h2>'+ns['hold_clf_table'].round(3).to_html(index=False,border=0,na_rep='Not defined')
body+='''<h2>Interpretation</h2><ul><li>The combined run extends the original training data with 2023. Under this cutoff, our July–August 2026 additions are not used.</li><li>Wind has mixed definitions: original weighted regional wind on original timestamps, and our available-station mean elsewhere. Temperature observation processing also differs. The provenance CSV records the source for every row.</li><li>The low-load threshold is recomputed from each training set, so classification labels and scores are not directly comparable across datasets without a common fixed threshold.</li><li>This is an additional experiment on a previously inspected test period, not a fresh untouched final validation. Keep a new hold-out for final claims. Upstream CV selection and full-training quantile definitions are retained, not nested validation.</li><li>Observed weather is a perfect-weather assumption. Results do not represent operational forecast performance or an official grid safety limit.</li></ul></html>'''
(OUT/'COMPARISON.html').write_text(body)
(DATA/'README.txt').write_text('Combined original-priority dataset\n\n'+json.dumps(manifest,indent=2)+'\n\nWind_speed_mps harmonizes the column name only, not the two wind measurement definitions. See row_provenance.csv. Use experiments/run_combined.py for gap-aware lags and the April-June 2026 comparison. Do not pass this discontinuous CSV to the unmodified original notebook.\n')
print(compare[['Dataset','Target','Model','MAE (MW)','MAPE (%)']].round(2).to_string(index=False),flush=True)
print(ns['hold_clf_table'].to_string(index=False),flush=True)
print('PASS: combined deduplication, original-row precedence, gap-aware lags, identical test targets, all model cells.',flush=True)
