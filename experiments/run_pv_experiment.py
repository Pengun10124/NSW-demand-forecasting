"""Fixed-date Ridge comparison with historical and observed rooftop PV inputs."""
from pathlib import Path
import os,json,ast,hashlib
os.environ.setdefault('MPLBACKEND','Agg');os.environ.setdefault('MPLCONFIGDIR','/tmp/nsw-comparison-mpl')
import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
REPO=Path(__file__).resolve().parents[1];os.chdir(REPO)
DATA=REPO/'data/pv';OUT=REPO/'experiments/results/pv_fixed_validation'
DATA.mkdir(exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protected={str(p):sha(p) for p in (REPO/'experiments/results').rglob('*') if p.is_file() and OUT not in p.parents}
base=REPO/'data/combined/NSW1_Combined_Original_Priority.csv'
raw=pd.read_csv(base,parse_dates=['Interval_Start_AEST'])
joined_path=DATA/'NSW1_Combined_With_Rooftop_PV.csv'
if not joined_path.exists():
 frames=[];sources=[]
 for month in sorted(raw.Interval_Start_AEST.dt.strftime('%Y-%m').unique()):
  p=Path('/Users/maomengmeng/Desktop/database/AEMO')/month/'NSW1/ROOFTOP_PV_ACTUAL_FILE01.csv'
  a=pd.read_csv(p);a=a[a.REGIONID.eq('NSW1')&a.TYPE.eq('MEASUREMENT')].copy()
  a['Interval_Start_AEST']=pd.to_datetime(a.INTERVAL_DATETIME)-pd.Timedelta(minutes=30)
  # Some archived files include an extra day in the next month; use owning month.
  a=a[a.Interval_Start_AEST.dt.strftime('%Y-%m').eq(month)]
  assert a.Interval_Start_AEST.is_unique
  assert a.POWER.dropna().ge(0).all()
  a['source_month']=month;frames.append(a)
  sources.append({'month':month,'path':str(p),'sha256':sha(p),'measurement_rows':len(a),'missing_power':int(a.POWER.isna().sum())})
 pv=pd.concat(frames,ignore_index=True);assert pv.Interval_Start_AEST.is_unique
 audit=raw[['Interval_Start_AEST']].merge(pv[['Interval_Start_AEST','POWER','QI','TYPE','source_month']],on='Interval_Start_AEST',how='left',validate='one_to_one')
 audit.to_csv(DATA/'pv_row_provenance.csv',index=False)
 raw=raw.merge(pv[['Interval_Start_AEST','POWER']].rename(columns={'POWER':'Rooftop_PV_Power_MW'}),on='Interval_Start_AEST',how='left',validate='one_to_one')
 raw.to_csv(joined_path,index=False)
 (DATA/'source_manifest.json').write_text(json.dumps(sources,indent=2))
else:raw=pd.read_csv(joined_path,parse_dates=['Interval_Start_AEST'])
original=pd.read_csv(base,parse_dates=['Interval_Start_AEST'])
pd.testing.assert_frame_equal(raw[original.columns],original)
assert len(raw)==55536
nb=json.loads((REPO/'notebooks/NSW_Minimum_Load_Combined_Fixed_Validation.ipynb').read_text());ns={}
for i in [2,8,12,16,20]:
 code=''.join(nb['cells'][i]['source'])
 if i==2:code=code.replace('RESULT_BASE / "combined_fixed_validation"',repr(str(OUT)))
 if i==8:ns['raw']=raw
 if i==12:code=code.replace('assert len(feat) == 1081','assert len(feat) == 1143')
 tree=ast.parse(code)
 if i in [16,20]:tree.body=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) or (i==16 and isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='VALIDATION_WINDOWS' for t in x.targets))]
 exec(compile(tree,f'base_cell_{i}','exec'),ns)
d=raw.copy();d['Date']=d.Interval_Start_AEST.dt.normalize();d['mid']=d.TOD_Slot_0_47.between(20,29)
g=d.groupby('Date').Rooftop_PV_Power_MW
pv_daily=pd.DataFrame({'pv_mean':g.mean(),'pv_max':g.max(),'pv_count':g.count(),'pv_mid_mean':d[d.mid].groupby('Date').Rooftop_PV_Power_MW.mean(),'pv_mid_count':d[d.mid].groupby('Date').Rooftop_PV_Power_MW.count()})
# Do not pretend incomplete daily summaries are complete. No target rows dropped.
pv_daily.loc[pv_daily.pv_count.ne(48),['pv_mean','pv_max']]=np.nan
pv_daily.loc[pv_daily.pv_mid_count.ne(10),'pv_mid_mean']=np.nan
seg=pv_daily.index.to_series().diff().ne(pd.Timedelta(days=1)).cumsum()
for name in ['pv_mean','pv_max','pv_mid_mean']:
 pv_daily[name+'_lag1']=pv_daily.groupby(seg)[name].shift(1)
 pv_daily[name+'_lag7']=pv_daily.groupby(seg)[name].shift(7)
pv_daily['pv_mid_change']=pv_daily.pv_mid_mean-pv_daily.pv_mid_mean_lag1
pv_daily.to_csv(DATA/'daily_pv_features.csv')
feat=ns['feat'].join(pv_daily);base_cols=ns['FEATURES'];historical=[c for c in pv_daily if c.endswith('_lag1') or c.endswith('_lag7')]
variants={'No PV':base_cols,'Historical PV only':base_cols+historical,'Observed PV plus history':base_cols+historical+['pv_mean','pv_max','pv_mid_mean','pv_mid_change']}
train=feat[feat.index<'2026-04-01'];assert len(train)==990
split=ns['FixedDateSplit']();foldrows=[]
for k,(tr,va) in enumerate(split.split(train),1):
 foldrows.append({'fold':k,'train_days':len(tr),'validation_start':str(train.index[va[0]].date()),'validation_end':str(train.index[va[-1]].date()),'validation_days':len(va)})
pd.DataFrame(foldrows).to_csv(OUT/'fixed_validation_windows.csv',index=False)
rows=[];predrows=[];cvrows=[]
for variant,columns in variants.items():
 for target,label in ns['TARGETS'].items():
  model=ns['make_models']()['Linear Regression (Ridge)']
  errors=[]
  for k,(tr,va) in enumerate(split.split(train),1):
   m=clone(model).fit(train.iloc[tr][columns],train.iloc[tr][target]);pred=m.predict(train.iloc[va][columns]);e=np.abs(pred-train.iloc[va][target].values);errors.extend(e)
   cvrows.append({'Variant':variant,'Target':label,'Fold':k,'MAE_MW':float(e.mean())})
  rows.append({'Variant':variant,'Target':label,'Period':'Fixed-date CV','Days':len(errors),'MAE_MW':float(np.mean(errors))})
  model.fit(train[columns],train[target])
  for period,start,end in [('April-June','2026-04-01','2026-06-30'),('July','2026-07-01','2026-07-31'),('August','2026-08-01','2026-08-31'),('July-August','2026-07-01','2026-08-31')]:
   test=feat.loc[start:end];p=model.predict(test[columns]);e=p-test[target].values
   rows.append({'Variant':variant,'Target':label,'Period':period,'Days':len(test),'MAE_MW':float(np.abs(e).mean()),'RMSE_MW':float(np.sqrt(np.mean(e**2))),'Bias_MW':float(e.mean()),'R2':float(r2_score(test[target],p))})
   if period!='July-August':
    predrows.extend({'Date':str(dt.date()),'Variant':variant,'Target':label,'Actual_MW':float(actual),'Predicted_MW':float(pr),'Error_MW':float(pr-actual)} for dt,actual,pr in zip(test.index,test[target],p))
  print(f'Completed {variant}: {label}',flush=True)
metrics=pd.DataFrame(rows);preds=pd.DataFrame(predrows)
# Verify the unchanged baseline exactly reproduces the prior test and CV results.
old=pd.read_csv(REPO/'experiments/results/fixed_validation_july_august_2026/metrics.csv')
for target in ns['TARGETS'].values():
 for period in ['July','August','July-August']:
  expected=old[(old.Target==target)&(old.Period==period)&old.Model.str.startswith('Ridge')]['MAE (MW)'].iloc[0]
  actual=metrics[(metrics.Target==target)&(metrics.Period==period)&metrics.Variant.eq('No PV')].MAE_MW.iloc[0]
  assert np.isclose(expected,actual,atol=1e-7)
cvold=pd.read_csv(REPO/'experiments/results/combined_fixed_validation/cv_regression_results.csv')
for target in ns['TARGETS'].values():
 expected=cvold[(cvold.Target==target)&cvold.Model.eq('Linear Regression (Ridge)')]['MAE (MW)'].iloc[0]
 actual=metrics[(metrics.Target==target)&metrics.Period.eq('Fixed-date CV')&metrics.Variant.eq('No PV')].MAE_MW.iloc[0]
 assert np.isclose(expected,actual,atol=1e-7)
metrics.to_csv(OUT/'metrics.csv',index=False);preds.to_csv(OUT/'daily_predictions.csv',index=False);pd.DataFrame(cvrows).to_csv(OUT/'fold_scores.csv',index=False)
meta={'training_days':990,'training_end':'2026-03-31','source_rows':len(raw),'missing_pv_rows':int(raw.Rooftop_PV_Power_MW.isna().sum()),'incomplete_pv_days':int(pv_daily.pv_count.ne(48).sum()),'variants':variants,'data_sha256':sha(joined_path),'baseline_reproduced':True,'original_columns_unchanged':True,'old_result_files_verified_unchanged':len(protected),'model':'Ridge alpha=10, median imputer fitted on training fold, StandardScaler','protocol':'Same fixed folds, same training dates, no new tuning. Historical PV uses strictly prior days. Observed PV is an idealized explanatory experiment using same-day actual PV. Observed weather remains idealized in all variants. All test periods have already been inspected.'}
assert all(sha(Path(p))==v for p,v in protected.items())
(OUT/'metadata.json').write_text(json.dumps(meta,indent=2))
plt=ns['plt'];fig,axes=plt.subplots(2,1,figsize=(13,8))
for ax,label in zip(axes,ns['TARGETS'].values()):
 a=preds[(preds.Target==label)&preds.Date.ge('2026-07-01')]
 b=a[a.Variant.eq('No PV')];ax.plot(pd.to_datetime(b.Date),b.Actual_MW,label='Actual',color='black')
 for variant in variants:
  b=a[a.Variant.eq(variant)];ax.plot(pd.to_datetime(b.Date),b.Predicted_MW,label=variant,alpha=.8)
 ax.set_title(label);ax.set_ylabel('MW');ax.legend()
fig.tight_layout();fig.savefig(OUT/'predictions.png',dpi=140);plt.close('all')
html='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Rooftop PV experiment</title><style>body{font:15px/1.6 system-ui;max-width:1250px;margin:40px auto;padding:0 20px}td,th{padding:8px;border-bottom:1px solid #ddd}table{border-collapse:collapse}img{max-width:100%}</style><h1>Rooftop PV: fixed-validation Ridge experiment</h1><p>All variants use the same 990 training days before April 2026, same five validation windows and same Ridge alpha=10. No test-driven tuning. Original code, data and results are preserved on codex/current-results-snapshot. Main matches origin/main. This experiment belongs to codex/pv-model-experiment.</p><p><b>Observed PV is an explanatory scenario, not an operational forecast:</b> actual same-day PV is unknown in advance. Historical PV uses only lag-1 and lag-7 measurements. All variants still use observed weather as in the original experiment. Previously inspected test periods are exploratory.</p><h2>Results (MAE in MW; lower is better)</h2>'''
html+=metrics.pivot_table(index=['Target','Period'],columns='Variant',values='MAE_MW',sort=False).round(2).to_html(border=0)
html+='<h2>Predictions</h2><img src="predictions.png"><h2>Data handling</h2><p>Join AEMO NSW1 MEASUREMENT rooftop PV POWER to interval start by subtracting 30 minutes from INTERVAL_DATETIME. Keep only the owning source month because some archives contain extra next-month days. Do not combine MEASUREMENT and SATELLITE values. Missing source records/power remain blank, no interpolation or outlier removal. Incomplete daily summaries remain blank and are handled by fold-fitted median imputation. Lags reset across the 2024 gap. Non-PV input columns are unchanged.</p>'
html+=f'<p>Rows: {len(raw):,}; missing PV rows: {meta["missing_pv_rows"]}; incomplete PV days: {meta["incomplete_pv_days"]}. Source hashes, quality indicators and month ownership are retained under data/pv.</p><h2>Features</h2><p>Historical: daily mean, daily maximum and midday mean, each lagged one and seven days. Observed scenario additionally includes current daily mean, maximum, midday mean and the change in midday mean since yesterday. Features were specified before running this comparison.</p></html>'
(OUT/'REPORT.html').write_text(html)
(DATA/'README.txt').write_text('Rooftop PV source: local AEMO monthly ROOFTOP_PV_ACTUAL_FILE01.csv archives, NSW1 MEASUREMENT only. Units MW; interval endpoint shifted to interval start by 30 minutes. See source_manifest.json and pv_row_provenance.csv. Re-run experiments/run_pv_experiment.py; the saved joined CSV makes reruns independent of Desktop/database. Original eight columns remain unchanged. The observed-PV scenario is explanatory and cannot be used for forecasting without a PV forecast.\n')
print(metrics[['Variant','Target','Period','MAE_MW']].round(3).to_string(index=False));print(json.dumps(meta,indent=2))
