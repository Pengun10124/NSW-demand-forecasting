"""Frozen fixed-validation Ridge evaluation on our July-August 2026 data."""
from pathlib import Path
import os,json,ast,hashlib
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR','/tmp/nsw-comparison-mpl')
import pandas as pd
import numpy as np
REPO=Path(__file__).resolve().parents[1];os.chdir(REPO)
OUT=REPO/'experiments/results/fixed_validation_july_august_2026'
OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protected={str(p):sha(p) for folder in ['data','notebooks','experiments/results'] for p in (REPO/folder).rglob('*') if p.is_file() and OUT not in p.parents}
nb=json.loads((REPO/'notebooks/NSW_Minimum_Load_Combined_Fixed_Validation.ipynb').read_text());ns={}
def run(i,definitions=False):
 code=''.join(nb['cells'][i]['source'])
 if i==2:code=code.replace('RESULT_BASE / "combined_fixed_validation"',repr(str(OUT)))
 if i==12:code=code.replace('assert len(feat) == 1081','assert len(feat) == 1143')
 tree=ast.parse(code)
 if definitions:tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
 exec(compile(tree,f'notebook_cell_{i}','exec'),ns)
run(2)
raw=pd.read_csv(ns['DATA_PATH'],parse_dates=['Interval_Start_AEST']);ns['raw']=raw
assert raw.Interval_Start_AEST.is_unique and raw.Interval_Start_AEST.is_monotonic_increasing
assert raw.groupby(raw.Interval_Start_AEST.dt.normalize()).size().eq(48).all()
test_raw=raw[raw.Interval_Start_AEST.between('2026-07-01','2026-08-31 23:30')]
assert len(test_raw)==2976 and test_raw.Demand_MW.notna().all()
assert pd.DatetimeIndex(test_raw.Interval_Start_AEST).equals(pd.date_range('2026-07-01','2026-08-31 23:30',freq='30min'))
prov=pd.read_csv(REPO/'data/combined/row_provenance.csv',parse_dates=['Interval_Start_AEST'])
assert prov[prov.Interval_Start_AEST.isin(test_raw.Interval_Start_AEST)].source.eq('our_main').all()
run(8);run(12);run(16,True);run(18,True);run(20,True)
feat=ns['feat'];features=ns['FEATURES'];train=feat[feat.index<'2026-04-01'];test=feat.loc['2026-07-01':'2026-08-31'];prior=feat.loc['2026-04-01':'2026-06-30']
assert len(train)==990 and len(test)==62
old=pd.read_csv(REPO/'experiments/results/combined_fixed_validation/holdout_regression_results.csv')
rows=[];predictions=[];models={}
for t,label in ns['TARGETS'].items():
 model=ns['make_models']()['Linear Regression (Ridge)'].fit(train[features],train[t]);models[t]=model
 old_mae=old[(old.Target==label)&(old.Model=='Linear Regression (Ridge)')]['MAE (MW)'].iloc[0]
 assert np.isclose(ns['regression_scores'](prior[t],model.predict(prior[features]))['MAE (MW)'],old_mae,atol=1e-7)
 for name,m in {'Ridge (frozen March 31)':model,**ns['make_baselines'](t)}.items():
  if name!='Ridge (frozen March 31)':m.fit(train[features],train[t])
  p=pd.Series(m.predict(test[features]),index=test.index)
  for period,part in [('July-August',test),('July',test.loc['2026-07']),('August',test.loc['2026-08'])]:
   rows.append({'Target':label,'Model':name,'Period':period,'Days':len(part),**ns['regression_scores'](part[t],p.loc[part.index]),'Bias (MW)':float((p.loc[part.index]-part[t]).mean())})
  for d in test.index:predictions.append({'Date':str(d.date()),'Target':label,'Model':name,'Actual_MW':test.loc[d,t],'Prediction_MW':p.loc[d],'Error_MW':p.loc[d]-test.loc[d,t]})
metrics=pd.DataFrame(rows);pred=pd.DataFrame(predictions)
metrics.to_csv(OUT/'metrics.csv',index=False);pred.to_csv(OUT/'daily_predictions.csv',index=False)
test[features].to_csv(OUT/'test_features.csv')
test_raw.isna().sum().rename('missing_half_hour_records').to_csv(OUT/'missing_inputs.csv')
plt=ns['plt'];fig,axes=plt.subplots(2,1,figsize=(13,8))
for ax,(t,label) in zip(axes,ns['TARGETS'].items()):
 ax.plot(test.index,test[t],label='Actual');ax.plot(test.index,models[t].predict(test[features]),label='Frozen Ridge');ax.set_title(label);ax.set_ylabel('MW');ax.legend()
fig.tight_layout();fig.savefig(OUT/'predictions.png',dpi=150);plt.close('all')
assert all(sha(Path(p))==v for p,v in protected.items())
meta={'training_cutoff_exclusive':'2026-04-01','training_days':990,'test_days':62,'test_rows':2976,'test_source':'our_main','old_test_scores_reproduced':True,'old_files_unchanged':len(protected),'new_model_selection_or_tuning':False,'input_sha256':sha(ns['DATA_PATH']),'notebook_sha256':sha(REPO/'notebooks/NSW_Minimum_Load_Combined_Fixed_Validation.ipynb'),'protocol':'Frozen Ridge alpha=10, original preprocessing; rolling one-day evaluation using observed previous-day demand and observed same-day weather. No July/August targets used for model fitting. July/August earlier days are available as lag inputs for later days.'}
(OUT/'metadata.json').write_text(json.dumps(meta,indent=2))
text='''<!doctype html><html lang="en"><meta charset="utf-8"><title>July-August 2026 frozen model test</title><style>body{font:15px/1.6 system-ui;max-width:1250px;margin:40px auto;padding:0 20px}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd}img{max-width:100%}</style><h1>July-August 2026: frozen fixed-validation model</h1><p>Training is unchanged: 990 days before April 1, 2026. Ridge alpha=10 was previously selected using fixed validation windows. No retuning, no fitting on April-August data. Reconstructed models reproduce the previous April-June test MAEs. All prior files remain unchanged.</p><p>Test: 62 complete days, 2,976 half-hour records from our data. Targets are daily minimum and 10:00-15:00 minimum. This is rolling one-day evaluation with previously observed demand lags and same-day observed weather, not a two-month-ahead forecast. Prior runs have used portions of these data, so this is not a wholly untouched research holdout.</p><h2>Selected model: overall and monthly scores</h2>'''
text+=metrics[metrics.Model.str.startswith('Ridge')].round(3).to_html(index=False,border=0)
text+='<h2>Baseline comparison</h2>'+metrics[metrics.Period=='July-August'].round(3).to_html(index=False,border=0)
text+='<h2>Daily predictions</h2><img src="predictions.png"><h2>Input quality and interpretation</h2><p>Wind changes from original weighted regional wind to our station mean; inherited filling and weather source differences remain. Missing feature values use the existing training-fitted median imputer. Scores across April-June and July-August reflect different seasons and data sources, so they do not isolate model changes. Positive bias means overprediction. See missing_inputs.csv, test_features.csv, daily_predictions.csv and metadata.json for the audit trail.</p></html>'
(OUT/'REPORT.html').write_text(text)
print(metrics.round(3).to_string(index=False));print(json.dumps(meta,indent=2))
