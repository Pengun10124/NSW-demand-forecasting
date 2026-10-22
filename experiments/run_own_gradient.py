"""Train only the existing Histogram Gradient Boosting regressor on own-source data."""
from pathlib import Path
import os,json,hashlib,argparse
os.environ.setdefault('MPLBACKEND','Agg');os.environ.setdefault('MPLCONFIGDIR','/tmp/nsw-comparison-mpl')
import numpy as np,pandas as pd,joblib,sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--output',default='experiments/results/own_gradient_90min');args=parser.parse_args()
OUT=ROOT/args.output;OUT.mkdir(parents=True,exist_ok=False)
DATA=ROOT/'data/own_reprocessed'
features=json.loads((DATA/'feature_columns.json').read_text())
frame=pd.read_csv(DATA/'daily_model_ready.csv',parse_dates=['Date']).set_index('Date').sort_index()
assert frame.index.is_unique and len(features)==26
assert frame[features+['day_min','mid_min']].notna().all().all()
assert not any('pv' in f.lower() for f in features)
train=frame[frame.index<'2026-04-01'];assert len(train)==697
TARGETS={'day_min':'Daily minimum (any time)','mid_min':'Midday minimum (10:00-15:00)'}
PARAMS={'max_iter':300,'learning_rate':0.05,'random_state':42}
windows=[('2024-10-23','2025-02-04'),('2025-02-05','2025-05-20'),('2025-05-21','2025-09-02'),('2025-09-03','2025-12-16'),('2025-12-17','2026-03-31')]
periods=[('April-June','2026-04-01','2026-06-30'),('July','2026-07-01','2026-07-31'),('August','2026-08-01','2026-08-31'),('July-August','2026-07-01','2026-08-31')]
metrics=[];predictions=[];folds=[];oof=[]
def score(y,p):
 e=np.asarray(p)-np.asarray(y)
 return dict(Days=len(e),MAE_MW=float(np.abs(e).mean()),RMSE_MW=float(np.sqrt(np.mean(e**2))),Bias_MW=float(e.mean()),R2=float(r2_score(y,p)))
for target,label in TARGETS.items():
 cvactual=[];cvpred=[]
 for k,(start,end) in enumerate(windows,1):
  tr=train[train.index<start];va=train.loc[start:end]
  if tr.empty or va.empty:raise ValueError('Empty validation fold')
  assert tr.index.max()<va.index.min()
  m=HistGradientBoostingRegressor(**PARAMS).fit(tr[features],tr[target]);p=m.predict(va[features])
  folds.append(dict(Target=label,Fold=k,Train_days=len(tr),Window_start=start,Window_end=end,Actual_start=str(va.index.min().date()),Actual_end=str(va.index.max().date()),**score(va[target],p)))
  cvactual.extend(va[target]);cvpred.extend(p)
  oof.extend(dict(Date=str(dt.date()),Target=label,Fold=k,Actual_MW=float(y),Predicted_MW=float(v)) for dt,y,v in zip(va.index,va[target],p))
 metrics.append(dict(Target=label,Period='Fixed-window CV, available days',**score(cvactual,cvpred)))
 model=HistGradientBoostingRegressor(**PARAMS).fit(train[features],train[target])
 joblib.dump({'model':model,'features':features,'target':target,'training_end':'2026-03-31'},OUT/(target+'.joblib'))
 for name,start,end in periods:
  test=frame.loc[start:end];p=model.predict(test[features]);metrics.append(dict(Target=label,Period=name,**score(test[target],p)))
  if name!='July-August':predictions.extend(dict(Date=str(dt.date()),Target=label,Actual_MW=float(y),Predicted_MW=float(v),Error_MW=float(v-y)) for dt,y,v in zip(test.index,test[target],p))
 print('Completed Gradient Boosting:',label,flush=True)
results=pd.DataFrame(metrics);pred=pd.DataFrame(predictions)
results.to_csv(OUT/'metrics.csv',index=False);pred.to_csv(OUT/'daily_predictions.csv',index=False)
pd.DataFrame(folds).to_csv(OUT/'fold_scores.csv',index=False);pd.DataFrame(oof).to_csv(OUT/'cv_predictions.csv',index=False)
frame[features+list(TARGETS)].to_csv(OUT/'used_daily_features.csv')
# Test dates missing from evaluation are explicitly enumerated, never hidden.
elig=pd.read_csv(DATA/'daily_features_with_eligibility.csv',parse_dates=['Date']).set_index('Date')
excluded=[]
for dt in pd.date_range('2026-04-01','2026-08-31'):
 if dt not in frame.index:
  missing=[c for c in features+list(TARGETS) if pd.isna(elig.loc[dt,c])]
  excluded.append({'Date':str(dt.date()),'Reason':'Unavailable required fields: '+', '.join(missing)})
pd.DataFrame(excluded).to_csv(OUT/'excluded_test_dates.csv',index=False)
meta={'model_class':'sklearn.ensemble.HistGradientBoostingRegressor','sklearn_version':sklearn.__version__,'parameters':PARAMS,'features':features,'training_days':len(train),'training_cutoff_exclusive':'2026-04-01','data_sha256':hashlib.sha256((DATA/'daily_model_ready.csv').read_bytes()).hexdigest(),'original_fixed_windows':windows,'protocol':'Only Gradient Boosting fitted. No Ridge or other linear estimator is imported, fitted or selected. No tuning on test months. Fixed original windows with available own-source days only: counts differ from earlier runs because 2024 and excluded days are absent. Two targets trained separately. Same-day observed weather is an idealized retrospective input; earlier test observations supply later calendar-correct lags. No median imputation; incomplete feature rows are excluded.','limitations':'Previously inspected test months, not an untouched final test. Test dates are reduced by cleaning/history requirements; compare prior predictions only on identical dates. No donor workbook or donor-derived merged training table is used.'}
(OUT/'metadata.json').write_text(json.dumps(meta,indent=2))
fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True)
for ax,label in zip(axes,TARGETS.values()):
 a=pred[(pred.Target==label)&pred.Date.ge('2026-07-01')].copy();a.Date=pd.to_datetime(a.Date)
 a=a.set_index('Date').reindex(pd.date_range('2026-07-01','2026-08-31'))
 ax.plot(a.index,a.Actual_MW,'o-',label='Actual',color='black',markersize=3)
 ax.plot(a.index,a.Predicted_MW,'o-',label='Gradient Boosting',markersize=3)
 ax.set_title(label+' (gaps = excluded dates)');ax.set_ylabel('MW');ax.legend();ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'predictions.png',dpi=140);plt.close(fig)
(OUT/'REPORT.html').write_text('<html lang="en"><meta charset="utf-8"><title>Own-source Gradient Boosting</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto}td,th{padding:9px}img{max-width:100%}</style><h1>Own-source Gradient Boosting</h1><p>Only Histogram Gradient Boosting, original settings: 300 iterations, learning rate 0.05, seed 42. Both targets use 697 eligible training days before April 2026. No PV inputs. No linear models trained.</p><p>Data uses the 90-minute weather fill limit and whole-day exclusion rule. Historical features respect calendar gaps. Actual observed weather is used, not an archived forecast. April-August dates were already inspected in prior experiments. MAE is in MW; lower is better. Evaluation covers retained dates only, so full-period historical scores are not directly comparable.</p>'+results.round(3).to_html(index=False,border=0)+'<h2>Validation fold coverage</h2>'+pd.DataFrame(folds).round(3).to_html(index=False,border=0)+'<img src="predictions.png"></html>')
print(results.round(3).to_string(index=False))
