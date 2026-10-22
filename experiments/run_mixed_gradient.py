from pathlib import Path
import tarfile,json,ast,os
os.environ.setdefault('MPLBACKEND','Agg');os.environ.setdefault('MPLCONFIGDIR','/tmp/nsw-comparison-mpl')
import pandas as pd,numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
ROOT=Path(__file__).resolve().parents[1]
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--output',default='experiments/results/mixed_recleaned_gradient');args=parser.parse_args()
OUT=ROOT/args.output;OUT.mkdir(parents=True,exist_ok=False)
donor=pd.read_excel(ROOT/'data/original/NSW_ML_Training_Data_2024_2026.xlsx',sheet_name='ML Training Data')
steps=json.loads((ROOT/'experiments/original_feature_steps.json').read_text())
own=pd.read_csv(ROOT/'data/own_reprocessed/model_inputs_all_unique.csv',parse_dates=['Interval_Start_AEST'])
donor=donor.rename(columns={'Wind_NSW_Weighted_mps':'Wind_speed_mps'});donor=donor[list(own.columns)];donor.Interval_Start_AEST=pd.to_datetime(donor.Interval_Start_AEST)
assert len(donor)==35040
own['Source']='Own reprocessed';donor['Source']='Original workbook'
raw=pd.concat([own,donor],ignore_index=True).drop_duplicates('Interval_Start_AEST',keep='last').sort_values('Interval_Start_AEST')
assert raw.Interval_Start_AEST.is_unique
raw.to_csv(OUT/'mixed_halfhour_inputs.csv',index=False)
ns={'pd':pd,'np':np,'raw':raw,'MIDDAY_SLOTS':range(20,30),'SYDNEY_LAT':-33.95,'TARGETS':{'day_min':'Daily minimum (any time)','mid_min':'Midday minimum (10:00-15:00)'}}
exec(steps['aggregate'],ns)
# Daily calendar gaps remain explicit; avoid shifting seven retained rows across exclusions.
d=ns['daily'];d=d.reindex(pd.date_range(d.index.min(),d.index.max(),freq='D'));d.index.name='Date'
source=steps['features'];exec(source,ns)
# Calendar neighbours derived from the known calendar, independently of missing weather days.
h=json.loads((ROOT/'data/own_reprocessed/holiday_calendar.json').read_text())
hset={pd.Timestamp(f'{y}-{v}') for y,vs in h.items() for v in vs}
# Original workbook supplies 2024 holiday dates.
hset.update(pd.DatetimeIndex(donor.loc[donor.Is_NSW_Public_Holiday.eq(1),'Interval_Start_AEST']).normalize())
d['Is_Holiday']=d.index.isin(hset).astype(int)
feat=ns['build_features'](d);cols=ns['FEATURES']
# Original Gradient Boosting handles original weather missing values natively.
required=['day_min','mid_min','day_min_lag1','mid_min_lag1','day_min_lag7','mid_min_lag7','day_min_roll7','mid_min_roll7','day_mean_lag1']
feat=feat.dropna(subset=required);train=feat[feat.index<'2026-04-01']
new=pd.read_csv(ROOT/'experiments/results/own_gradient_90min/daily_predictions.csv')
old=pd.read_csv(ROOT/'experiments/reference_predictions/previous_mixed_gradient.csv')
metrics=[];comparisons=[];predictions=[]
for target,label in ns['TARGETS'].items():
 m=HistGradientBoostingRegressor(max_iter=300,learning_rate=.05,random_state=42).fit(train[cols],train[target])
 test=feat.loc['2026-04-01':'2026-08-31'];pr=m.predict(test[cols]);p=pd.DataFrame({'Date':test.index.strftime('%Y-%m-%d'),'Target':label,'Actual_MW':test[target].values,'Predicted_MW':pr});predictions.append(p)
 for period,start,end in [('April-June','2026-04-01','2026-06-30'),('July','2026-07-01','2026-07-31'),('August','2026-08-01','2026-08-31'),('July-August','2026-07-01','2026-08-31')]:
  allp=p[p.Date.between(start,end)];metrics.append({'Target':label,'Period':period,'Days':len(allp),'MAE_MW':float((allp.Predicted_MW-allp.Actual_MW).abs().mean())})
  z=allp.merge(new[['Date','Target','Actual_MW','Predicted_MW']],on=['Date','Target'],suffixes=('_mixed','_own'),validate='one_to_one').merge(old[['Date','Target','Predicted_MW']],on=['Date','Target'],validate='one_to_one')
  assert np.allclose(z.Actual_MW_mixed,z.Actual_MW_own,atol=1e-5,rtol=0)
  comparisons.append({'Target':label,'Period':period,'Matched_days':len(z),'Mixed_recleaned_MAE_MW':float((z.Predicted_MW_mixed-z.Actual_MW_own).abs().mean()),'Own_only_MAE_MW':float((z.Predicted_MW_own-z.Actual_MW_own).abs().mean()),'Previous_mixed_MAE_MW':float((z.Predicted_MW-z.Actual_MW_own).abs().mean())})
 print('Completed',label,flush=True)
a=pd.DataFrame(metrics);c=pd.DataFrame(comparisons);a.to_csv(OUT/'metrics.csv',index=False);c.to_csv(OUT/'matched_comparison.csv',index=False);pd.concat(predictions).to_csv(OUT/'daily_predictions.csv',index=False)
meta={'training_days':len(train),'source_rows':raw.Source.value_counts().to_dict(),'training_cutoff':'2026-03-31','features':cols,'model':'HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42)','overlap':'Original workbook whole row wins. Own inputs use new 90-minute cleaning. Original workbook values are not reprocessed.','protocol':'No linear model, no tuning, same-day observed weather, calendar-correct past-demand features. Original weather NaNs handled natively by Gradient Boosting. Standalone reproducible mixed-data experiment; no upload performed by this script.'}
(OUT/'metadata.json').write_text(json.dumps(meta,indent=2))
(OUT/'REPORT.html').write_text('<html><meta charset="utf-8"><title>Original plus reprocessed own data</title><style>body{font:16px/1.6 system-ui;margin:40px}td,th{padding:9px}</style><h1>Original plus reprocessed own data: Gradient Boosting</h1><p>Original workbook wins overlapping timestamps. New own-source cleaning applies only to supplementary data. Both targets use the existing Gradient Boosting settings and training ends March 31. Observed weather is an idealized input. No linear models, tuning, push or branch changes.</p><p>Training days: '+str(len(train))+'</p><h2>All available test dates</h2>'+a.round(3).to_html(index=False,border=0)+'<h2>Same-date comparison, MAE in MW</h2>'+c.round(3).to_html(index=False,border=0)+'</html>')
print('Training days',len(train));print(c.round(3).to_string(index=False))
