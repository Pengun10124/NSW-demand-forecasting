from pathlib import Path
import os,json,ast
os.environ['MPLBACKEND']='Agg';os.environ['MPLCONFIGDIR']='/tmp/nsw-comparison-mpl'
import pandas as pd,numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
repo=Path('/Users/maomengmeng/Desktop/codebase/NSW-demand-forecasting');os.chdir(repo)
out=Path('/Users/maomengmeng/Documents/ChatGPT/ENGG2122/nsw1-model-data/august_diagnosis');out.mkdir(exist_ok=True)
n=json.loads((repo/'notebooks/NSW_Minimum_Load_Combined_Fixed_Validation.ipynb').read_text());ns={}
for i in [2,8,12,20]:
 code=''.join(n['cells'][i]['source'])
 if i==2:code=code.replace('RESULT_BASE / "combined_fixed_validation"',repr(str(out)))
 if i==8:ns['raw']=pd.read_csv(repo/'data/combined/NSW1_Combined_Original_Priority.csv',parse_dates=['Interval_Start_AEST'])
 if i==12:code=code.replace('assert len(feat) == 1081','assert len(feat) == 1143')
 tree=ast.parse(code)
 if i==20:tree.body=[x for x in tree.body if isinstance(x,ast.FunctionDef)]
 exec(compile(tree,'cell','exec'),ns)
raw=ns['raw'];feat=ns['feat'];daily=ns['daily'];cols=ns['FEATURES'];tr=feat.loc[feat.index<'2026-04-01'];test=feat.loc['2026-07':'2026-08']
pred=pd.read_csv(repo/'experiments/results/fixed_validation_july_august_2026/daily_predictions.csv',parse_dates=['Date']);pred=pred[pred.Model.str.startswith('Ridge')].copy();pred['abs_error']=pred.Error_MW.abs()
# Compare every half-hour with local AEMO five-minute source files.
aemo=pd.concat([pd.read_csv(Path('/Users/maomengmeng/Desktop/database/AEMO')/f'2026-{m:02d}/NSW1'/f'PRICE_AND_DEMAND_2026{m:02d}_NSW1.csv',parse_dates=['SETTLEMENTDATE']) for m in [7,8]])
assert aemo.SETTLEMENTDATE.is_unique
agg=aemo.set_index('SETTLEMENTDATE').TOTALDEMAND.resample('30min',closed='right',label='right').agg(['mean','count']);agg.index-=pd.Timedelta(minutes=30)
check=raw.set_index('Interval_Start_AEST').loc['2026-07':'2026-08'].join(agg);check['difference']=check.Demand_MW-check['mean'];check.to_csv(out/'source_reconciliation.csv')
assert len(check)==2976 and check['count'].eq(6).all() and check.difference.abs().max()<1e-5
quality=pd.read_csv(repo/'data/provenance/quality_and_provenance.csv',parse_dates=['start']);quality=quality.set_index('start').loc['2026-07':'2026-08']
q=quality.groupby(quality.index.normalize()).agg(temp_missing=('temperature_missing','sum'),wind_filled=('weather_wind_speed_m_s_forward_filled','sum'),min_stations=('weather_wind_speed_m_s_station_count','min'))
# Source PV as explanatory evidence, not fitted using test targets.
core=pd.read_csv('/Users/maomengmeng/Desktop/NSW1_Core_Study_Data_30min_20261014/recent_20250901_20260831.csv',parse_dates=['SETTLEMENTDATE'])
core['Date']=(core.SETTLEMENTDATE-pd.Timedelta(minutes=30)).dt.normalize();core['slot']=(core.SETTLEMENTDATE-pd.Timedelta(minutes=30)).dt.hour*2+(core.SETTLEMENTDATE-pd.Timedelta(minutes=30)).dt.minute//30
solar=core[core.slot.between(20,29)].groupby('Date').rooftop_pv_power_mw.agg(['mean','max','count']).rename(columns=lambda c:'pv_mid_'+c)
d=daily.join(q).join(solar);d['day_jump']=d.day_min.diff();d['mid_jump']=d.mid_min.diff();d['pv_jump']=d.pv_mid_mean.diff()
d.loc['2026-07':'2026-08'].to_csv(out/'daily_diagnostics.csv')
err=pred.join(d,on='Date');err.nlargest(12,'abs_error').to_csv(out/'largest_errors.csv',index=False)
rows=[]
weather=['Temp_mean','Temp_max','Temp_min','Temp_midday','HDD','CDD','Wind_mean']
for name,train,features in [('Frozen',tr,cols),('No wind',tr,[c for c in cols if c!='Wind_mean']),('No weather',tr,[c for c in cols if c not in weather]),('Refit through June',feat.loc[feat.index<'2026-07-01'],cols)]:
 for t,label in ns['TARGETS'].items():
  m=ns['make_models']()['Linear Regression (Ridge)'].fit(train[features],train[t]);p=pd.Series(m.predict(test[features]),index=test.index)
  for month in [7,8]:
   idx=test.index.month==month
   rows.append({'experiment':name,'target':t,'month':month,'MAE':mean_absolute_error(test.loc[idx,t],p[idx]),'bias':(p[idx]-test.loc[idx,t]).mean()})
ab=pd.DataFrame(rows);ab.to_csv(out/'diagnostic_experiments.csv',index=False)
summary=[]
for month in [7,8]:
 a=d.loc[f'2026-{month:02d}'];r=raw[raw.Interval_Start_AEST.dt.month.eq(month)&raw.Interval_Start_AEST.dt.year.eq(2026)]
 summary.append({'month':month,'temp_missing':int(a.temp_missing.sum()),'wind_filled':int(a.wind_filled.sum()),'days_temp_incomplete':int(a.temp_missing.gt(0).sum()),'temp_min':r.Sydney_Airport_Temp_C.min(),'temp_max':r.Sydney_Airport_Temp_C.max(),'mean_abs_daily_min_change':a.day_jump.abs().mean(),'mean_abs_midday_min_change':a.mid_jump.abs().mean(),'daily_min_std':a.day_min.std(),'midday_min_std':a.mid_min.std(),'min_hour_midday_fraction':a.min_hour.between(10,14.5).mean(),'mean_pv':a.pv_mid_mean.mean(),'pv_jump_abs_mean':a.pv_jump.abs().mean()})
pd.DataFrame(summary).to_csv(out/'monthly_diagnostics.csv',index=False)
print('SOURCE_MAX_DIFF',check.difference.abs().max());print(pd.DataFrame(summary).round(3).to_string(index=False));print(ab.round(3).to_string(index=False))
print('TOP ERRORS\n',err.nlargest(12,'abs_error')[['Date','Target','Actual_MW','Prediction_MW','Error_MW','temp_missing','wind_filled','Temp_mean','pv_mid_mean','pv_jump','day_jump','mid_jump']].round(2).to_string(index=False))
for t,label in ns['TARGETS'].items():
 a=err[(err.Target==label)&err.Date.dt.month.eq(8)];print(t,'top5_share',a.nlargest(5,'abs_error').abs_error.sum()/a.abs_error.sum(),'complete_temp_MAE',a[a.temp_missing.eq(0)].abs_error.mean(),'incomplete_temp_MAE',a[a.temp_missing.gt(0)].abs_error.mean(),'corr_pv_error',a.Error_MW.corr(a.pv_mid_mean),'corr_pv_jump_error',a.Error_MW.corr(a.pv_jump))
# Same-day PV is an explanatory oracle experiment, not an operational forecast input.
core_all=[]
for f in ['baseline_2023.csv','baseline_2025.csv','recent_20250901_20260831.csv']:
 a=pd.read_csv(Path('/Users/maomengmeng/Desktop/NSW1_Core_Study_Data_30min_20261014')/f,parse_dates=['SETTLEMENTDATE'])
 a['start']=a.SETTLEMENTDATE-pd.Timedelta(minutes=30);core_all.append(a)
a=pd.concat(core_all).drop_duplicates('start',keep='last');a['Date']=a.start.dt.normalize();a['slot']=a.start.dt.hour*2+a.start.dt.minute//30
pv=a[a.slot.between(20,29)].groupby('Date').rooftop_pv_power_mw.agg(['mean','count']);pv=pv[pv['count']==10]['mean'].rename('Observed_midday_PV')
f=feat.join(pv);train_pv=f[(f.index<'2026-04-01')&f.Observed_midday_PV.notna()];test_pv=f.loc['2026-07':'2026-08'];assert test_pv.Observed_midday_PV.notna().all()
extra=[]
for include in [False,True]:
 use=cols+(['Observed_midday_PV'] if include else [])
 for t in ns['TARGETS']:
  fit=ns['make_models']()['Linear Regression (Ridge)'].fit(train_pv[use],train_pv[t]);p=pd.Series(fit.predict(test_pv[use]),index=test_pv.index)
  for month in [7,8]:
   idx=test_pv.index.month==month
   extra.append({'target':t,'month':month,'training_days':len(train_pv),'observed_PV_added':include,'MAE':mean_absolute_error(test_pv.loc[idx,t],p[idx])})
pd.DataFrame(extra).to_csv(out/'observed_pv_diagnostic_only.csv',index=False)
print('OBSERVED PV DIAGNOSTIC - matched training days, not a deployable forecasting result\n',pd.DataFrame(extra).round(3).to_string(index=False))
