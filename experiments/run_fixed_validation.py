"""Execute only the new fixed-date notebook; preserve earlier experiments."""
from pathlib import Path
import os,json,io,ast,contextlib,time,hashlib,base64
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR','/tmp/nsw-comparison-mpl')
from joblib import parallel_config
import pandas as pd
REPO=Path(__file__).resolve().parents[1]
os.chdir(REPO)
NB=REPO/'notebooks/NSW_Minimum_Load_Combined_Fixed_Validation.ipynb'
OUT=REPO/'experiments/results/combined_fixed_validation'
OUT.mkdir(parents=True,exist_ok=True)
protected=[p for folder in ['data','notebooks','experiments/results'] for p in (REPO/folder).rglob('*') if p.is_file() and p!=NB and OUT not in p.parents]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
before={str(p):digest(p) for p in protected}
nb=json.loads(NB.read_text());ns={'__name__':'__main__'};start=time.monotonic();count=0
with (OUT/'run.log').open('w') as log,parallel_config(backend='threading',n_jobs=2):
 for i,c in enumerate(nb['cells']):
  if c['cell_type']!='code':continue
  count+=1;buf=io.StringIO();outputs=[]
  print(f'Executing cell {i}',flush=True)
  with contextlib.redirect_stdout(buf):
   tree=ast.parse(''.join(c['source']));last=tree.body.pop() if tree.body and isinstance(tree.body[-1],ast.Expr) else None
   exec(compile(tree,f'fixed_cell_{i}','exec'),ns)
   val=eval(compile(ast.Expression(last.value),f'fixed_cell_{i}','eval'),ns) if last else None
  content=buf.getvalue();log.write(content);log.flush()
  if content:outputs.append({'output_type':'stream','name':'stdout','text':content.splitlines(keepends=True)})
  if val is not None:
   data={'text/plain':repr(val)}
   if hasattr(val,'_repr_html_'):data['text/html']=val._repr_html_()
   outputs.append({'output_type':'execute_result','execution_count':count,'metadata':{},'data':data})
  if 'plt' in ns:
   plt=ns['plt']
   for f in plt.get_fignums():
    png=io.BytesIO();plt.figure(f).savefig(png,format='png',dpi=100,bbox_inches='tight')
    outputs.append({'output_type':'display_data','metadata':{},'data':{'image/png':base64.b64encode(png.getvalue()).decode()}})
   plt.close('all')
  c['outputs']=outputs;c['execution_count']=count
# Test targets must match the original workbook, not a reprocessed replacement.
original=pd.read_excel(REPO/'data/NSW_ML_Training_Data_2024_2026.xlsx',sheet_name='ML Training Data')
original['Date']=pd.to_datetime(original.Interval_Start_AEST).dt.normalize()
a=original[original.Date.between('2026-04-01','2026-06-30')]
pd.testing.assert_series_equal(ns['hold'].day_min,a.groupby('Date').Demand_MW.min(),check_names=False)
pd.testing.assert_series_equal(ns['hold'].mid_min,a[a.TOD_Slot_0_47.between(20,29)].groupby('Date').Demand_MW.min(),check_names=False)
assert all(digest(Path(p))==v for p,v in before.items()),'An old file changed'
NB.write_text(json.dumps(nb,indent=1))
summary=ns['cv_table'][['Target','Model','MAE (MW)']].rename(columns={'MAE (MW)':'Validation MAE (MW)'}).merge(ns['hold_table'],on=['Target','Model']).rename(columns={'MAE (MW)':'Test MAE (MW)'})
chosen={name:ns['best_reg'][t] for t,name in ns['TARGETS'].items()}
summary['Selected by validation']=summary.apply(lambda r:r.Model==chosen[r.Target],axis=1)
summary.to_csv(OUT/'model_comparison.csv',index=False)
meta={'train_days':len(ns['train']),'test_days':len(ns['hold']),'test_start':'2026-04-01','test_end':'2026-06-30','selected_models':chosen,'tuned_parameters':{t:m.get_params() for t,m in ns['tuned'].items()},'all_cells_passed':True,'original_test_targets_verified':True,'old_files_verified_unchanged':len(before),'combined_input_sha256':digest(ns['DATA_PATH']),'elapsed_seconds':round(time.monotonic()-start,1)}
(OUT/'run_metadata.json').write_text(json.dumps(meta,indent=2))
folds=pd.read_csv(OUT/'fixed_validation_windows.csv')
html='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Fixed-date validation comparison</title><style>body{font:15px/1.6 system-ui;max-width:1300px;margin:40px auto;padding:0 20px}table{border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf3f8}</style><h1>Combined data: fixed validation dates</h1><p>Original rows win all overlaps. Train: 990 days, including 2023. Test: 91 days, April–June 2026. July–August 2026 is excluded. Lag features reset at the January–June 2024 gap. All previous files are preserved and all test demand targets match the original workbook.</p><h2>Fixed validation windows</h2>'''
html+=folds.to_html(index=False,border=0)+'<h2>Regression model comparison</h2><p>Lower MAE is better. Models are selected by validation MAE only. The gradient boosting grid is retuned using these fixed windows.</p>'+summary.round(3).to_html(index=False,border=0)
html+='<h2>Interpretation and limitations</h2><p>This predicts the daily minimum and the 10:00–15:00 minimum, not the full 48-point daily demand curve. The test period has already been inspected, so this is an exploratory comparison. Observed weather substitutes for forecasts; wind definitions differ between sources. Tuning and reporting use the same validation folds, not nested CV. Classification retains the full-training quantile threshold.</p><h2>Classification</h2>'+ns['hold_clf_table'].round(3).to_html(index=False,border=0)+'</html>'
(OUT/'COMPARISON.html').write_text(html)
print(summary.to_string(index=False));print(json.dumps(meta,indent=2))
