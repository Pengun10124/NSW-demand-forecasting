"""Run unchanged upstream model grids against original and added local inputs."""
from pathlib import Path
import os, json, time, re, html, contextlib, hashlib
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MPLCONFIGDIR','/tmp/nsw-comparison-mpl')
import pandas as pd
from joblib import parallel_config

REPO=Path(__file__).resolve().parents[1]
OUT=REPO/'experiments/results'
OUT.mkdir(parents=True,exist_ok=True)
nb=json.loads((REPO/'notebooks/NSW_Minimum_Load_Forecasting.ipynb').read_text())
CONFIGS=[
 {'id':'original_2024_2026','label':'Original data','file':'NSW_ML_Training_Data_2024_2026.xlsx','sheet':'ML Training Data','cutoff':'2026-04-01','end':'2026-07-01','wind':'Wind_NSW_Weighted_mps'},
 {'id':'our_2025_2026','label':'Our main data','file':'NSW1_Minimum_Load_Inputs.xlsx','sheet':'ML Training Data','cutoff':'2026-04-01','end':'2026-07-01','wind':'Wind_NSW_Mean_mps'},
 {'id':'our_baseline_2023','label':'Our 2023 baseline','file':'NSW1_Minimum_Load_Inputs.xlsx','sheet':'Baseline 2023','cutoff':'2023-10-01','end':'2024-01-01','wind':'Wind_NSW_Mean_mps'},
]
summaries=[]; results=[]; classifications=[]
for config in CONFIGS:
    folder=OUT/config['id'];folder.mkdir(exist_ok=True)
    ns={'__name__':'__main__'}
    start=time.monotonic()
    print('RUN '+config['label'],flush=True)
    with (folder/'run.log').open('w') as log, contextlib.redirect_stdout(log), parallel_config(backend='threading',n_jobs=2):
        for i,c in enumerate(nb['cells']):
            if c['cell_type']!='code':continue
            code=''.join(c['source'])
            if i==2:
                code=code.replace('HOLDOUT_START = "2026-04-01"',f'HOLDOUT_START = {config["cutoff"]!r}')
                code=code.replace('RESULTS_DIR = "results"',f'RESULTS_DIR = {str(folder)!r}')
            if i==4:
                raw=pd.read_excel(REPO/'data'/config['file'],sheet_name=config['sheet'])
                raw['Interval_Start_AEST']=pd.to_datetime(raw['Interval_Start_AEST'],errors='raise')
                raw=raw[raw.Interval_Start_AEST<pd.Timestamp(config['end'])].sort_values('Interval_Start_AEST').reset_index(drop=True)
                ns['raw']=raw
                print(f'Loaded {config["file"]}, {config["sheet"]}: {len(raw)} rows')
                continue
            if i==8:code=code.replace('Wind_NSW_Weighted_mps',config['wind'])
            # Data-dependent chart date label; model code and grid remain unchanged.
            if i==10:code=code.replace('Jul 2024 to Jun 2026',config['label'])
            exec(compile(code,f'{config["id"]}_cell_{i}','exec'),ns)
            if i==20:print(f'{config["label"]}: model search complete',file=__import__('sys').__stdout__,flush=True)
            ns['plt'].close('all')
    meta={**config,'rows':len(raw),'first_interval':str(raw.Interval_Start_AEST.min()),'last_interval':str(raw.Interval_Start_AEST.max()),
          'train_days':len(ns['train']),'holdout_days':len(ns['hold']),
          'missing_temperature_half_hours':int(raw.Sydney_Airport_Temp_C.isna().sum()),
          'missing_wind_half_hours':int(raw[config['wind']].isna().sum()),
          'thresholds_mw':ns['THRESHOLDS'],'seconds':round(time.monotonic()-start,1),
          'source_sha256':hashlib.sha256((REPO/'data'/config['file']).read_bytes()).hexdigest()}
    summaries.append(meta)
    (folder/'run_metadata.json').write_text(json.dumps(meta,indent=2))
    for target,tname in ns['TARGETS'].items():
        reg=ns['hold_table']; row=reg[(reg.Target==tname)&(reg.Model==ns['best_reg'][target])].iloc[0].to_dict()
        row={'Dataset':config['label'],**row,'Training days':len(ns['train']),'Test days':len(ns['hold']),'Test start':config['cutoff'],'Test end':str(ns['hold'].index.max().date())}
        results.append(row)
    cl=ns['hold_clf_table'].copy();cl.insert(0,'Dataset',config['label']);classifications.extend(cl.to_dict('records'))
    print(f'PASS {config["label"]}: {meta["train_days"]} train / {meta["holdout_days"]} test days, {meta["seconds"]}s',flush=True)

reg=pd.DataFrame(results);cls=pd.DataFrame(classifications)
reg.to_csv(OUT/'regression_comparison.csv',index=False)
cls.to_csv(OUT/'classification_comparison.csv',index=False)
(OUT/'experiment_metadata.json').write_text(json.dumps(summaries,indent=2))
body='''<!doctype html><html lang="en"><meta charset="utf-8"><title>NSW Dataset Trial Comparison</title><style>body{font:15px/1.6 system-ui;max-width:1400px;margin:40px auto;padding:0 25px;color:#203040}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:left}th{background:#eef3f6}h1,h2{color:#233e55}</style><h1>NSW dataset trials</h1><p>Three datasets were run through the upstream model definitions, tuning grids and seed 42. Both original source files and the original notebook are preserved. These are retrospective experiments with observed weather.</p><h2>How to compare</h2><ul><li>The original and our main data use the same April–June 2026 hold-out (91 days), but the original training starts in July 2024 and ours in January 2025. Weather sources and demand preparation also differ. Differences in errors do not isolate data quality.</li><li>The 2023 baseline uses October–December 2023 (92 days). Its season, year and training history differ: treat it as a separate period trial, not a direct ranking.</li><li>The additional interactive notebook still defaults to June–August 2026. This comparison script explicitly uses April–June for the overlapping period.</li><li>No models were retuned after reviewing hold-out scores. Upstream CV-based model selection and full-training 10% risk thresholds are retained; this is not nested validation. A statistical low-load label is not an official grid safety threshold.</li><li>When no low-load positives occur, recall/F1 returned as zero are conventions and ROC-AUC is undefined. Inspect positive counts before interpreting classification.</li></ul><h2>Hold-out regression: selected model per dataset</h2>'''
body+=reg.round(3).to_html(index=False,border=0)
body+='<h2>Hold-out classification</h2>'+cls.round(3).to_html(index=False,border=0,na_rep='Not defined')
body+='<h2>Coverage</h2>'+pd.DataFrame([{k:m[k] for k in ['label','first_interval','last_interval','train_days','holdout_days','missing_temperature_half_hours','missing_wind_half_hours']} for m in summaries]).to_html(index=False,border=0)
body+='<p>Each run folder contains detailed CV/hold-out CSVs, seven figures, a log and source hashes. Model scores reflect this run only and do not establish causal effects. The data guide and source records are in ../../data/.</p></html>'
(OUT/'COMPARISON.html').write_text(body)
print(reg[['Dataset','Target','Model','MAE (MW)','MAPE (%)']].round(2).to_string(index=False),flush=True)
print(cls.to_string(index=False),flush=True)
print('ALL THREE TRIALS PASSED',flush=True)
