"""S17-N42: metric-aligned conditional combiner over candidate actions."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
from xgboost import XGBRegressor
from baram.constants import CAPACITIES_KWH
from baram.evaluation.official import evaluate_official

REPO=Path('.')
CACHE=REPO/'artifacts/cache/920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b'
MEM=['D','M102_TOP100','M113_LGBM_DART','M115_XGBOOST']
W=np.array([0.30,0.7/3,0.7/3,0.7/3])
CAND=[*MEM,'BLEND']

def score(df,v):
    fr=pd.DataFrame({'forecast_id':df['forecast_kst_dtm'].astype(str)+'_'+df['group_id'].astype(str),
        'forecast_kst_dtm':df['forecast_kst_dtm'],'group_id':df['group_id'],
        'actual_kwh':df['actual_kwh'],'prediction_kwh':v})
    s=evaluate_official(fr,CAPACITIES_KWH)
    return {'total':float(s.total),'one_minus_nmae':float(s.one_minus_nmae),'ficr':float(s.ficr)}

def ctx():
    want=['forecast_kst_dtm','group_id','hour','month','lead_hour',
          'ldaps_spatial__wind10_speed__idw','gfs_spatial__wind10_speed__idw',
          'source_disagreement__wind10_speed_idw__abs']
    have=set(pq.ParquetFile(CACHE/'train_features.parquet').schema.names)
    cols=[c for c in want if c in have]
    b=next(pq.ParquetFile(CACHE/'train_features.parquet').iter_batches(batch_size=52_560,columns=cols,use_threads=False))
    d=b.to_pandas(); d['forecast_kst_dtm']=pd.to_datetime(d['forecast_kst_dtm']); return d,cols

def main():
    A=pd.read_parquet(REPO/'artifacts/backtests/s17_n7_strict_actions/actions.parquet')
    A['forecast_kst_dtm']=pd.to_datetime(A['forecast_kst_dtm'])
    C,ccols=ctx(); A=A.merge(C,on=['forecast_kst_dtm','group_id'],how='left',validate='one_to_one')
    A['operating_day']=(A['forecast_kst_dtm']-pd.Timedelta(hours=1)).dt.normalize()
    A['label_available']=A['operating_day']+pd.Timedelta(days=1)
    A['BLEND']=A[MEM].to_numpy(float)@W
    cap=A['group_id'].map(CAPACITIES_KWH).to_numpy(float)
    A['y_cf']=A['actual_kwh'].to_numpy(float)/cap
    Mall=A[CAND].to_numpy(float)/cap[:,None]
    base=['hour','month','lead_hour','group_id',*[c for c in ccols if c.startswith(('ldaps_','gfs_','source_'))]]
    A['spread']=Mall.std(axis=1); A['rng']=Mall.max(axis=1)-Mall.min(axis=1)
    for i,m in enumerate(CAND): A[f'cf_{m}']=Mall[:,i]
    feats=[*base,'spread','rng',*[f'cf_{m}' for m in CAND]]
    out={}
    for target in ['dev-2023-Q3','dev-2023-Q4']:
        te=A[A['fold_id'].eq(target)].copy()
        fb=te['operating_day'].min()-pd.Timedelta(days=1)+pd.Timedelta(hours=14)
        tr=A[(A['operating_day']<te['operating_day'].min())&(A['label_available']<=fb)].copy()
        if tr['label_available'].max()>fb: raise RuntimeError('leak')
        mg={g:float(np.mean(tr.loc[(tr['group_id']==g)&(tr['y_cf']>=0.10),'y_cf'])) for g in (1,2,3)}
        def value(df,cand_cf):
            e=np.abs(cand_cf-df['y_cf'].to_numpy(float))
            un=np.select([e<=0.06,e<=0.08],[4.0,3.0],default=0.0)
            g=df['group_id'].to_numpy(int); m=np.array([mg[x] for x in g])
            return -e + df['y_cf'].to_numpy(float)*un/(4.0*m)
        stack=[]
        for j,m in enumerate(CAND):
            d=tr[feats].copy(); d['cand']=j
            d['v']=value(tr,tr[f'cf_{m}'].to_numpy(float)); stack.append(d)
        TR=pd.concat(stack,ignore_index=True)
        reg=XGBRegressor(n_estimators=400,learning_rate=0.05,max_depth=5,subsample=0.9,
            colsample_bytree=0.8,reg_lambda=5.0,tree_method='hist',random_state=20260809,n_jobs=6)
        reg.fit(TR[[*feats,'cand']].astype('float32'),TR['v'].to_numpy(float))
        pv=np.column_stack([reg.predict(te[feats].assign(cand=j)[[*feats,'cand']].astype('float32')) for j in range(len(CAND))])
        Mte=te[CAND].to_numpy(float)
        pick=pv.argmax(axis=1)
        hard=Mte[np.arange(len(Mte)),pick]
        shrink=0.5*hard+0.5*te['BLEND'].to_numpy(float)
        # oracle over the same candidate set
        ev=value(te,None) if False else None
        e_all=np.abs(te[CAND].to_numpy(float)/te['group_id'].map(CAPACITIES_KWH).to_numpy(float)[:,None]-te['y_cf'].to_numpy(float)[:,None])
        un=np.select([e_all<=0.06,e_all<=0.08],[4.0,3.0],default=0.0)
        gm=np.array([mg[x] for x in te['group_id'].to_numpy(int)])[:,None]
        v_true=-e_all+te['y_cf'].to_numpy(float)[:,None]*un/(4.0*gm)
        orc=Mte[np.arange(len(Mte)),v_true.argmax(axis=1)]
        out[target]={'training_rows':int(len(tr)),'train_label_max':tr['label_available'].max().isoformat(),
          'first_basis':fb.isoformat(),
          'pick_distribution':{CAND[i]:int((pick==i).sum()) for i in range(len(CAND))},
          'value_argmax_accuracy':float((pick==v_true.argmax(axis=1)).mean()),
          'deployed':score(te,te['BLEND'].to_numpy(float)),
          'combiner_hard':score(te,hard),'combiner_shrunk':score(te,shrink),
          'oracle_value_pick':score(te,orc)}
        print(target,json.dumps({k:(round(v['total'],7) if isinstance(v,dict) and 'total' in v else v) for k,v in out[target].items()}))
    Path('reports/s17_n42_metric_aligned_gate.json').write_text(json.dumps(out,indent=1)+'\n')

main()
