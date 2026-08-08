"""S17-N41: is member-optimality learnable by a past-only gate?"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
from xgboost import XGBClassifier
from baram.constants import CAPACITIES_KWH
from baram.evaluation.official import evaluate_official

REPO=Path('.')
CACHE=REPO/'artifacts/cache/920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b'
MEM=['D','M102_TOP100','M113_LGBM_DART','M115_XGBOOST']
W_DEPLOY=np.array([0.30,0.7/3,0.7/3,0.7/3])

def score(df,v):
    fr=pd.DataFrame({'forecast_id':df['forecast_kst_dtm'].astype(str)+'_'+df['group_id'].astype(str),
        'forecast_kst_dtm':df['forecast_kst_dtm'],'group_id':df['group_id'],
        'actual_kwh':df['actual_kwh'],'prediction_kwh':v})
    s=evaluate_official(fr,CAPACITIES_KWH)
    return {'total':float(s.total),'one_minus_nmae':float(s.one_minus_nmae),'ficr':float(s.ficr)}

def ctx():
    cols=['forecast_kst_dtm','group_id','hour','month','lead_hour',
          'ldaps_spatial__wind10_speed__idw','gfs_spatial__wind10_speed__idw',
          'source_disagreement__wind10_speed_idw__abs']
    have=set(pq.ParquetFile(CACHE/'train_features.parquet').schema.names)
    cols=[c for c in cols if c in have]
    b=next(pq.ParquetFile(CACHE/'train_features.parquet').iter_batches(batch_size=52_560,columns=cols,use_threads=False))
    d=b.to_pandas(); d['forecast_kst_dtm']=pd.to_datetime(d['forecast_kst_dtm']); return d,cols

def main():
    A=pd.read_parquet(REPO/'artifacts/backtests/s17_n7_strict_actions/actions.parquet')
    A['forecast_kst_dtm']=pd.to_datetime(A['forecast_kst_dtm'])
    C,ccols=ctx()
    A=A.merge(C,on=['forecast_kst_dtm','group_id'],how='left',validate='one_to_one')
    A['operating_day']=(A['forecast_kst_dtm']-pd.Timedelta(hours=1)).dt.normalize()
    A['label_available']=A['operating_day']+pd.Timedelta(days=1)
    M=A[MEM].to_numpy(float); cap=A['group_id'].map(CAPACITIES_KWH).to_numpy(float)
    err=np.abs(M-A['actual_kwh'].to_numpy(float)[:,None])/cap[:,None]
    A['best_member']=np.argmin(err,axis=1)
    A['spread']=M.std(axis=1); A['rng']=M.max(axis=1)-M.min(axis=1); A['mean_act']=M.mean(axis=1)
    for i,m in enumerate(MEM): A[f'dev_{m}']=M[:,i]-A['mean_act']
    feats=[*[f'dev_{m}' for m in MEM],'spread','rng','mean_act',*[c for c in ccols if c not in ('forecast_kst_dtm','group_id')],'group_id']
    out={}
    for target in ['dev-2023-Q3','dev-2023-Q4']:
        te=A[A['fold_id'].eq(target)].copy()
        first_basis=(te['operating_day'].min()-pd.Timedelta(days=1)+pd.Timedelta(hours=14))
        tr=A[(A['fold_id']!=target)&(A['operating_day']<te['operating_day'].min())&(A['label_available']<=first_basis)].copy()
        if len(tr)==0: raise RuntimeError('no training rows')
        if tr['label_available'].max()>first_basis: raise RuntimeError('label leak')
        g=XGBClassifier(objective='multi:softprob',num_class=4,n_estimators=300,learning_rate=0.05,
            max_depth=4,subsample=0.9,colsample_bytree=0.8,reg_lambda=5.0,tree_method='hist',
            random_state=20260809,n_jobs=6)
        g.fit(tr[feats].astype('float32'),tr['best_member'].to_numpy(int))
        p=g.predict_proba(te[feats].astype('float32'))
        pick=p.argmax(axis=1)
        Mte=te[MEM].to_numpy(float)
        hard=Mte[np.arange(len(Mte)),pick]
        soft=(p*Mte).sum(axis=1)
        shrink=0.5*soft+0.5*(Mte@W_DEPLOY)
        acc=float((pick==te['best_member'].to_numpy(int)).mean())
        base_rate=float(pd.Series(te['best_member']).value_counts(normalize=True).max())
        out[target]={'training_rows':int(len(tr)),'train_label_max':tr['label_available'].max().isoformat(),
            'first_basis':first_basis.isoformat(),'gate_accuracy':acc,'majority_baseline':base_rate,
            'deployed':score(te,Mte@W_DEPLOY),'gate_hard':score(te,hard),'gate_soft':score(te,soft),
            'gate_soft_shrunk':score(te,shrink),
            'oracle_per_row':score(te,Mte[np.arange(len(Mte)),te['best_member'].to_numpy(int)])}
        print(target,json.dumps({k:(v if not isinstance(v,dict) else round(v['total'],7)) for k,v in out[target].items()}))
    both=A[A['fold_id'].isin(['dev-2023-Q3','dev-2023-Q4'])]
    Path('reports/s17_n41_gate.json').write_text(json.dumps(out,indent=1)+'\n')

main()
