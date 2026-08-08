"""S17-N36 T1 bounded screen: does the seq__/geom__ block move official Total on the inner fold?"""
from __future__ import annotations
import gc, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
from xgboost import XGBClassifier
from baram.constants import CAPACITIES_KWH
from baram.evaluation.official import evaluate_official

REPO=Path('.')
CACHE=REPO/'artifacts/cache/920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b'
PROBE=REPO/'artifacts/backtests/metric-aligned-probe'
NEW=REPO/'artifacts/features/s17_n36'
PREFIX_ROWS=52_560; GRID_ROWS=17_520
ACTIONS_CF=np.arange(0.075,1.076,0.0025)
MODEL_PARAMS={'objective':'multi:softprob','n_estimators':100,'learning_rate':0.03,'max_depth':5,
 'min_child_weight':20.0,'subsample':0.9,'colsample_bytree':0.8,'reg_alpha':0.1,'reg_lambda':5.0,
 'max_bin':256,'tree_method':'hist','random_state':20260802,'n_jobs':6}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as s:
        while b:=s.read(8<<20): h.update(b)
    return h.hexdigest()

def prefix(path,cols,rows):
    pf=pq.ParquetFile(path)
    b=next(pf.iter_batches(batch_size=rows,columns=cols,use_threads=False))
    if b.num_rows!=rows: raise RuntimeError(f'prefix {path} {b.num_rows}/{rows}')
    return b.to_pandas()

def selected(fold):
    m=json.loads((PROBE/f'M115_XGBOOST-{fold}.json').read_text())
    n=list(m['selected_feature_names'])
    if len(n)!=100: raise RuntimeError('feature contract')
    return n

def build_surface(fold):
    names=selected(fold)
    wanted={x for x in names if not x.startswith('sitewind__')}
    fsch=set(pq.ParquetFile(CACHE/'train_features.parquet').schema.names)
    gsch=set(pq.ParquetFile(CACHE/'train_grid_pivot.parquet').schema.names)
    esch=set(pq.ParquetFile(CACHE/'train_geometric.parquet').schema.names)
    dyn={'group_1','group_2','group_3'}
    missing=sorted(wanted-fsch-gsch-esch-dyn)
    if missing: raise RuntimeError(f'missing {missing}')
    base=['forecast_id','forecast_kst_dtm','data_available_kst_dtm','issuance_batch','group_id']
    feats=prefix(CACHE/'train_features.parquet',list(dict.fromkeys([*base,*sorted(wanted&fsch)])),PREFIX_ROWS)
    grid=prefix(CACHE/'train_grid_pivot.parquet',list(dict.fromkeys(['forecast_kst_dtm',*sorted(wanted&gsch)])),GRID_ROWS)
    geo=prefix(CACHE/'train_geometric.parquet',list(dict.fromkeys(['forecast_kst_dtm','data_available_kst_dtm','group_id',*sorted(wanted&esch)])),PREFIX_ROWS)
    newp=NEW/'train_seq_geom.parquet'
    newcols=[c for c in pq.ParquetFile(newp).schema.names]
    new=prefix(newp,newcols,GRID_ROWS)
    for f in (feats,grid,geo,new): f['forecast_kst_dtm']=pd.to_datetime(f['forecast_kst_dtm'])
    for f in (feats,geo): f['data_available_kst_dtm']=pd.to_datetime(f['data_available_kst_dtm'])
    if new['forecast_kst_dtm'].max()>=pd.Timestamp('2024-01-01 00:00:01'): raise RuntimeError('2024 leak')
    if not new['forecast_kst_dtm'].equals(grid['forecast_kst_dtm']): raise RuntimeError('new/grid time mismatch')
    order=pd.MultiIndex.from_frame(feats[['forecast_kst_dtm','group_id']])
    s=feats.merge(grid,on='forecast_kst_dtm',validate='many_to_one') \
           .merge(geo,on=['forecast_kst_dtm','data_available_kst_dtm','group_id'],validate='one_to_one') \
           .merge(new,on='forecast_kst_dtm',validate='many_to_one')
    if not pd.MultiIndex.from_frame(s[['forecast_kst_dtm','group_id']]).equals(order): raise RuntimeError('merge order')
    for g in (1,2,3): s[f'group_{g}']=s['group_id'].eq(g).astype('int8')
    if len(s)!=PREFIX_ROWS or s['forecast_kst_dtm'].max()!=pd.Timestamp('2024-01-01 00:00:00'): raise RuntimeError('surface contract')
    seq=[c for c in new.columns if c.startswith('seq__')]; geom=[c for c in new.columns if c.startswith('geom__')]
    return s,names,seq,geom

def val_keys(fold):
    k=pd.read_parquet(PROBE/f'M115_XGBOOST-{fold}-policies.parquet',columns=['forecast_id','forecast_kst_dtm','group_id'])
    k['forecast_kst_dtm']=pd.to_datetime(k['forecast_kst_dtm']); return k

def labels_2022_2023():
    l=prefix(CACHE/'labels_long.parquet',['forecast_kst_dtm','group_id','actual_kwh','operating_year'],PREFIX_ROWS)
    l['forecast_kst_dtm']=pd.to_datetime(l['forecast_kst_dtm'])
    if l['operating_year'].max()>2023: raise RuntimeError('2024 label decoded')
    if l['forecast_kst_dtm'].max()!=pd.Timestamp('2024-01-01 00:00:00'): raise RuntimeError('label prefix end')
    return l

def contract(tr):
    cap=tr['group_id'].map(CAPACITIES_KWH).to_numpy(float); rate=tr['actual_kwh'].to_numpy(float)/cap
    el=np.isfinite(rate)&(rate>=0.10); cl=np.clip(rate,0.10,1.074999)
    rb=np.floor((cl-0.10)/0.02).astype(np.int16); ab=np.asarray(sorted(np.unique(rb[el])),dtype=np.int16)
    mp={int(b):i for i,b in enumerate(ab)}; classes=np.asarray([mp[int(v)] for v in rb[el]],dtype=np.int32)
    centers=np.asarray([float(np.mean(rate[el][classes==i])) for i in range(len(ab))],dtype=float)
    return el,classes,centers

def action(prob,centers,vg,tr):
    ec=centers>=0.10; prob=prob[:,ec].astype(float); prob/=prob.sum(axis=1,keepdims=True); centers=centers[ec]
    cal=prob**(1.0/0.75); cal/=cal.sum(axis=1,keepdims=True)
    err=np.abs(ACTIONS_CF[:,None]-centers[None,:]); units=np.select([err<=0.06,err<=0.08],[4.0,3.0],default=0.0)
    cap=tr['group_id'].map(CAPACITIES_KWH).to_numpy(float); rate=tr['actual_kwh'].to_numpy(float)/cap
    mg={g:float(np.mean(rate[(tr['group_id'].to_numpy()==g)&(rate>=0.10)])) for g in (1,2,3)}
    ch=np.empty(len(prob))
    for g in (1,2,3):
        s=vg==g; gp=cal[s]
        u=-(gp@err.T)+2.0*(gp@(centers[None,:]*units).T)/(4.0*mg[g])
        ch[s]=ACTIONS_CF[np.argmax(u,axis=1)]
    return ch*pd.Series(vg).map(CAPACITIES_KWH).to_numpy(float)

def run_arm(surface,feats,tr,va):
    el,cl,ce=contract(tr)
    m=XGBClassifier(num_class=len(ce),**MODEL_PARAMS)
    m.fit(tr[feats].astype('float32').loc[el],cl,sample_weight=np.clip(
        (tr['actual_kwh'].to_numpy(float)/tr['group_id'].map(CAPACITIES_KWH).to_numpy(float))[el],0.10,None))
    p=m.predict_proba(va[feats].astype('float32'),iteration_range=(0,100))
    a=action(p,ce,va['group_id'].to_numpy(int),tr)
    del m,p; gc.collect(); return a

def main():
    fold='dev-2023-Q2'
    surface,names,seq,geom=build_surface(fold)
    vk=val_keys(fold); start=vk['forecast_kst_dtm'].min()
    lab=labels_2022_2023()
    past=lab.loc[lab['forecast_kst_dtm']<start]
    if past['forecast_kst_dtm'].max()>=start: raise RuntimeError('future training label')
    tr=surface.loc[surface['forecast_kst_dtm']<start].merge(
        past[['forecast_kst_dtm','group_id','actual_kwh']],on=['forecast_kst_dtm','group_id'],
        how='inner',validate='one_to_one')
    va=vk.merge(surface,on=['forecast_id','forecast_kst_dtm','group_id'],how='left',validate='one_to_one')
    va=va.merge(lab[['forecast_kst_dtm','group_id','actual_kwh']],on=['forecast_kst_dtm','group_id'],how='left',validate='one_to_one')
    if va['actual_kwh'].isna().any(): raise RuntimeError('validation label gap')
    avail=set(surface.columns)
    base=[n for n in names if n in avail]
    dropped=[n for n in names if n not in avail]
    print('base features',len(base),'dropped(sitewind, not reconstructed in screen)',len(dropped))
    arms={'BASE':base,'PLUS_SEQ':base+seq,'PLUS_GEOM':base+geom,'PLUS_BOTH':base+seq+geom}
    out={}
    for tag,fl in arms.items():
        a=run_arm(surface,fl,tr,va)
        frame=pd.DataFrame({'forecast_id':va['forecast_id'],'forecast_kst_dtm':va['forecast_kst_dtm'],
            'group_id':va['group_id'],'actual_kwh':va['actual_kwh'],'prediction_kwh':a})
        sc=evaluate_official(frame,CAPACITIES_KWH)
        out[tag]={'total':float(sc.total),'one_minus_nmae':float(sc.one_minus_nmae),'ficr':float(sc.ficr),
                  'n_features':len(fl)}
        print(tag,json.dumps(out[tag]))
    base_total=out['BASE']['total']
    for k,v in out.items(): v['delta_vs_base']=v['total']-base_total
    Path('reports/s17_n36_t1_screen.json').write_text(json.dumps(
        {'fold':fold,'training_rows':int(len(tr)),'validation_rows':int(len(va)),
         'seq_columns':len(seq),'geom_columns':len(geom),'arms':out,
         'base_feature_count':len(base),'dropped_sitewind':dropped,
         'new_features_sha256':sha(NEW/'train_seq_geom.parquet')},indent=1)+'\n')
    print(json.dumps(out,indent=1))

main()
