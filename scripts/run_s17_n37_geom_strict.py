"""S17-N37 comparison index 5: geom__ block inside the M115 arm, strict prequential."""
from __future__ import annotations
import gc, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, zipfile
from xgboost import XGBClassifier
from baram.constants import CAPACITIES_KWH
from baram.evaluation.official import evaluate_official

REPO=Path('.')
CACHE=REPO/'artifacts/cache/920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b'
PROBE=REPO/'artifacts/backtests/metric-aligned-probe'
NEW=REPO/'artifacts/features/s17_n36'
OUT=REPO/'artifacts/backtests/s17_n37_geom'; OUT.mkdir(parents=True,exist_ok=True)
FOLDS=('dev-2023-Q2','dev-2023-Q3','dev-2023-Q4'); OUTER=('dev-2023-Q3','dev-2023-Q4')
KEYS=('fold_id','group_id','forecast_kst_dtm')
PREFIX_ROWS=52_560; GRID_ROWS=17_520; FULL=78_912
WEIGHT=0.70/3.0
ACTIONS_CF=np.arange(0.075,1.076,0.0025)
MP={'objective':'multi:softprob','n_estimators':100,'learning_rate':0.03,'max_depth':5,
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
    if b.num_rows!=rows: raise RuntimeError(f'prefix {path}')
    return b.to_pandas()

def npy_prefix(npz,member,rows):
    with zipfile.ZipFile(npz) as a, a.open(a.getinfo(member)) as st:
        v=np.lib.format.read_magic(st)
        shape,f,dt=(np.lib.format.read_array_header_1_0(st) if v==(1,0)
                    else np.lib.format.read_array_header_2_0(st))
        if shape!=(FULL,) or f or dt!=np.dtype('float32'): raise RuntimeError('npz header')
        need=rows*dt.itemsize; buf=st.read(need)
        if len(buf)!=need: raise RuntimeError('npz truncated')
    return np.frombuffer(buf,dtype=dt).copy()

def add_sitewind(m,legacy,allw):
    m['sitewind__legacy']=legacy; m['sitewind__allweather']=allw
    m['sitewind__mean']=(legacy+allw)/2.0; m['sitewind__delta']=allw-legacy
    m['sitewind__disagreement']=np.abs(allw-legacy)
    for s in ('legacy','allweather','mean'):
        v=m[f'sitewind__{s}']; m[f'sitewind__{s}2']=v**2; m[f'sitewind__{s}3']=v**3
        m[f'sitewind__{s}_powercurve']=np.clip((v-3.0)/9.0,0.0,1.0)**3

def selected():
    r={}
    for f in FOLDS:
        j=json.loads((PROBE/f'M115_XGBOOST-{f}.json').read_text())
        n=list(j['selected_feature_names'])
        if len(n)!=100 or int(j['selected_iteration'])!=100: raise RuntimeError('contract')
        r[f]=n
    return r

def surface(names):
    w=set().union(*map(set,names.values())); w={x for x in w if not x.startswith('sitewind__')}
    fs=set(pq.ParquetFile(CACHE/'train_features.parquet').schema.names)
    gs=set(pq.ParquetFile(CACHE/'train_grid_pivot.parquet').schema.names)
    es=set(pq.ParquetFile(CACHE/'train_geometric.parquet').schema.names)
    miss=sorted(w-fs-gs-es-{'group_1','group_2','group_3'})
    if miss: raise RuntimeError(f'missing {miss}')
    base=['forecast_id','forecast_kst_dtm','data_available_kst_dtm','issuance_batch','group_id']
    F=prefix(CACHE/'train_features.parquet',list(dict.fromkeys([*base,*sorted(w&fs)])),PREFIX_ROWS)
    G=prefix(CACHE/'train_grid_pivot.parquet',list(dict.fromkeys(['forecast_kst_dtm',*sorted(w&gs)])),GRID_ROWS)
    E=prefix(CACHE/'train_geometric.parquet',list(dict.fromkeys(['forecast_kst_dtm','data_available_kst_dtm','group_id',*sorted(w&es)])),PREFIX_ROWS)
    np_cols=[c for c in pq.ParquetFile(NEW/'train_seq_geom.parquet').schema.names if c=='forecast_kst_dtm' or c.startswith('geom__')]
    N=prefix(NEW/'train_seq_geom.parquet',np_cols,GRID_ROWS)
    for d in (F,G,E,N): d['forecast_kst_dtm']=pd.to_datetime(d['forecast_kst_dtm'])
    for d in (F,E): d['data_available_kst_dtm']=pd.to_datetime(d['data_available_kst_dtm'])
    if N['forecast_kst_dtm'].max()!=pd.Timestamp('2024-01-01 00:00:00'): raise RuntimeError('2024 guard')
    order=pd.MultiIndex.from_frame(F[['forecast_kst_dtm','group_id']])
    s=F.merge(G,on='forecast_kst_dtm',validate='many_to_one') \
       .merge(E,on=['forecast_kst_dtm','data_available_kst_dtm','group_id'],validate='one_to_one') \
       .merge(N,on='forecast_kst_dtm',validate='many_to_one')
    if not pd.MultiIndex.from_frame(s[['forecast_kst_dtm','group_id']]).equals(order): raise RuntimeError('order')
    for g in (1,2,3): s[f'group_{g}']=s['group_id'].eq(g).astype('int8')
    if len(s)!=PREFIX_ROWS: raise RuntimeError('rows')
    geom=[c for c in N.columns if c.startswith('geom__')]
    return s,geom

def labels_prefix():
    l=prefix(CACHE/'labels_long.parquet',['forecast_kst_dtm','group_id','actual_kwh','operating_year'],PREFIX_ROWS)
    l['forecast_kst_dtm']=pd.to_datetime(l['forecast_kst_dtm'])
    if l['operating_year'].max()>2023: raise RuntimeError('2024 label')
    return l

def contract(tr):
    cap=tr['group_id'].map(CAPACITIES_KWH).to_numpy(float); rate=tr['actual_kwh'].to_numpy(float)/cap
    el=np.isfinite(rate)&(rate>=0.10); cl=np.clip(np.nan_to_num(rate,nan=0.10),0.10,1.074999)
    rb=np.floor((cl-0.10)/0.02).astype(np.int16); ab=np.asarray(sorted(np.unique(rb[el])),dtype=np.int16)
    mp={int(b):i for i,b in enumerate(ab)}; cls=np.asarray([mp[int(v)] for v in rb[el]],dtype=np.int32)
    ce=np.asarray([float(np.mean(rate[el][cls==i])) for i in range(len(ab))],dtype=float)
    return el,cls,ce

def action(prob,ce,vg,tr):
    m=ce>=0.10; p=prob[:,m].astype(float); p/=p.sum(axis=1,keepdims=True); c=ce[m]
    cal=p**(1.0/0.75); cal/=cal.sum(axis=1,keepdims=True)
    err=np.abs(ACTIONS_CF[:,None]-c[None,:]); un=np.select([err<=0.06,err<=0.08],[4.0,3.0],default=0.0)
    cap=tr['group_id'].map(CAPACITIES_KWH).to_numpy(float); rate=tr['actual_kwh'].to_numpy(float)/cap
    mg={g:float(np.mean(rate[(tr['group_id'].to_numpy()==g)&(rate>=0.10)])) for g in (1,2,3)}
    ch=np.empty(len(p))
    for g in (1,2,3):
        s=vg==g; gp=cal[s]
        u=-(gp@err.T)+2.0*(gp@(c[None,:]*un).T)/(4.0*mg[g])
        ch[s]=ACTIONS_CF[np.argmax(u,axis=1)]
    return ch*pd.Series(vg).map(CAPACITIES_KWH).to_numpy(float)

def fit(feats,tr,va):
    el,cls,ce=contract(tr)
    mdl=XGBClassifier(num_class=len(ce),**MP)
    w=np.clip((tr['actual_kwh'].to_numpy(float)/tr['group_id'].map(CAPACITIES_KWH).to_numpy(float))[el],0.10,None)
    mdl.fit(tr[feats].astype('float32').loc[el],cls,sample_weight=w)
    p=mdl.predict_proba(va[feats].astype('float32'),iteration_range=(0,100))
    a=action(p,ce,va['group_id'].to_numpy(int),tr)
    del mdl,p; gc.collect(); return a

def n7():
    f=pd.read_parquet(REPO/'artifacts/backtests/s17_n7_strict_actions/actions.parquet',
        columns=['fold_id','group_id','forecast_kst_dtm','M115_XGBOOST','CHAMPION'])
    f['forecast_kst_dtm']=pd.to_datetime(f['forecast_kst_dtm'])
    if len(f)!=19_440: raise RuntimeError('n7 rows')
    return f

def main():
    names=selected(); S,geom=surface(names); lab=labels_prefix(); A=n7()
    rows=[]; det={}; raw={}
    for fold in FOLDS:
        npz=PROBE/f'M64B_ALLWEATHER_SITEWIND_CLASS-{fold}-sitewind-features.npz'
        M=S.copy(); add_sitewind(M,npy_prefix(npz,'legacy.npy',PREFIX_ROWS),npy_prefix(npz,'allweather.npy',PREFIX_ROWS))
        vk=pd.read_parquet(PROBE/f'M115_XGBOOST-{fold}-policies.parquet',
            columns=['forecast_id','forecast_kst_dtm','group_id'])
        vk['forecast_kst_dtm']=pd.to_datetime(vk['forecast_kst_dtm'])
        start=vk['forecast_kst_dtm'].min()
        past=lab.loc[lab['forecast_kst_dtm']<start]
        tr=M.loc[M['forecast_kst_dtm']<start].merge(past[['forecast_kst_dtm','group_id','actual_kwh']],
            on=['forecast_kst_dtm','group_id'],how='inner',validate='one_to_one')
        va=vk.merge(M,on=['forecast_id','forecast_kst_dtm','group_id'],how='left',validate='one_to_one')
        ctrl=fit(names[fold],tr,va)
        treat=fit([*names[fold],*geom],tr,va)
        pr=va[['forecast_id','forecast_kst_dtm','group_id','data_available_kst_dtm']].copy()
        pr['CTRL']=ctrl; pr['TREAT']=treat
        r=A.loc[A['fold_id'].eq(fold)].merge(pr,on=['forecast_kst_dtm','group_id'],how='left',validate='one_to_one')
        if r[['CTRL','TREAT']].isna().any().any(): raise RuntimeError('align')
        od=(r['forecast_kst_dtm']-pd.Timedelta(hours=1)).dt.normalize()
        basis=od-pd.Timedelta(days=1)+pd.Timedelta(hours=14)
        det[fold]={'feature_availability_safe':bool(r['data_available_kst_dtm'].le(basis).all()),
                   'label_max':past['forecast_kst_dtm'].max().isoformat(),'first_basis':basis.min().isoformat(),
                   'training_rows':int(len(tr)),'validation_rows':int(len(va)),
                   'ctrl_vs_n7_m115_max_abs':float(np.abs(r['CTRL']-r['M115_XGBOOST']).max())}
        cap=r['group_id'].map(CAPACITIES_KWH).to_numpy(float)
        zero=np.clip(r['CHAMPION'].to_numpy(float)+WEIGHT*(r['CTRL'].to_numpy(float)-r['M115_XGBOOST'].to_numpy(float)),0.0,1.075*cap)
        trt =np.clip(r['CHAMPION'].to_numpy(float)+WEIGHT*(r['TREAT'].to_numpy(float)-r['CTRL'].to_numpy(float)),0.0,1.075*cap)
        r['M115_REFIT_ZERO']=zero; r['GEOM_M115_REPLACED']=trt
        if fold=='dev-2023-Q2':
            r['M115_REFIT_ZERO']=r['CHAMPION']; r['GEOM_M115_REPLACED']=r['CHAMPION']
        rows.append(r[[*KEYS,'CHAMPION','M115_REFIT_ZERO','GEOM_M115_REPLACED','M115_XGBOOST','CTRL','TREAT']])
        raw[fold]={'ctrl':ctrl,'treat':treat,'keys':va[['forecast_id','forecast_kst_dtm','group_id']]}
        del M,tr,va; gc.collect()
        print('fold done',fold,flush=True)
    P=pd.concat(rows,ignore_index=True).sort_values(list(KEYS),kind='stable').reset_index(drop=True)
    P.to_parquet(OUT/'predictions.parquet',index=False)
    truth=lab[['forecast_kst_dtm','group_id','actual_kwh']]
    ev={}
    for arm in ('CHAMPION','M115_REFIT_ZERO','GEOM_M115_REPLACED'):
        per={}
        for fold in OUTER:
            part=P.loc[P['fold_id'].eq(fold)].merge(truth,on=['forecast_kst_dtm','group_id'],how='left',validate='one_to_one')
            fr=pd.DataFrame({'forecast_id':part['forecast_kst_dtm'].astype(str)+'_'+part['group_id'].astype(str),
                'forecast_kst_dtm':part['forecast_kst_dtm'],'group_id':part['group_id'],
                'actual_kwh':part['actual_kwh'],'prediction_kwh':part[arm]})
            sc=evaluate_official(fr,CAPACITIES_KWH)
            per[fold]={'total':float(sc.total),'one_minus_nmae':float(sc.one_minus_nmae),'ficr':float(sc.ficr)}
        both=P.loc[P['fold_id'].isin(OUTER)].merge(truth,on=['forecast_kst_dtm','group_id'],how='left',validate='one_to_one')
        fr=pd.DataFrame({'forecast_id':both['forecast_kst_dtm'].astype(str)+'_'+both['group_id'].astype(str),
            'forecast_kst_dtm':both['forecast_kst_dtm'],'group_id':both['group_id'],
            'actual_kwh':both['actual_kwh'],'prediction_kwh':both[arm]})
        sc=evaluate_official(fr,CAPACITIES_KWH)
        ev[arm]={'pooled_outer':{'total':float(sc.total),'one_minus_nmae':float(sc.one_minus_nmae),'ficr':float(sc.ficr)},'per_fold':per}
    rawev={}
    for fold in FOLDS:
        k=raw[fold]['keys'].copy()
        k=k.merge(truth,on=['forecast_kst_dtm','group_id'],how='left',validate='one_to_one')
        for tag,vals in (('CTRL',raw[fold]['ctrl']),('TREAT',raw[fold]['treat'])):
            fr=pd.DataFrame({'forecast_id':k['forecast_id'],'forecast_kst_dtm':k['forecast_kst_dtm'],
                'group_id':k['group_id'],'actual_kwh':k['actual_kwh'],'prediction_kwh':vals})
            sc=evaluate_official(fr,CAPACITIES_KWH)
            rawev.setdefault(fold,{})[tag]={'total':float(sc.total),'one_minus_nmae':float(sc.one_minus_nmae),'ficr':float(sc.ficr)}
        rawev[fold]['delta']=rawev[fold]['TREAT']['total']-rawev[fold]['CTRL']['total']
    out={'family':['CHAMPION','M115_REFIT_ZERO','GEOM_M115_REPLACED'],'outer_folds':list(OUTER),
         'geom_columns':len(geom),'deployment_weight':WEIGHT,'blended':ev,'raw_m115_arm':rawev,
         'fold_details':det,'predictions_sha256':sha(OUT/'predictions.parquet'),
         'new_features_sha256':sha(NEW/'train_seq_geom.parquet')}
    Path('reports/s17_n37_geom_comparison.json').write_text(json.dumps(out,indent=1)+'\n')
    print(json.dumps({'blended':{k:v['pooled_outer'] for k,v in ev.items()},
        'raw_delta':{f:rawev[f]['delta'] for f in FOLDS}},indent=1))

main()
