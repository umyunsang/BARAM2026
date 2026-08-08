"""S17-N43: quantile-function conditional density vs the binned-softmax density."""
from __future__ import annotations
import gc, json
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, zipfile
import lightgbm as lgb
from baram.constants import CAPACITIES_KWH
from baram.evaluation.official import evaluate_official

REPO=Path('.')
CACHE=REPO/'artifacts/cache/920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b'
PROBE=REPO/'artifacts/backtests/metric-aligned-probe'
FOLDS=('dev-2023-Q2','dev-2023-Q3','dev-2023-Q4'); OUTER=('dev-2023-Q3','dev-2023-Q4')
PREFIX_ROWS=52_560; GRID_ROWS=17_520; FULL=78_912; W=0.70/3.0
ACT=np.arange(0.075,1.076,0.0025)
QS=np.round(np.arange(0.025,0.9751,0.025),4)

def prefix(p,c,r):
    b=next(pq.ParquetFile(p).iter_batches(batch_size=r,columns=c,use_threads=False))
    return b.to_pandas()

def npyp(npz,m,r):
    with zipfile.ZipFile(npz) as a, a.open(a.getinfo(m)) as st:
        v=np.lib.format.read_magic(st)
        s,f,dt=(np.lib.format.read_array_header_1_0(st) if v==(1,0) else np.lib.format.read_array_header_2_0(st))
        buf=st.read(r*dt.itemsize)
    return np.frombuffer(buf,dtype=dt).copy()

def sitewind(m,l,a):
    m['sitewind__legacy']=l; m['sitewind__allweather']=a; m['sitewind__mean']=(l+a)/2.0
    m['sitewind__delta']=a-l; m['sitewind__disagreement']=np.abs(a-l)
    for s in ('legacy','allweather','mean'):
        v=m[f'sitewind__{s}']; m[f'sitewind__{s}2']=v**2; m[f'sitewind__{s}3']=v**3
        m[f'sitewind__{s}_powercurve']=np.clip((v-3.0)/9.0,0.0,1.0)**3

def main():
    names={f:list(json.loads((PROBE/f'M115_XGBOOST-{f}.json').read_text())['selected_feature_names']) for f in FOLDS}
    w=set().union(*map(set,names.values())); w={x for x in w if not x.startswith('sitewind__')}
    fs=set(pq.ParquetFile(CACHE/'train_features.parquet').schema.names)
    gs=set(pq.ParquetFile(CACHE/'train_grid_pivot.parquet').schema.names)
    es=set(pq.ParquetFile(CACHE/'train_geometric.parquet').schema.names)
    base=['forecast_id','forecast_kst_dtm','data_available_kst_dtm','group_id']
    F=prefix(CACHE/'train_features.parquet',list(dict.fromkeys([*base,*sorted(w&fs)])),PREFIX_ROWS)
    G=prefix(CACHE/'train_grid_pivot.parquet',list(dict.fromkeys(['forecast_kst_dtm',*sorted(w&gs)])),GRID_ROWS)
    E=prefix(CACHE/'train_geometric.parquet',list(dict.fromkeys(['forecast_kst_dtm','data_available_kst_dtm','group_id',*sorted(w&es)])),PREFIX_ROWS)
    for d in (F,G,E): d['forecast_kst_dtm']=pd.to_datetime(d['forecast_kst_dtm'])
    for d in (F,E): d['data_available_kst_dtm']=pd.to_datetime(d['data_available_kst_dtm'])
    S=F.merge(G,on='forecast_kst_dtm',validate='many_to_one').merge(E,on=['forecast_kst_dtm','data_available_kst_dtm','group_id'],validate='one_to_one')
    for g in (1,2,3): S[f'group_{g}']=S['group_id'].eq(g).astype('int8')
    L=prefix(CACHE/'labels_long.parquet',['forecast_kst_dtm','group_id','actual_kwh','operating_year'],PREFIX_ROWS)
    L['forecast_kst_dtm']=pd.to_datetime(L['forecast_kst_dtm'])
    if L['operating_year'].max()>2023: raise RuntimeError('2024')
    A=pd.read_parquet(REPO/'artifacts/backtests/s17_n7_strict_actions/actions.parquet',
        columns=['fold_id','group_id','forecast_kst_dtm','actual_kwh','M115_XGBOOST','CHAMPION'])
    A['forecast_kst_dtm']=pd.to_datetime(A['forecast_kst_dtm'])
    out={}; blends={}
    for fold in FOLDS:
        npz=PROBE/f'M64B_ALLWEATHER_SITEWIND_CLASS-{fold}-sitewind-features.npz'
        M=S.copy(); sitewind(M,npyp(npz,'legacy.npy',PREFIX_ROWS),npyp(npz,'allweather.npy',PREFIX_ROWS))
        vk=pd.read_parquet(PROBE/f'M115_XGBOOST-{fold}-policies.parquet',columns=['forecast_id','forecast_kst_dtm','group_id'])
        vk['forecast_kst_dtm']=pd.to_datetime(vk['forecast_kst_dtm']); start=vk['forecast_kst_dtm'].min()
        past=L.loc[L['forecast_kst_dtm']<start]
        tr=M.loc[M['forecast_kst_dtm']<start].merge(past[['forecast_kst_dtm','group_id','actual_kwh']],
            on=['forecast_kst_dtm','group_id'],how='inner',validate='one_to_one')
        va=vk.merge(M,on=['forecast_id','forecast_kst_dtm','group_id'],how='left',validate='one_to_one')
        cap=tr['group_id'].map(CAPACITIES_KWH).to_numpy(float); ycf=tr['actual_kwh'].to_numpy(float)/cap
        el=np.isfinite(ycf)&(ycf>=0.10)
        X=tr[names[fold]].astype('float32').loc[el]; y=ycf[el]
        Q=np.zeros((len(va),len(QS)),dtype=float)
        Xv=va[names[fold]].astype('float32')
        for j,q in enumerate(QS):
            m=lgb.LGBMRegressor(objective='quantile',alpha=float(q),n_estimators=300,learning_rate=0.05,
                num_leaves=31,min_child_samples=40,subsample=0.9,subsample_freq=1,colsample_bytree=0.8,
                reg_lambda=5.0,random_state=20260809,n_jobs=6,verbose=-1)
            m.fit(X,y); Q[:,j]=m.predict(Xv); del m
        Q=np.sort(Q,axis=1)
        mg={g:float(np.mean(y[tr['group_id'].to_numpy()[el]==g])) for g in (1,2,3)}
        e=np.abs(ACT[None,:,None]-Q[:,None,:])
        un=np.select([e<=0.06,e<=0.08],[4.0,3.0],default=0.0)
        gm=np.array([mg[g] for g in va['group_id'].to_numpy(int)])[:,None]
        util=-e.mean(axis=2)+ (Q[:,None,:]*un).mean(axis=2)/(4.0*gm)
        a_cf=ACT[util.argmax(axis=1)]
        act=a_cf*va['group_id'].map(CAPACITIES_KWH).to_numpy(float)
        # calibration of the new density
        vt=va[['forecast_kst_dtm','group_id']].merge(L[['forecast_kst_dtm','group_id','actual_kwh']],
            on=['forecast_kst_dtm','group_id'],how='left',validate='one_to_one')
        ytrue=vt['actual_kwh'].to_numpy(float)/va['group_id'].map(CAPACITIES_KWH).to_numpy(float)
        pit=(Q<ytrue[:,None]).mean(axis=1)
        fr=pd.DataFrame({'forecast_id':va['forecast_id'],'forecast_kst_dtm':va['forecast_kst_dtm'],
            'group_id':va['group_id'],'actual_kwh':vt['actual_kwh'],'prediction_kwh':act})
        sc=evaluate_official(fr,CAPACITIES_KWH)
        out[fold]={'raw_quantile_total':float(sc.total),'raw_1_nmae':float(sc.one_minus_nmae),'raw_ficr':float(sc.ficr),
            'pit_mean':float(np.nanmean(pit)),'pit_sd':float(np.nanstd(pit)),
            'iqr_median_cf':float(np.median(Q[:,int(0.75/0.025)-1]-Q[:,int(0.25/0.025)-1])),
            'training_rows':int(el.sum())}
        r=A.loc[A['fold_id'].eq(fold)].merge(
            pd.DataFrame({'forecast_kst_dtm':va['forecast_kst_dtm'],'group_id':va['group_id'],'Q':act}),
            on=['forecast_kst_dtm','group_id'],how='left',validate='one_to_one')
        capr=r['group_id'].map(CAPACITIES_KWH).to_numpy(float)
        r['BL']=np.clip(r['CHAMPION'].to_numpy(float)+W*(r['Q'].to_numpy(float)-r['M115_XGBOOST'].to_numpy(float)),0.0,1.075*capr)
        blends[fold]=r
        print(fold,json.dumps(out[fold]),flush=True)
        del M,tr,va; gc.collect()
    P=pd.concat([blends[f] for f in OUTER],ignore_index=True)
    for tag,col in (('champion','CHAMPION'),('quantile_blended','BL')):
        fr=pd.DataFrame({'forecast_id':P['forecast_kst_dtm'].astype(str)+'_'+P['group_id'].astype(str),
            'forecast_kst_dtm':P['forecast_kst_dtm'],'group_id':P['group_id'],
            'actual_kwh':P['actual_kwh'],'prediction_kwh':P[col]})
        s=evaluate_official(fr,CAPACITIES_KWH)
        out[f'pooled_outer_{tag}']={'total':float(s.total),'one_minus_nmae':float(s.one_minus_nmae),'ficr':float(s.ficr)}
    Path('reports/s17_n43_quantile_density.json').write_text(json.dumps(out,indent=1)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k.startswith('pooled')},indent=1))

main()
