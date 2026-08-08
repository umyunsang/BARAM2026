"""S17-N44: direct band-mass objective (smoothed step reward) trained with LightGBM."""
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
PREFIX=52_560; GRID=17_520; W=0.70/3.0
S_SMOOTH=0.01   # predeclared, not tuned

def pfx(p,c,r): return next(pq.ParquetFile(p).iter_batches(batch_size=r,columns=c,use_threads=False)).to_pandas()
def npyp(z,m,r):
    with zipfile.ZipFile(z) as a, a.open(a.getinfo(m)) as st:
        v=np.lib.format.read_magic(st)
        s,f,dt=(np.lib.format.read_array_header_1_0(st) if v==(1,0) else np.lib.format.read_array_header_2_0(st))
        return np.frombuffer(st.read(r*dt.itemsize),dtype=dt).copy()
def sw(m,l,a):
    m['sitewind__legacy']=l; m['sitewind__allweather']=a; m['sitewind__mean']=(l+a)/2.0
    m['sitewind__delta']=a-l; m['sitewind__disagreement']=np.abs(a-l)
    for s in ('legacy','allweather','mean'):
        v=m[f'sitewind__{s}']; m[f'sitewind__{s}2']=v**2; m[f'sitewind__{s}3']=v**3
        m[f'sitewind__{s}_powercurve']=np.clip((v-3.0)/9.0,0.0,1.0)**3
def sig(x): return 1.0/(1.0+np.exp(-np.clip(x,-40,40)))

def main():
    names={f:list(json.loads((PROBE/f'M115_XGBOOST-{f}.json').read_text())['selected_feature_names']) for f in FOLDS}
    w=set().union(*map(set,names.values())); w={x for x in w if not x.startswith('sitewind__')}
    fs=set(pq.ParquetFile(CACHE/'train_features.parquet').schema.names)
    gs=set(pq.ParquetFile(CACHE/'train_grid_pivot.parquet').schema.names)
    es=set(pq.ParquetFile(CACHE/'train_geometric.parquet').schema.names)
    base=['forecast_id','forecast_kst_dtm','data_available_kst_dtm','group_id']
    F=pfx(CACHE/'train_features.parquet',list(dict.fromkeys([*base,*sorted(w&fs)])),PREFIX)
    G=pfx(CACHE/'train_grid_pivot.parquet',list(dict.fromkeys(['forecast_kst_dtm',*sorted(w&gs)])),GRID)
    E=pfx(CACHE/'train_geometric.parquet',list(dict.fromkeys(['forecast_kst_dtm','data_available_kst_dtm','group_id',*sorted(w&es)])),PREFIX)
    for d in (F,G,E): d['forecast_kst_dtm']=pd.to_datetime(d['forecast_kst_dtm'])
    for d in (F,E): d['data_available_kst_dtm']=pd.to_datetime(d['data_available_kst_dtm'])
    S=F.merge(G,on='forecast_kst_dtm',validate='many_to_one').merge(E,on=['forecast_kst_dtm','data_available_kst_dtm','group_id'],validate='one_to_one')
    for g in (1,2,3): S[f'group_{g}']=S['group_id'].eq(g).astype('int8')
    L=pfx(CACHE/'labels_long.parquet',['forecast_kst_dtm','group_id','actual_kwh','operating_year'],PREFIX)
    L['forecast_kst_dtm']=pd.to_datetime(L['forecast_kst_dtm'])
    if L['operating_year'].max()>2023: raise RuntimeError('2024')
    A=pd.read_parquet(REPO/'artifacts/backtests/s17_n7_strict_actions/actions.parquet',
        columns=['fold_id','group_id','forecast_kst_dtm','actual_kwh','M115_XGBOOST','CHAMPION'])
    A['forecast_kst_dtm']=pd.to_datetime(A['forecast_kst_dtm'])
    out={}; blends={}
    for fold in FOLDS:
        npz=PROBE/f'M64B_ALLWEATHER_SITEWIND_CLASS-{fold}-sitewind-features.npz'
        M=S.copy(); sw(M,npyp(npz,'legacy.npy',PREFIX),npyp(npz,'allweather.npy',PREFIX))
        vk=pd.read_parquet(PROBE/f'M115_XGBOOST-{fold}-policies.parquet',columns=['forecast_id','forecast_kst_dtm','group_id'])
        vk['forecast_kst_dtm']=pd.to_datetime(vk['forecast_kst_dtm']); start=vk['forecast_kst_dtm'].min()
        past=L.loc[L['forecast_kst_dtm']<start]
        tr=M.loc[M['forecast_kst_dtm']<start].merge(past[['forecast_kst_dtm','group_id','actual_kwh']],
            on=['forecast_kst_dtm','group_id'],how='inner',validate='one_to_one')
        va=vk.merge(M,on=['forecast_id','forecast_kst_dtm','group_id'],how='left',validate='one_to_one')
        cap=tr['group_id'].map(CAPACITIES_KWH).to_numpy(float); ycf=tr['actual_kwh'].to_numpy(float)/cap
        el=np.isfinite(ycf)&(ycf>=0.10)
        X=tr[names[fold]].astype('float32').loc[el].to_numpy(); y=ycf[el]
        gid=tr['group_id'].to_numpy(int)[el]
        mg={g:float(np.mean(y[gid==g])) for g in (1,2,3)}
        k=np.array([y[i]/(4.0*mg[gid[i]]) for i in range(len(y))])
        def obj(pred,ds):
            a=pred; d=a-y; e=np.abs(d); s=np.sign(d); s[s==0]=1.0
            z6=(0.06-e)/S_SMOOTH; z8=(0.08-e)/S_SMOOTH
            dsig6=sig(z6)*(1-sig(z6))/S_SMOOTH; dsig8=sig(z8)*(1-sig(z8))/S_SMOOTH
            grad=s*(1.0 + k*(3.0*dsig8 + 1.0*dsig6))
            hess=np.full_like(grad, 1.0) + k*(3.0*dsig8+1.0*dsig6)*2.0
            return grad, np.maximum(hess,1e-3)
        init=float(np.median(y))
        ds=lgb.Dataset(X,label=y,init_score=np.full(len(y),init))
        booster=lgb.train({'objective':obj,'learning_rate':0.03,'num_leaves':31,'min_data_in_leaf':40,
            'feature_fraction':0.8,'bagging_fraction':0.9,'bagging_freq':1,'lambda_l2':5.0,
            'seed':20260809,'num_threads':6,'verbose':-1},ds,num_boost_round=400)
        Xv=va[names[fold]].astype('float32').to_numpy()
        a_cf=np.clip(init+booster.predict(Xv),0.075,1.075)
        act=a_cf*va['group_id'].map(CAPACITIES_KWH).to_numpy(float)
        vt=va[['forecast_kst_dtm','group_id']].merge(L[['forecast_kst_dtm','group_id','actual_kwh']],
            on=['forecast_kst_dtm','group_id'],how='left',validate='one_to_one')
        fr=pd.DataFrame({'forecast_id':va['forecast_id'],'forecast_kst_dtm':va['forecast_kst_dtm'],
            'group_id':va['group_id'],'actual_kwh':vt['actual_kwh'],'prediction_kwh':act})
        sc=evaluate_official(fr,CAPACITIES_KWH)
        out[fold]={'raw_total':float(sc.total),'raw_1_nmae':float(sc.one_minus_nmae),'raw_ficr':float(sc.ficr),
                   'train_rows':int(el.sum())}
        r=A.loc[A['fold_id'].eq(fold)].merge(pd.DataFrame({'forecast_kst_dtm':va['forecast_kst_dtm'],
            'group_id':va['group_id'],'B':act}),on=['forecast_kst_dtm','group_id'],how='left',validate='one_to_one')
        capr=r['group_id'].map(CAPACITIES_KWH).to_numpy(float)
        r['BL']=np.clip(r['CHAMPION'].to_numpy(float)+W*(r['B'].to_numpy(float)-r['M115_XGBOOST'].to_numpy(float)),0.0,1.075*capr)
        blends[fold]=r; print(fold,json.dumps(out[fold]),flush=True)
        del M,tr,va,booster; gc.collect()
    P=pd.concat([blends[f] for f in OUTER],ignore_index=True)
    for tag,col in (('champion','CHAMPION'),('bandmass_blended','BL')):
        fr=pd.DataFrame({'forecast_id':P['forecast_kst_dtm'].astype(str)+'_'+P['group_id'].astype(str),
            'forecast_kst_dtm':P['forecast_kst_dtm'],'group_id':P['group_id'],
            'actual_kwh':P['actual_kwh'],'prediction_kwh':P[col]})
        s=evaluate_official(fr,CAPACITIES_KWH)
        out[f'pooled_outer_{tag}']={'total':float(s.total),'one_minus_nmae':float(s.one_minus_nmae),'ficr':float(s.ficr)}
    Path('reports/s17_n44_band_mass.json').write_text(json.dumps(out,indent=1)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k.startswith('pooled')},indent=1))

main()
