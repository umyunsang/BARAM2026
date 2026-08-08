"""S17-N51: is the deployed member under-fitted? capacity sweep on trees and feature count."""
from __future__ import annotations
import gc, json
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, zipfile
from xgboost import XGBClassifier
from baram.constants import CAPACITIES_KWH
from baram.evaluation.official import evaluate_official
REPO=Path('.'); CACHE=REPO/'artifacts/cache/920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b'
PROBE=REPO/'artifacts/backtests/metric-aligned-probe'; NEW=REPO/'artifacts/features/s17_n36'
FOLDS=('dev-2023-Q2','dev-2023-Q3','dev-2023-Q4'); OUTER=('dev-2023-Q3','dev-2023-Q4')
PREFIX=52_560; GRID=17_520; ACT=np.arange(0.075,1.076,0.0025)
BASE={'objective':'multi:softprob','learning_rate':0.03,'max_depth':5,'min_child_weight':20.0,
 'subsample':0.9,'colsample_bytree':0.8,'reg_alpha':0.1,'reg_lambda':5.0,'max_bin':256,
 'tree_method':'hist','random_state':20260802,'n_jobs':6}
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
def main():
    names={f:list(json.loads((PROBE/f'M115_XGBOOST-{f}.json').read_text())['selected_feature_names']) for f in FOLDS}
    allf=list(pq.ParquetFile(CACHE/'train_features.parquet').schema.names)
    keep=[c for c in allf if c not in ('forecast_id','forecast_kst_dtm','data_available_kst_dtm','issuance_batch','operating_year','actual_kwh')]
    base=['forecast_id','forecast_kst_dtm','data_available_kst_dtm','group_id']
    F=pfx(CACHE/'train_features.parquet',list(dict.fromkeys([*base,*keep])),PREFIX)
    gcols=[c for c in pq.ParquetFile(NEW/'train_seq_geom.parquet').schema.names if c=='forecast_kst_dtm' or c.startswith('geom__')]
    N=pfx(NEW/'train_seq_geom.parquet',gcols,GRID)
    N=N.rename(columns={c:('n36__'+c) for c in N.columns if c.startswith('geom__')})
    ecols=list(pq.ParquetFile(CACHE/'train_geometric.parquet').schema.names)
    E=pfx(CACHE/'train_geometric.parquet',ecols,PREFIX)
    gpv=list(pq.ParquetFile(CACHE/'train_grid_pivot.parquet').schema.names)
    GP=pfx(CACHE/'train_grid_pivot.parquet',gpv,GRID)
    for d in (F,N,E,GP): d['forecast_kst_dtm']=pd.to_datetime(d['forecast_kst_dtm'])
    for d in (F,E): d['data_available_kst_dtm']=pd.to_datetime(d['data_available_kst_dtm'])
    S=F.merge(GP,on='forecast_kst_dtm',validate='many_to_one') \
       .merge(E,on=['forecast_kst_dtm','data_available_kst_dtm','group_id'],validate='one_to_one') \
       .merge(N,on='forecast_kst_dtm',validate='many_to_one')
    for g in (1,2,3): S[f'group_{g}']=S['group_id'].eq(g).astype('int8')
    geom=[c for c in N.columns if c.startswith('n36__geom__')]
    wide=[c for c in S.columns if c not in ('forecast_id','forecast_kst_dtm','data_available_kst_dtm','issuance_batch') and pd.api.types.is_numeric_dtype(S[c])]
    L=pfx(CACHE/'labels_long.parquet',['forecast_kst_dtm','group_id','actual_kwh','operating_year'],PREFIX)
    L['forecast_kst_dtm']=pd.to_datetime(L['forecast_kst_dtm'])
    if L['operating_year'].max()>2023: raise RuntimeError('2024')
    res={}
    for fold in FOLDS:
        npz=PROBE/f'M64B_ALLWEATHER_SITEWIND_CLASS-{fold}-sitewind-features.npz'
        M=S.copy(); sw(M,npyp(npz,'legacy.npy',PREFIX),npyp(npz,'allweather.npy',PREFIX))
        vk=pd.read_parquet(PROBE/f'M115_XGBOOST-{fold}-policies.parquet',columns=['forecast_id','forecast_kst_dtm','group_id'])
        vk['forecast_kst_dtm']=pd.to_datetime(vk['forecast_kst_dtm']); start=vk['forecast_kst_dtm'].min()
        past=L.loc[L['forecast_kst_dtm']<start]
        tr=M.loc[M['forecast_kst_dtm']<start].merge(past[['forecast_kst_dtm','group_id','actual_kwh']],
            on=['forecast_kst_dtm','group_id'],how='inner',validate='one_to_one')
        va=vk.merge(M,on=['forecast_id','forecast_kst_dtm','group_id'],how='left',validate='one_to_one')
        vt=va[['forecast_kst_dtm','group_id']].merge(L[['forecast_kst_dtm','group_id','actual_kwh']],
            on=['forecast_kst_dtm','group_id'],how='left',validate='one_to_one')
        cap=tr['group_id'].map(CAPACITIES_KWH).to_numpy(float); ycf=tr['actual_kwh'].to_numpy(float)/cap
        el=np.isfinite(ycf)&(ycf>=0.10)
        cl=np.clip(np.nan_to_num(ycf,nan=0.10),0.10,1.074999)
        rb=np.floor((cl-0.10)/0.02).astype(np.int32); ab=np.asarray(sorted(np.unique(rb[el])))
        mp={int(b):i for i,b in enumerate(ab)}; cls=np.asarray([mp[int(v)] for v in rb[el]],dtype=np.int32)
        ce=np.asarray([float(np.mean(ycf[el][cls==i])) for i in range(len(ab))])
        gid=tr['group_id'].to_numpy(int)[el]; mg={g:float(np.mean(ycf[el][gid==g])) for g in (1,2,3)}
        keepc=ce>=0.10; ck=ce[keepc]
        err=np.abs(ACT[:,None]-ck[None,:]); un=np.select([err<=0.06,err<=0.08],[4.0,3.0],default=0.0)
        vg=va['group_id'].to_numpy(int); capv=va['group_id'].map(CAPACITIES_KWH).to_numpy(float)
        for tag,cols,ne in [('sel100_n100',names[fold],100),('sel100_n600',names[fold],600),
                            ('sel100geom_n600',[*names[fold],*geom],600),
                            ('wide_n600',[c for c in wide if c in M.columns],600),
                            ('wide_n1500',[c for c in wide if c in M.columns],1500)]:
            m=XGBClassifier(num_class=len(ce),n_estimators=ne,**BASE)
            m.fit(tr[cols].astype('float32').loc[el],cls,sample_weight=np.clip(ycf[el],0.10,None))
            p=m.predict_proba(va[cols].astype('float32')); del m; gc.collect()
            pk=p[:,keepc]/p[:,keepc].sum(axis=1,keepdims=True)
            cal=pk**(1/0.75); cal/=cal.sum(axis=1,keepdims=True)
            ch=np.empty(len(pk))
            for g in (1,2,3):
                s=vg==g; gp=cal[s]
                u=-(gp@err.T)+2.0*(gp@(ck[None,:]*un).T)/(4.0*mg[g]); ch[s]=ACT[np.argmax(u,axis=1)]
            fr=pd.DataFrame({'forecast_id':va['forecast_id'],'forecast_kst_dtm':va['forecast_kst_dtm'],
                'group_id':va['group_id'],'actual_kwh':vt['actual_kwh'],'prediction_kwh':ch*capv})
            sc=evaluate_official(fr,CAPACITIES_KWH)
            res.setdefault(fold,{})[tag]={'total':float(sc.total),'one_minus_nmae':float(sc.one_minus_nmae),
                'ficr':float(sc.ficr),'n_features':len(cols),'n_estimators':ne}
            print(fold,tag,round(sc.total,6),round(sc.one_minus_nmae,6),flush=True)
        del M,tr,va; gc.collect()
    Path('reports/s17_n51_capacity_sweep.json').write_text(json.dumps(res,indent=1)+'\n')
main()
