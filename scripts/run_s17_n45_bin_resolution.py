"""S17-N45: density bin resolution inside the working family."""
from __future__ import annotations
import gc, json
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, zipfile
from xgboost import XGBClassifier
from baram.constants import CAPACITIES_KWH
from baram.evaluation.official import evaluate_official

REPO=Path('.'); CACHE=REPO/'artifacts/cache/920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b'
PROBE=REPO/'artifacts/backtests/metric-aligned-probe'
FOLDS=('dev-2023-Q2','dev-2023-Q3','dev-2023-Q4'); OUTER=('dev-2023-Q3','dev-2023-Q4')
PREFIX=52_560; GRID=17_520; W=0.70/3.0
ACT=np.arange(0.075,1.076,0.0025)
WIDTHS=[0.02,0.01,0.005]; TS=[0.5,0.6,0.75,0.9]; G=2.0
MP={'objective':'multi:softprob','n_estimators':100,'learning_rate':0.03,'max_depth':5,
 'min_child_weight':20.0,'subsample':0.9,'colsample_bytree':0.8,'reg_alpha':0.1,'reg_lambda':5.0,
 'max_bin':256,'tree_method':'hist','random_state':20260802,'n_jobs':6}

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
    w=set().union(*map(set,names.values())); w={x for x in w if not x.startswith('sitewind__')}
    fs=set(pq.ParquetFile(CACHE/'train_features.parquet').schema.names)
    gs=set(pq.ParquetFile(CACHE/'train_grid_pivot.parquet').schema.names)
    es=set(pq.ParquetFile(CACHE/'train_geometric.parquet').schema.names)
    base=['forecast_id','forecast_kst_dtm','data_available_kst_dtm','group_id']
    F=pfx(CACHE/'train_features.parquet',list(dict.fromkeys([*base,*sorted(w&fs)])),PREFIX)
    Gp=pfx(CACHE/'train_grid_pivot.parquet',list(dict.fromkeys(['forecast_kst_dtm',*sorted(w&gs)])),GRID)
    E=pfx(CACHE/'train_geometric.parquet',list(dict.fromkeys(['forecast_kst_dtm','data_available_kst_dtm','group_id',*sorted(w&es)])),PREFIX)
    for d in (F,Gp,E): d['forecast_kst_dtm']=pd.to_datetime(d['forecast_kst_dtm'])
    for d in (F,E): d['data_available_kst_dtm']=pd.to_datetime(d['data_available_kst_dtm'])
    S=F.merge(Gp,on='forecast_kst_dtm',validate='many_to_one').merge(E,on=['forecast_kst_dtm','data_available_kst_dtm','group_id'],validate='one_to_one')
    for g in (1,2,3): S[f'group_{g}']=S['group_id'].eq(g).astype('int8')
    L=pfx(CACHE/'labels_long.parquet',['forecast_kst_dtm','group_id','actual_kwh','operating_year'],PREFIX)
    L['forecast_kst_dtm']=pd.to_datetime(L['forecast_kst_dtm'])
    if L['operating_year'].max()>2023: raise RuntimeError('2024')
    A=pd.read_parquet(REPO/'artifacts/backtests/s17_n7_strict_actions/actions.parquet',
        columns=['fold_id','group_id','forecast_kst_dtm','actual_kwh','M115_XGBOOST','CHAMPION'])
    A['forecast_kst_dtm']=pd.to_datetime(A['forecast_kst_dtm'])
    res={}; store={}
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
        Xt=tr[names[fold]].astype('float32').loc[el]; Xv=va[names[fold]].astype('float32')
        gid=tr['group_id'].to_numpy(int)[el]
        mg={g:float(np.mean(ycf[el][gid==g])) for g in (1,2,3)}
        vt=va[['forecast_kst_dtm','group_id']].merge(L[['forecast_kst_dtm','group_id','actual_kwh']],
            on=['forecast_kst_dtm','group_id'],how='left',validate='one_to_one')
        store[fold]={'va':va,'vt':vt,'acts':{}}
        for bw in WIDTHS:
            cl=np.clip(ycf,0.10,1.074999)
            rb=np.floor((cl-0.10)/bw).astype(np.int32)
            ab=np.asarray(sorted(np.unique(rb[el]))); mp={int(b):i for i,b in enumerate(ab)}
            cls=np.asarray([mp[int(v)] for v in rb[el]],dtype=np.int32)
            ce=np.asarray([float(np.mean(ycf[el][cls==i])) for i in range(len(ab))])
            m=XGBClassifier(num_class=len(ce),**MP)
            m.fit(Xt,cls,sample_weight=np.clip(ycf[el],0.10,None))
            p=m.predict_proba(Xv,iteration_range=(0,100)); del m; gc.collect()
            keep=ce>=0.10; p=p[:,keep].astype(float); p/=p.sum(axis=1,keepdims=True); c=ce[keep]
            err=np.abs(ACT[:,None]-c[None,:]); un=np.select([err<=0.06,err<=0.08],[4.0,3.0],default=0.0)
            vg=va['group_id'].to_numpy(int)
            for T in TS:
                cal=p**(1.0/T); cal/=cal.sum(axis=1,keepdims=True)
                ch=np.empty(len(p))
                for g in (1,2,3):
                    s=vg==g; gp=cal[s]
                    u=-(gp@err.T)+G*(gp@(c[None,:]*un).T)/(4.0*mg[g])
                    ch[s]=ACT[np.argmax(u,axis=1)]
                act=ch*pd.Series(vg).map(CAPACITIES_KWH).to_numpy(float)
                store[fold]['acts'][f'bw{bw}_T{T}']=act
                fr=pd.DataFrame({'forecast_id':va['forecast_id'],'forecast_kst_dtm':va['forecast_kst_dtm'],
                    'group_id':va['group_id'],'actual_kwh':vt['actual_kwh'],'prediction_kwh':act})
                sc=evaluate_official(fr,CAPACITIES_KWH)
                res.setdefault(fold,{})[f'bw{bw}_T{T}']={'total':float(sc.total),'classes':int(len(c))}
            print(fold,bw,'done',flush=True)
        del M,tr; gc.collect()
    q2={k:v['total'] for k,v in res['dev-2023-Q2'].items()}
    best=max(q2,key=q2.get)
    pooled={}
    for tag in q2:
        parts=[]
        for fold in OUTER:
            r=A.loc[A['fold_id'].eq(fold)].merge(pd.DataFrame({'forecast_kst_dtm':store[fold]['va']['forecast_kst_dtm'],
                'group_id':store[fold]['va']['group_id'],'X':store[fold]['acts'][tag]}),
                on=['forecast_kst_dtm','group_id'],how='left',validate='one_to_one')
            cp=r['group_id'].map(CAPACITIES_KWH).to_numpy(float)
            r['BL']=np.clip(r['CHAMPION'].to_numpy(float)+W*(r['X'].to_numpy(float)-r['M115_XGBOOST'].to_numpy(float)),0.0,1.075*cp)
            parts.append(r)
        P=pd.concat(parts,ignore_index=True)
        fr=pd.DataFrame({'forecast_id':P['forecast_kst_dtm'].astype(str)+'_'+P['group_id'].astype(str),
            'forecast_kst_dtm':P['forecast_kst_dtm'],'group_id':P['group_id'],
            'actual_kwh':P['actual_kwh'],'prediction_kwh':P['BL']})
        s=evaluate_official(fr,CAPACITIES_KWH); pooled[tag]=float(s.total)
    out={'raw_by_fold':res,'q2_selected':best,'q2_selected_total':q2[best],
         'blended_pooled_outer':pooled,'deployed_tag':'bw0.02_T0.75','champion':0.6314827308346854}
    Path('reports/s17_n45_bin_resolution.json').write_text(json.dumps(out,indent=1)+'\n')
    print('Q2 best',best,q2[best])
    print('blended: deployed-equiv',pooled['bw0.02_T0.75'],'| q2-selected',pooled[best],'| champion 0.6314827308')

main()
