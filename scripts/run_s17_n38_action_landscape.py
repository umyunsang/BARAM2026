"""S17-N38 T0 diagnostic: action-rule landscape at a FIXED predictive distribution."""
from __future__ import annotations
import gc, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, zipfile
from xgboost import XGBClassifier
from baram.constants import CAPACITIES_KWH
from baram.evaluation.official import evaluate_official
import importlib.util as iu
spec=iu.spec_from_file_location('n37','scripts/run_s17_n37_geom_strict.py')

REPO=Path('.')
CACHE=REPO/'artifacts/cache/920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b'
PROBE=REPO/'artifacts/backtests/metric-aligned-probe'
NEW=REPO/'artifacts/features/s17_n36'
FOLDS=('dev-2023-Q2','dev-2023-Q3','dev-2023-Q4'); OUTER=('dev-2023-Q3','dev-2023-Q4')
PREFIX_ROWS=52_560; GRID_ROWS=17_520; FULL=78_912; WEIGHT=0.70/3.0
ACTIONS_CF=np.arange(0.075,1.076,0.0025)
MP={'objective':'multi:softprob','n_estimators':100,'learning_rate':0.03,'max_depth':5,
 'min_child_weight':20.0,'subsample':0.9,'colsample_bytree':0.8,'reg_alpha':0.1,'reg_lambda':5.0,
 'max_bin':256,'tree_method':'hist','random_state':20260802,'n_jobs':6}

def prefix(path,cols,rows):
    b=next(pq.ParquetFile(path).iter_batches(batch_size=rows,columns=cols,use_threads=False))
    if b.num_rows!=rows: raise RuntimeError('prefix')
    return b.to_pandas()

def npy_prefix(npz,member,rows):
    with zipfile.ZipFile(npz) as a, a.open(a.getinfo(member)) as st:
        v=np.lib.format.read_magic(st)
        shape,f,dt=(np.lib.format.read_array_header_1_0(st) if v==(1,0) else np.lib.format.read_array_header_2_0(st))
        if shape!=(FULL,) or f or dt!=np.dtype('float32'): raise RuntimeError('hdr')
        buf=st.read(rows*dt.itemsize)
    return np.frombuffer(buf,dtype=dt).copy()

def add_sitewind(m,l,a):
    m['sitewind__legacy']=l; m['sitewind__allweather']=a; m['sitewind__mean']=(l+a)/2.0
    m['sitewind__delta']=a-l; m['sitewind__disagreement']=np.abs(a-l)
    for s in ('legacy','allweather','mean'):
        v=m[f'sitewind__{s}']; m[f'sitewind__{s}2']=v**2; m[f'sitewind__{s}3']=v**3
        m[f'sitewind__{s}_powercurve']=np.clip((v-3.0)/9.0,0.0,1.0)**3

def selected():
    r={}
    for f in FOLDS:
        j=json.loads((PROBE/f'M115_XGBOOST-{f}.json').read_text()); r[f]=list(j['selected_feature_names'])
    return r

def surface(names):
    w=set().union(*map(set,names.values())); w={x for x in w if not x.startswith('sitewind__')}
    fs=set(pq.ParquetFile(CACHE/'train_features.parquet').schema.names)
    gs=set(pq.ParquetFile(CACHE/'train_grid_pivot.parquet').schema.names)
    es=set(pq.ParquetFile(CACHE/'train_geometric.parquet').schema.names)
    base=['forecast_id','forecast_kst_dtm','data_available_kst_dtm','issuance_batch','group_id']
    F=prefix(CACHE/'train_features.parquet',list(dict.fromkeys([*base,*sorted(w&fs)])),PREFIX_ROWS)
    G=prefix(CACHE/'train_grid_pivot.parquet',list(dict.fromkeys(['forecast_kst_dtm',*sorted(w&gs)])),GRID_ROWS)
    E=prefix(CACHE/'train_geometric.parquet',list(dict.fromkeys(['forecast_kst_dtm','data_available_kst_dtm','group_id',*sorted(w&es)])),PREFIX_ROWS)
    for d in (F,G,E): d['forecast_kst_dtm']=pd.to_datetime(d['forecast_kst_dtm'])
    for d in (F,E): d['data_available_kst_dtm']=pd.to_datetime(d['data_available_kst_dtm'])
    s=F.merge(G,on='forecast_kst_dtm',validate='many_to_one').merge(E,on=['forecast_kst_dtm','data_available_kst_dtm','group_id'],validate='one_to_one')
    for g in (1,2,3): s[f'group_{g}']=s['group_id'].eq(g).astype('int8')
    return s

def labels_prefix():
    l=prefix(CACHE/'labels_long.parquet',['forecast_kst_dtm','group_id','actual_kwh','operating_year'],PREFIX_ROWS)
    l['forecast_kst_dtm']=pd.to_datetime(l['forecast_kst_dtm'])
    if l['operating_year'].max()>2023: raise RuntimeError('2024')
    return l

def contract(tr):
    cap=tr['group_id'].map(CAPACITIES_KWH).to_numpy(float); rate=tr['actual_kwh'].to_numpy(float)/cap
    el=np.isfinite(rate)&(rate>=0.10); cl=np.clip(np.nan_to_num(rate,nan=0.10),0.10,1.074999)
    rb=np.floor((cl-0.10)/0.02).astype(np.int16); ab=np.asarray(sorted(np.unique(rb[el])),dtype=np.int16)
    mp={int(b):i for i,b in enumerate(ab)}; cls=np.asarray([mp[int(v)] for v in rb[el]],dtype=np.int32)
    ce=np.asarray([float(np.mean(rate[el][cls==i])) for i in range(len(ab))],dtype=float)
    return el,cls,ce

def act(prob,ce,vg,mg,T,G):
    m=ce>=0.10; p=prob[:,m].astype(float); p/=p.sum(axis=1,keepdims=True); c=ce[m]
    cal=p**(1.0/T); cal/=cal.sum(axis=1,keepdims=True)
    err=np.abs(ACTIONS_CF[:,None]-c[None,:]); un=np.select([err<=0.06,err<=0.08],[4.0,3.0],default=0.0)
    ch=np.empty(len(p))
    for g in (1,2,3):
        s=vg==g; gp=cal[s]
        u=-(gp@err.T)+G*(gp@(c[None,:]*un).T)/(4.0*mg[g])
        ch[s]=ACTIONS_CF[np.argmax(u,axis=1)]
    return ch*pd.Series(vg).map(CAPACITIES_KWH).to_numpy(float)

def score(keys,vals,truth):
    k=keys.merge(truth,on=['forecast_kst_dtm','group_id'],how='left',validate='one_to_one')
    fr=pd.DataFrame({'forecast_id':k['forecast_id'],'forecast_kst_dtm':k['forecast_kst_dtm'],
        'group_id':k['group_id'],'actual_kwh':k['actual_kwh'],'prediction_kwh':vals})
    s=evaluate_official(fr,CAPACITIES_KWH)
    return {'total':float(s.total),'one_minus_nmae':float(s.one_minus_nmae),'ficr':float(s.ficr)}

def main():
    names=selected(); S=surface(names); lab=labels_prefix()
    truth=lab[['forecast_kst_dtm','group_id','actual_kwh']]
    A=pd.read_parquet(REPO/'artifacts/backtests/s17_n7_strict_actions/actions.parquet',
        columns=['fold_id','group_id','forecast_kst_dtm','M115_XGBOOST','CHAMPION'])
    A['forecast_kst_dtm']=pd.to_datetime(A['forecast_kst_dtm'])
    Ts=[0.60,0.75,0.90,1.00,1.25]; Gs=[1.0,1.5,2.0,2.5,3.0]
    grid={}; blended={}
    for fold in FOLDS:
        npz=PROBE/f'M64B_ALLWEATHER_SITEWIND_CLASS-{fold}-sitewind-features.npz'
        M=S.copy(); add_sitewind(M,npy_prefix(npz,'legacy.npy',PREFIX_ROWS),npy_prefix(npz,'allweather.npy',PREFIX_ROWS))
        vk=pd.read_parquet(PROBE/f'M115_XGBOOST-{fold}-policies.parquet',columns=['forecast_id','forecast_kst_dtm','group_id'])
        vk['forecast_kst_dtm']=pd.to_datetime(vk['forecast_kst_dtm']); start=vk['forecast_kst_dtm'].min()
        past=lab.loc[lab['forecast_kst_dtm']<start]
        tr=M.loc[M['forecast_kst_dtm']<start].merge(past[['forecast_kst_dtm','group_id','actual_kwh']],
            on=['forecast_kst_dtm','group_id'],how='inner',validate='one_to_one')
        va=vk.merge(M,on=['forecast_id','forecast_kst_dtm','group_id'],how='left',validate='one_to_one')
        el,cls,ce=contract(tr)
        mdl=XGBClassifier(num_class=len(ce),**MP)
        w=np.clip((tr['actual_kwh'].to_numpy(float)/tr['group_id'].map(CAPACITIES_KWH).to_numpy(float))[el],0.10,None)
        mdl.fit(tr[names[fold]].astype('float32').loc[el],cls,sample_weight=w)
        prob=mdl.predict_proba(va[names[fold]].astype('float32'),iteration_range=(0,100))
        rate=tr['actual_kwh'].to_numpy(float)/tr['group_id'].map(CAPACITIES_KWH).to_numpy(float)
        mg={g:float(np.mean(rate[(tr['group_id'].to_numpy()==g)&(rate>=0.10)])) for g in (1,2,3)}
        keys=va[['forecast_id','forecast_kst_dtm','group_id']]
        vg=va['group_id'].to_numpy(int)
        grid[fold]={}
        base_actions={}
        for T in Ts:
            for G in Gs:
                a=act(prob,ce,vg,mg,T,G)
                grid[fold][f'T{T}_G{G}']=score(keys,a,truth)
                base_actions[(T,G)]=a
        # blended champion-level for the deployed and exact rules
        r=A.loc[A['fold_id'].eq(fold)].merge(
            pd.DataFrame({'forecast_kst_dtm':keys['forecast_kst_dtm'],'group_id':keys['group_id'],
                          **{f'A_T{T}_G{G}':v for (T,G),v in base_actions.items()}}),
            on=['forecast_kst_dtm','group_id'],how='left',validate='one_to_one')
        cap=r['group_id'].map(CAPACITIES_KWH).to_numpy(float)
        blended[fold]={}
        for (T,G) in base_actions:
            blend=np.clip(r['CHAMPION'].to_numpy(float)+WEIGHT*(r[f'A_T{T}_G{G}'].to_numpy(float)-r['M115_XGBOOST'].to_numpy(float)),0.0,1.075*cap)
            blended[fold][f'T{T}_G{G}']=blend
        bl_keys=r[['forecast_kst_dtm','group_id']].copy(); bl_keys['forecast_id']=bl_keys['forecast_kst_dtm'].astype(str)+'_'+bl_keys['group_id'].astype(str)
        blended[fold]={'_keys':bl_keys,**blended[fold]}
        del M,tr,va,mdl,prob; gc.collect(); print('fold',fold,flush=True)
    # pooled outer blended scores
    pooled={}
    for tag in [f'T{T}_G{G}' for T in Ts for G in Gs]:
        parts=[]
        for fold in OUTER:
            k=blended[fold]['_keys'].copy(); k['prediction_kwh']=blended[fold][tag]; parts.append(k)
        P=pd.concat(parts,ignore_index=True)
        pooled[tag]=score(P[['forecast_id','forecast_kst_dtm','group_id']],P['prediction_kwh'].to_numpy(),truth)
    out={'raw_arm_grid':grid,'blended_pooled_outer':pooled,
         'deployed':'T0.75_G2.0','exact_gradient_rule':'T1.0_G1.0',
         'note':'G is the multiplier on the settlement term relative to the exact Total gradient, for which G=1 and T=1 is the exact plug-in Bayes rule.'}
    Path('reports/s17_n38_action_landscape.json').write_text(json.dumps(out,indent=1)+'\n')
    q2={k:v['total'] for k,v in grid['dev-2023-Q2'].items()}
    best_q2=max(q2,key=q2.get)
    print('Q2 raw best',best_q2,q2[best_q2],'| deployed',q2['T0.75_G2.0'],'| exact',q2['T1.0_G1.0'])
    print('blended pooled outer: deployed',pooled['T0.75_G2.0']['total'],'exact',pooled['T1.0_G1.0']['total'],
          '| Q2-selected',best_q2,pooled[best_q2]['total'])

main()
