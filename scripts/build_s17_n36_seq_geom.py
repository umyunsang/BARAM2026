"""S17-N36 T0: build FD10 (seq__) and FD11 (geom__) blocks. No fit, no label, no metric."""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq

REPO=Path('.')
CACHE=REPO/'artifacts/cache/920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b'
OUT=REPO/'artifacts/features/s17_n36'; OUT.mkdir(parents=True,exist_ok=True)

GEOM_VARS={'gfs':['heightAboveGround_10_10u','heightAboveGround_10_10v','heightAboveGround_100_100u',
                  'heightAboveGround_100_100v','meanSea_0_prmsl','surface_0_sp','heightAboveGround_2_2t',
                  'wind100_speed','wind10_speed'],
           'ldaps':['heightAboveGround_10_10u','heightAboveGround_10_10v','meanSea_0_prmsl','surface_0_sp',
                    'heightAboveGround_2_t','wind10_speed','etc_0_blh']}
PAIRS={'gfs':[('heightAboveGround_10_10u','heightAboveGround_10_10v','w10'),
              ('heightAboveGround_100_100u','heightAboveGround_100_100v','w100')],
       'ldaps':[('heightAboveGround_10_10u','heightAboveGround_10_10v','w10')]}

def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as s:
        while b:=s.read(8<<20): h.update(b)
    return h.hexdigest()

def grid_coords(split:str)->dict[str,pd.DataFrame]:
    import zipfile,io
    z=zipfile.ZipFile(REPO/'inputs/competition/open_wind_236727.zip')
    member={'train':{'gfs':'train/gfs_train.csv','ldaps':'train/ldaps_train.csv'},
            'test':{'gfs':'test/gfs_test.csv','ldaps':'test/ldaps_test.csv'}}[split]
    out={}
    for src,m in member.items():
        with z.open(m) as fh:
            d=pd.read_csv(fh,usecols=['grid_id','latitude','longitude']).drop_duplicates('grid_id')
        d=d.sort_values('grid_id').reset_index(drop=True)
        out[src]=d
    return out

def plane_fit(values:np.ndarray,X:np.ndarray)->np.ndarray:
    """values (n_rows, n_grid); X (n_grid,3) [1, dx_km, dy_km]. Returns (n_rows,2) b,c."""
    coef,_,_,_=np.linalg.lstsq(X,values.T,rcond=None)   # (3, n_rows)
    return coef[1:].T

def build(split:str)->tuple[pd.DataFrame,dict]:
    pivot_path=CACHE/f'{split}_grid_pivot.parquet'
    names=pq.ParquetFile(pivot_path).schema_arrow.names
    coords=grid_coords(split)
    meta={'pivot_sha256':sha(pivot_path),'blocks':{}}
    frames=[]
    base=pd.read_parquet(pivot_path,columns=['forecast_kst_dtm'])
    base['forecast_kst_dtm']=pd.to_datetime(base['forecast_kst_dtm'])
    out=base.copy()
    for src,varlist in GEOM_VARS.items():
        c=coords[src]; lat0=c['latitude'].mean(); lon0=c['longitude'].mean()
        dx=(c['longitude'].to_numpy()-lon0)*111.320*np.cos(np.radians(lat0))
        dy=(c['latitude'].to_numpy()-lat0)*110.574
        X=np.column_stack([np.ones(len(c)),dx,dy])
        ngrid=len(c)
        cache={}
        for var in varlist:
            cols=[f'{src}__grid{g:02d}__{var}' for g in range(1,ngrid+1)]
            if any(cl not in names for cl in cols): continue
            V=pd.read_parquet(pivot_path,columns=cols).to_numpy(dtype=np.float64)
            cache[var]=V
            g=plane_fit(V,X)
            out[f'geom__{src}__{var}__ddx']=g[:,0].astype(np.float32)
            out[f'geom__{src}__{var}__ddy']=g[:,1].astype(np.float32)
            out[f'geom__{src}__{var}__gradmag']=np.hypot(g[:,0],g[:,1]).astype(np.float32)
        for ucol,vcol,tag in PAIRS[src]:
            if ucol in cache and vcol in cache:
                gu=plane_fit(cache[ucol],X); gv=plane_fit(cache[vcol],X)
                out[f'geom__{src}__{tag}__div']=(gu[:,0]+gv[:,1]).astype(np.float32)
                out[f'geom__{src}__{tag}__vort']=(gv[:,0]-gu[:,1]).astype(np.float32)
        meta['blocks'][src]={'grids':ngrid,'lat0':float(lat0),'lon0':float(lon0),
                             'vars':[v for v in varlist if f'{src}__grid01__{v}' in names]}
    return out,meta

def add_seq(out:pd.DataFrame)->pd.DataFrame:
    """Within-issuance temporal operators. Operating day = (t - 1h).normalize()."""
    d=out.copy()
    d['operating_day']=(d['forecast_kst_dtm']-pd.Timedelta(hours=1)).dt.normalize()
    d=d.sort_values('forecast_kst_dtm',kind='stable').reset_index(drop=True)
    src_cols=[c for c in d.columns if c.startswith('geom__') and c.endswith('gradmag')]
    seq_src=src_cols[:6]
    g=d.groupby('operating_day',sort=False)
    for c in seq_src:
        s=d[c]
        d[f'seq__{c}__lag1']=g[c].shift(1).astype('float32')
        d[f'seq__{c}__lead1']=g[c].shift(-1).astype('float32')
        d[f'seq__{c}__lag3']=g[c].shift(3).astype('float32')
        d[f'seq__{c}__lead3']=g[c].shift(-3).astype('float32')
        d[f'seq__{c}__c3mean']=g[c].transform(lambda x:x.rolling(3,center=True,min_periods=1).mean()).astype('float32')
        d[f'seq__{c}__c5sd']=g[c].transform(lambda x:x.rolling(5,center=True,min_periods=2).std()).astype('float32')
        d[f'seq__{c}__daymean']=g[c].transform('mean').astype('float32')
        d[f'seq__{c}__dayrank']=g[c].rank(pct=True).astype('float32')
    return d.drop(columns=['operating_day'])

def main()->None:
    report={}
    for split in ('train','test'):
        out,meta=build(split)
        out=add_seq(out)
        path=OUT/f'{split}_seq_geom.parquet'
        out.to_parquet(path,index=False)
        newcols=[c for c in out.columns if c!='forecast_kst_dtm']
        finite={c:float(np.isfinite(out[c].to_numpy(dtype=np.float64)).mean()) for c in newcols}
        nunique={c:int(out[c].nunique(dropna=True)) for c in newcols}
        report[split]={**meta,'path':str(path),'sha256':sha(path),'rows':int(len(out)),
            'n_new_columns':len(newcols),'geom_columns':sum(c.startswith('geom__') for c in newcols),
            'seq_columns':sum(c.startswith('seq__') for c in newcols),
            'min_finite_fraction':min(finite.values()),'constant_columns':[c for c,u in nunique.items() if u<=1]}
    Path('reports/s17_n36_seq_geom_build.json').write_text(json.dumps(report,indent=1,default=str)+'\n')
    print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='blocks'} for k,v in report.items()},indent=1))

main()
