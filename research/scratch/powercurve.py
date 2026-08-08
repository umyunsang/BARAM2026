
"""Fit an empirical per-group power curve at 10-minute resolution and build
curve-integrated teacher targets.  Target space change: instead of regressing the
hourly MEAN wind, regress  pc = mean_{turbine,10min} f_g(ws)  which is the physically
correct expectation of capacity factor under a known curve.
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize

S='/Users/um-yunsang/BARAM2026/research/scratch/'
ve=pd.read_parquet(S+'scada_vestas.parquet'); un=pd.read_parquet(S+'scada_unison.parquet')
lab=pd.read_parquet(S+'labels.parquet').set_index('kst_dtm')
def he(s): return (s-pd.Timedelta("1s")).dt.ceil('h')
ve['he']=he(ve['kst_dtm']); un['he']=he(un['kst_dtm'])
GT={1:('vestas',range(1,7)),2:('vestas',range(7,13)),3:('unison',range(1,6))}
SRC={'vestas':ve,'unison':un}

def curve(v, vin, vr, vout, k):
    x=np.clip((v-vin)/np.maximum(vr-vin,0.1),0,1)
    f=x**k
    f=np.where(v>=vr,1.0,f)
    f=np.where((v<vin)|(v>vout),0.0,f)
    return f

WS={}
for g,(src,rng) in GT.items():
    df=SRC[src]
    cols=[f'{src}_wtg{i:02d}_ws' for i in rng]
    WS[g]=(df[['he']+cols], cols)

def pc_hourly(g, p):
    frame, cols = WS[g]
    v=frame[cols].to_numpy(float)
    f=curve(v,*p)
    m=np.nanmean(f,axis=1)
    s=pd.Series(m, index=frame['he'].to_numpy())
    return s.groupby(level=0).mean()

params={}
for g in (1,2,3):
    cap = 21600.0 if g<3 else 21000.0
    y = lab[f'kpx_group_{g}']/cap
    def obj(p):
        vin,vr,vout,k=p
        if not (1.0<vin<5.5 and vin+3<vr<16 and 15<vout<30 and 1.0<k<5.0): return 1e6
        s=pc_hourly(g,(vin,vr,vout,k))
        j=pd.concat([s.rename('p'), y.rename('y')],axis=1).dropna()
        return float(np.abs(j.p-j.y).mean())
    best=None
    for vin in (2.5,3.0,3.5,4.0):
        for vr in (9.5,10.5,11.5,12.5,13.5):
            for k in (1.5,2.0,2.5,3.0):
                v=obj((vin,vr,22.0,k))
                if best is None or v<best[0]: best=(v,(vin,vr,22.0,k))
    r=minimize(obj, best[1], method='Nelder-Mead',
               options=dict(maxiter=400,xatol=1e-3,fatol=1e-6))
    params[g]=tuple(r.x); print('g',g,'MAE(cf)',round(r.fun,5),'params',np.round(r.x,3), flush=True)

out={}
for g in (1,2,3):
    frame, cols = WS[g]
    v=frame[cols].to_numpy(float)
    f=curve(v,*params[g])
    he_idx=frame['he'].to_numpy()
    d=pd.DataFrame({'he':he_idx,'f_mean':np.nanmean(f,axis=1),'f_std':np.nanstd(f,axis=1),
                    'v_mean':np.nanmean(v,axis=1),'v_std':np.nanstd(v,axis=1),
                    'v_min':np.nanmin(v,axis=1),'v_max':np.nanmax(v,axis=1)})
    gg=d.groupby('he')
    out[f'g{g}_pc']       = gg['f_mean'].mean()
    out[f'g{g}_pc_intra'] = gg['f_mean'].std()
    out[f'g{g}_pc_spread']= gg['f_std'].mean()
    out[f'g{g}_v_mean']   = gg['v_mean'].mean()
    out[f'g{g}_v_intra']  = gg['v_mean'].std()
    out[f'g{g}_v_p10']    = gg['v_mean'].quantile(0.1)
    out[f'g{g}_v_p90']    = gg['v_mean'].quantile(0.9)
    out[f'g{g}_v_spread'] = gg['v_std'].mean()
T=pd.DataFrame(out); T.index.name='forecast_kst_dtm'
T.to_parquet(S+'teacher_targets.parquet')
import json; json.dump({str(k):list(map(float,v)) for k,v in params.items()}, open(S+'powercurve_params.json','w'), indent=1)
print(T.shape); print(T.describe().round(3).to_string())
