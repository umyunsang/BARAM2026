
"""Staged excavation harness.

One frozen surface so that every node differs in EXACTLY the treatment it declares.

Fixed (screening) protocol
  train  : forecast_kst_dtm <  2023-04-01 01:00
  valid  : 2023-04-01 01:00 .. 2024-01-01 00:00   (= pooled dev-2023 Q2+Q3+Q4)
  scoring: src/baram/evaluation/official.py semantics, pooled once, single policy
  lockbox: 2024 never touched

Architecture (frozen while S5 nodes run)
  A  teacher   : features -> pc      (curve-integrated capacity factor from measured wind)
  B  calibrator: cf | pc_hat         (conditional predictive distribution)
  C  decision  : settlement-optimal action over that distribution
Each stage exposes named hooks that a node may override; everything else is byte-identical.
"""
from __future__ import annotations
import sys, json, time, hashlib
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0,'/Users/um-yunsang/BARAM2026/research/nodes')
import featbuild
from lib import official_total, CAPS, sharpen_weights

S='/Users/um-yunsang/BARAM2026/research/scratch/'
N='/Users/um-yunsang/BARAM2026/research/nodes/'
SPLIT=pd.Timestamp('2023-04-01 01:00:00'); END=pd.Timestamp('2024-01-01 00:00:00')

MU=dict(objective='l2', n_estimators=900, learning_rate=0.035, num_leaves=63,
        min_child_samples=40, subsample=0.85, subsample_freq=1, colsample_bytree=0.4,
        reg_lambda=3.0, random_state=20260801, n_jobs=6, verbose=-1)
SG=dict(MU, n_estimators=500, num_leaves=31)
ACTIONS=np.arange(0.05,1.0801,0.0025)
TEMPS=(0.4,0.5,0.6,0.75,1.0)
GAMMAS=(0.0,0.35,0.75,1.0,1.5,2.0)

_CACHE={}

def surface(blocks=()):
    """Pooled feature/target frame; built once per (block-set) per process."""
    key=tuple(sorted(blocks))
    if key in _CACHE: return _CACHE[key]
    drop_pref = tuple(p[5:] for p in key if p.startswith('DROP:'))
    key_blocks = tuple(p for p in key if not p.startswith('DROP:'))
    import s6feats
    T=pd.read_parquet(S+'teacher_targets.parquet')
    LAB=pd.read_parquet(S+'labels.parquet').set_index('kst_dtm')
    FR={}
    for g in (1,2,3):
        X=featbuild.build2(g, geom=True, grid=True)
        if key_blocks:
            X=pd.concat([X, s6feats.build_blocks(g, X, key_blocks)], axis=1)
            X=X.loc[:, ~X.columns.duplicated()]
        if drop_pref:
            X=X.loc[:, [c for c in X.columns if not c.startswith(drop_pref)]]
        for k in (1,2,3): X[f'is_g{k}']=np.float32(k==g)
        X['pc_true']=T[f'g{g}_pc'].reindex(X.index).to_numpy()
        X['cf']=(LAB[f'kpx_group_{g}']/CAPS[g]).reindex(X.index).to_numpy()
        X['grp']=np.int8(g)
        FR[g]=X
    COLS=[c for c in FR[1].columns if c not in ('pc_true','cf','grp')]
    A=pd.concat(FR.values())
    _CACHE[key]=(A, FR, COLS)
    return A, FR, COLS


DEFAULT=dict(
    # --- S5 preprocessing hooks -------------------------------------------------
    teacher_weight = None,      # callable(df)->array | None
    teacher_rows   = None,      # callable(df)->bool array | None
    calib_rows     = None,      # callable(df)->bool array | None   (P1 gating lives here)
    calib_weight   = None,
    soft_cap       = None,      # dict g->float | None              (P7)
    group_offset   = None,      # dict g->float | None              (post-gate recalibration)
    # --- S6/S7 hooks (frozen during S5) ----------------------------------------
    mu_params      = MU,
    sigma_params   = SG,
    n_quantile     = 41,
    conditional_sigma = True,
)


def _fit_pooled(A, COLS, tr, params, target, weight=None, rows=None):
    m = tr & np.isfinite(A[target].to_numpy())
    if rows is not None: m = m & rows
    w = None if weight is None else weight[m]
    mdl = lgb.LGBMRegressor(**params)
    mdl.fit(A.loc[m, COLS], A.loc[m, target], sample_weight=w)
    return mdl, int(m.sum())


def run(node_id: str, tag: str, blocks=(), **over):
    cfg = dict(DEFAULT); cfg.update(over)
    A, FR, COLS = surface(blocks)
    tr = np.asarray(A.index < SPLIT)
    t0=time.time()

    # --- A: teacher ------------------------------------------------------------
    w = None if cfg['teacher_weight'] is None else cfg['teacher_weight'](A)
    r = None if cfg['teacher_rows']  is None else cfg['teacher_rows'](A)
    mu_mdl, n_mu = _fit_pooled(A, COLS, tr, cfg['mu_params'], 'pc_true', w, r)
    A['pc_hat'] = np.clip(mu_mdl.predict(A[COLS]), 0, 1)

    # --- B: calibrator ---------------------------------------------------------
    cal = tr & np.isfinite(A['cf'].to_numpy()) & np.isfinite(A['pc_hat'].to_numpy())
    if cfg['calib_rows'] is not None: cal = cal & cfg['calib_rows'](A)
    A['resid'] = A['cf'] - A['pc_hat']
    if cfg['conditional_sigma']:
        A['absres'] = np.abs(A['resid'])
        sg_mdl,_ = _fit_pooled(A, COLS, cal, cfg['sigma_params'], 'absres',
                               None if cfg['calib_weight'] is None else cfg['calib_weight'](A), None)
        A['sigma'] = np.clip(sg_mdl.predict(A[COLS]), 1e-3, 1.0)
    else:
        A['sigma'] = 1.0
    qs = np.linspace(0.01, 0.99, cfg['n_quantile'])
    nb = int(cfg.get('calib_buckets', 1))
    ztab = {}; bedges = {}
    for g in (1,2,3):
        m = cal & (A['grp'].to_numpy()==g)
        z = (A.loc[m,'resid']/A.loc[m,'sigma']).to_numpy()
        p = A.loc[m,'pc_hat'].to_numpy()
        ok = np.isfinite(z)
        if nb <= 1:
            ztab[g] = np.quantile(z[ok], qs)[None, :]
            bedges[g] = np.array([-np.inf, np.inf])
        else:
            e = np.quantile(p[ok], np.linspace(0, 1, nb+1)); e[0] = -np.inf; e[-1] = np.inf
            e = np.unique(e)
            tab = []
            for b in range(len(e)-1):
                sel = ok & (p >= e[b]) & (p < e[b+1])
                if sel.sum() < 120: sel = ok
                tab.append(np.quantile(z[sel], qs))
            ztab[g] = np.asarray(tab); bedges[g] = e

    # --- C: decision -----------------------------------------------------------
    parts=[]
    for g in (1,2,3):
        X = FR[g]; va = np.asarray((X.index>=SPLIT)&(X.index<=END))
        sel = (A['grp'].to_numpy()==g) & np.asarray((A.index>=SPLIT)&(A.index<=END))
        pc = A.loc[sel,'pc_hat'].to_numpy(); sd = A.loc[sel,'sigma'].to_numpy()
        cf = A.loc[sel,'cf'].to_numpy()
        keep = np.isfinite(cf)
        cap_hi = 1.05 if cfg['soft_cap'] is None else cfg['soft_cap'][g]
        off = 0.0 if cfg['group_offset'] is None else cfg['group_offset'][g]
        bi = np.clip(np.searchsorted(bedges[g], pc, side='right')-1, 0, ztab[g].shape[0]-1)
        infl = float(cfg.get('var_inflate', 1.0))
        samples = np.clip(pc[:,None] + off + infl*sd[:,None]*ztab[g][bi], 0.0, cap_hi)
        mean_gen = float(np.nanmean(A.loc[(A['grp'].to_numpy()==g)&tr,'cf']))
        err = np.abs(ACTIONS[None,:,None] - samples[:,None,:])
        units = np.where(err<=0.06,4.0,np.where(err<=0.08,3.0,0.0))
        rec = {'group_id':g,'actual_kwh':cf[keep]*CAPS[g],'pc_hat':pc[keep],'sigma':sd[keep]}
        for tp in TEMPS:
            wq = sharpen_weights(samples, tp)[:,None,:]
            nm = -(err*wq).sum(axis=2)
            fi = ((samples[:,None,:]*units)*wq).sum(axis=2)/(4.0*mean_gen)
            for gm in GAMMAS:
                a = ACTIONS[np.argmax(nm+gm*fi, axis=1)]
                rec[f'T{tp}_G{gm}'] = np.clip(a, 0, cap_hi)[keep]*CAPS[g]
        parts.append(pd.DataFrame(rec))
    P = pd.concat(parts, ignore_index=True)
    scores={}
    for tp in TEMPS:
        for gm in GAMMAS:
            c=f'T{tp}_G{gm}'
            scores[c]=official_total(P.rename(columns={c:'prediction_kwh'})[
                ['group_id','actual_kwh','prediction_kwh']])
    ser=pd.Series({k:v['total'] for k,v in scores.items()}).sort_values(ascending=False)
    best=ser.index[0]
    raw=official_total(P.assign(prediction_kwh=P.pc_hat*P.group_id.map(CAPS))[
        ['group_id','actual_kwh','prediction_kwh']])
    out=dict(node_id=node_id, tag=tag, best_policy=best, best=float(ser.iloc[0]),
             one_minus_nmae=scores[best]['one_minus_nmae'], ficr=scores[best]['ficr'],
             raw_point=raw['total'], raw_1mnmae=raw['one_minus_nmae'],
             grid={k:float(v) for k,v in ser.items()}, teacher_rows=n_mu,
             secs=round(time.time()-t0,1))
    print(f'{node_id:10s} {tag:36s} best={out["best"]:.6f} ({best})  '
          f'1-NMAE={out["one_minus_nmae"]:.6f} FICR={out["ficr"]:.6f} raw={out["raw_point"]:.6f} '
          f'[{out["secs"]}s]', flush=True)
    json.dump(out, open(N+f'{node_id}.json','w'), indent=1)
    P.to_parquet(N+f'{node_id}.parquet', index=False)
    return out
