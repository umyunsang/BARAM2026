"""S13-N1 * (S3/S4 rung, re-entered) exact decomposition of our point-forecast MAE.

Purpose within the ladder.  S12-N21 found that S5/S6/S7 were closed before the binding
constraint was known, so we are re-entering the ladder.  Before generating S5 preprocessing
nodes we need the S3-level fact that decides WHICH S5 family can possibly pay: how much of our
MAE is label/availability noise (attackable by preprocessing of the TARGET) and how much is
NWP -> hub-wind error (attackable only by features/models, i.e. S6/S7)?

The repository already carries the two quantities that make this decomposable exactly, on the
same hourly grid:
    cf        = metered capacity factor  (the scored truth)
    pc_true   = physics capacity factor obtained by integrating each group's fitted power curve
                over the MEASURED 10-minute nacelle wind  (research/scratch/teacher_targets.parquet)
    pc_hat    = our model's estimate of pc_true from NWP alone
so
    cf - pc_hat  =  (cf - pc_true)   +   (pc_true - pc_hat)
                    ^ availability /      ^ NWP -> hub-wind error
                      power-curve /         (feature & model addressable, S6/S7)
                      metering residual
                      (target-preprocessing addressable, S5)

Everything is evaluated on the rows the metric actually scores (cf >= 0.1) over the three
dev-2023 folds, so the shares are directly comparable to the 1-NMAE we are trying to move.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from harness import surface, MU
from lib import FOLDS, CAPS, official_total
import lightgbm as lgb

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
S = '/Users/um-yunsang/BARAM2026/research/scratch/'

if __name__ == '__main__':
    A, FR, COLS = surface(('G2', 'DROP:grid__'))
    T = pd.read_parquet(S + 'teacher_targets.parquet')
    cf = A['cf'].to_numpy(); grp = A['grp'].to_numpy(); idx = A.index
    pct = A['pc_true'].to_numpy()
    valid = np.isfinite(cf) & (cf >= 0.1)
    w_prod = np.where(valid, np.clip(cf, 0, 1.2), 0.05)

    # intra-hour measured-wind dispersion, for the "is the hour itself noisy" question
    vspread = np.concatenate([T[f'g{g}_v_spread'].reindex(idx).to_numpy() for g in (1, 2, 3)]) \
        if False else np.full(len(A), np.nan)
    for g in (1, 2, 3):
        m = grp == g
        vspread[m] = T[f'g{g}_v_spread'].reindex(idx[m]).to_numpy()

    rows = []
    for f, (a, b) in FOLDS.items():
        a = pd.Timestamp(a); b = pd.Timestamp(b)
        tr = np.asarray(idx < a); va = np.asarray((idx >= a) & (idx <= b))
        m = tr & np.isfinite(pct)
        mu = lgb.LGBMRegressor(**MU)
        mu.fit(A.loc[m, COLS], pct[m], sample_weight=w_prod[m])
        pch = np.clip(mu.predict(A[COLS]), 0, 1)
        sel = va & valid & np.isfinite(pct)
        rows.append(pd.DataFrame({'fold_id': f, 'group_id': grp[sel], 'dtm': idx[sel],
                                  'cf': cf[sel], 'pc_true': pct[sel], 'pc_hat': pch[sel],
                                  'v_spread': vspread[sel]}))
        print(f'  {f} teacher fitted', flush=True)
    D = pd.concat(rows, ignore_index=True)
    D['e_total'] = D.cf - D.pc_hat
    D['e_label'] = D.cf - D.pc_true      # availability / power-curve / metering
    D['e_nwp'] = D.pc_true - D.pc_hat    # NWP -> hub wind
    print(f'\nscored rows with a measured-wind teacher available: {len(D)}')

    out = {}
    print('\n--- MAE decomposition on scored rows (units = fraction of capacity) ---')
    for nm, c in [('|cf - pc_hat|  TOTAL', 'e_total'),
                  ('|cf - pc_true| LABEL/AVAILABILITY', 'e_label'),
                  ('|pc_true - pc_hat| NWP->WIND', 'e_nwp')]:
        v = D[c].abs()
        print(f'  {nm:36s} mean={v.mean():.5f}  median={v.median():.5f}  p90={v.quantile(0.9):.5f}')
        out[c] = {'mae': float(v.mean()), 'median': float(v.median()), 'p90': float(v.quantile(0.9))}
    rho = float(np.corrcoef(D.e_label, D.e_nwp)[0, 1])
    print(f'\n  corr(e_label, e_nwp) = {rho:+.4f}   (near 0 => the two channels are separable)')
    out['corr_label_nwp'] = rho

    # variance share (exact, additive) and MAE share (approximate)
    vt, vl, vn = D.e_total.var(), D.e_label.var(), D.e_nwp.var()
    print(f'  variance: total={vt:.6f}  label={vl:.6f} ({vl/vt:.1%})  nwp={vn:.6f} ({vn/vt:.1%})')
    out['variance_share'] = {'label': float(vl / vt), 'nwp': float(vn / vt)}

    print('\n--- counterfactual scores (what each channel is worth) ---')
    capv = D.group_id.map(CAPS).to_numpy()
    base = D[['group_id']].copy(); base['actual_kwh'] = D.cf * capv
    for nm, pred in [('our teacher pc_hat', D.pc_hat),
                     ('PERFECT wind (pc_true)', D.pc_true),
                     ('PERFECT label (cf) - sanity', D.cf)]:
        s = official_total(base.assign(prediction_kwh=np.clip(pred, 0, 1.1) * capv))
        print(f'  {nm:28s} 1-NMAE={s["one_minus_nmae"]:.6f}  FICR={s["ficr"]:.6f}  Total={s["total"]:.6f}')
        out[nm] = {'one_minus_nmae': s['one_minus_nmae'], 'ficr': s['ficr'], 'total': s['total']}

    print('\n--- how concentrated is the label channel? (availability deficit) ---')
    D['deficit'] = D.pc_true - D.cf
    for th in (0.02, 0.05, 0.10, 0.20):
        m = D.deficit >= th
        share_rows = m.mean()
        share_mae = D.loc[m, 'e_total'].abs().sum() / D.e_total.abs().sum()
        print(f'  deficit >= {th:.2f}: {share_rows:6.2%} of scored rows, '
              f'{share_mae:6.2%} of total absolute error')
        out[f'deficit_{th}'] = {'row_share': float(share_rows), 'mae_share': float(share_mae)}

    print('\n--- MAE of the NWP channel by intra-hour measured-wind dispersion quartile ---')
    D['vq'] = pd.qcut(D.v_spread, 4, labels=['Q1 calm', 'Q2', 'Q3', 'Q4 gusty'], duplicates='drop')
    print(D.groupby('vq', observed=True).agg(n=('e_nwp', 'size'),
                                             mae_nwp=('e_nwp', lambda x: x.abs().mean()),
                                             mae_label=('e_label', lambda x: x.abs().mean()),
                                             mae_total=('e_total', lambda x: x.abs().mean())).round(5).to_string())
    json.dump(out, open(N + 'S13-N1_mae_decomposition.json', 'w'), indent=1, default=str)
    D.to_parquet(N + 'S13-N1_decomp.parquet', index=False)
