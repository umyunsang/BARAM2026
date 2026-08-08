"""S13-N2 * (S3 rung) is the label channel a fixable OFFSET or irreducible OUTAGE noise?

S13-N1 split our point error exactly:
    total MAE 0.13858 = label/availability 0.04804  +  NWP->hub-wind 0.13022
and found the label channel concentrated in gusty hours (MAE 0.02779 in the calmest
intra-hour-dispersion quartile rising to 0.08091 in the gustiest) while the NWP channel is
flat across those quartiles (0.12885 -> 0.12254).  A label error that grows with within-hour
wind variability is not what random turbine outages look like -- outages are exogenous to the
weather.  It is what a MIS-SPECIFIED HIGH-WIND REGIME looks like: V126 and U136 both apply
storm control / cut-out above roughly 20-25 m/s, and a power curve fitted without that regime
will claim full output during hours the fleet was actually de-rating.

This matters because `pc_true` is the TEACHER TARGET of the whole two-stage architecture.  If
`pc_true` is biased relative to the metered truth in an identifiable regime, every downstream
stage inherits that bias, and fixing it is a pure S5 preprocessing action.

Question answered here: how much of the 0.04804 label MAE is (a) a systematic, regime-indexed
offset that preprocessing can remove, versus (b) unpredictable outage noise that nothing can.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from lib import CAPS

N = '/Users/um-yunsang/BARAM2026/research/nodes/'
S = '/Users/um-yunsang/BARAM2026/research/scratch/'

D = pd.read_parquet(N + 'S13-N1_decomp.parquet')
T = pd.read_parquet(S + 'teacher_targets.parquet')
D['v_mean'] = [T.loc[d, f'g{g}_v_mean'] if d in T.index else np.nan
               for d, g in zip(D.dtm, D.group_id)]
D['deficit'] = D.pc_true - D.cf

print(f'rows {len(D)}')
print('\n--- deficit = pc_true - cf, distribution ---')
q = D.deficit.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
print(q.round(4).to_string())
print(f'  mean={D.deficit.mean():+.5f}  median={D.deficit.median():+.5f}  '
      f'share(deficit>0)={float((D.deficit>0).mean()):.3f}')

print('\n--- deficit by MEASURED hub wind speed bin (the regime test) ---')
D['vbin'] = pd.cut(D.v_mean, [0, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 40])
tab = D.groupby('vbin', observed=True).agg(
    n=('deficit', 'size'), med_deficit=('deficit', 'median'), mean_deficit=('deficit', 'mean'),
    mean_pc=('pc_true', 'mean'), mean_cf=('cf', 'mean'),
    mae_label=('e_label', lambda x: x.abs().mean()))
print(tab.round(4).to_string())

print('\n--- per group, deficit by wind bin (median) ---')
piv = D.pivot_table(index='vbin', columns='group_id', values='deficit',
                    aggfunc='median', observed=True)
print(piv.round(4).to_string())

# how much MAE would a REGIME-INDEXED offset correction remove?  (in-sample upper bound)
print('\n--- upper bound: subtract the in-sample median deficit per (group, wind bin) ---')
med = D.groupby(['group_id', 'vbin'], observed=True).deficit.median()
D['pc_corr'] = D.pc_true - D.set_index(['group_id', 'vbin']).index.map(med).to_numpy()
for nm, c in [('raw   |cf - pc_true|', 'pc_true'), ('corrected |cf - pc_corr|', 'pc_corr')]:
    v = (D.cf - D[c]).abs()
    print(f'  {nm:26s} MAE={v.mean():.5f}')
resid_after = (D.cf - D.pc_corr).abs().mean()
print(f'  => systematic (removable) part = {0.04804 - resid_after:.5f} of the 0.04804 label MAE'
      f'  ({(0.04804-resid_after)/0.04804:.1%})')
print(f'  => irreducible outage-like part = {resid_after:.5f}')

print('\n--- persistence: is the deficit autocorrelated (i.e. multi-hour outages)? ---')
for g in (1, 2, 3):
    s = D[D.group_id == g].sort_values('dtm').set_index('dtm').deficit
    s = s.asfreq('h') if s.index.inferred_freq else s
    ac = [round(float(s.autocorr(lag=k)), 3) for k in (1, 2, 3, 6, 12, 24, 48, 168)]
    print(f'  g{g} autocorr at lags 1,2,3,6,12,24,48,168h: {ac}')

out = {'deficit_quantiles': q.round(6).to_dict(),
       'label_mae_raw': 0.04804, 'label_mae_after_regime_offset': float(resid_after),
       'removable_share': float((0.04804 - resid_after) / 0.04804),
       'by_wind_bin': tab.round(6).reset_index().astype(str).to_dict('records')}
json.dump(out, open(N + 'S13-N2_label_channel.json', 'w'), indent=1, default=str)
