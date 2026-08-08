"""S12-N20 * the period offset is NOT a constant -- it has the opposite SIGN for us.

S12-N19 measured the organiser's own baseline on our local protocol and found:

                          local (dev-2023 Q2-Q4)   online (2024 test)   offset
  organiser RF baseline   1-NMAE 0.850785          0.86371              +0.012925
  our lineage (M266/blend) 1-NMAE 0.861866         0.858775             -0.003091

The organiser's plain RandomForest GAINS 0.0129 of 1-NMAE moving from our 2023 window to the
2024 test period.  Our pipeline LOSES 0.0031 on the same move.  The relative transfer
disadvantage is 0.016016 -- larger than the entire 0.014839 of 1-NMAE that separates us from
Total 0.66.  So the earlier S12 reading ("our point accuracy is simply below the field") is
only half right: on the online scale our best local point forecast maps to roughly 0.870-0.878,
i.e. around the field median, and what actually costs us is that our advantage over a plain RF
does not survive the year change.

AGENTS.md already warns that the local->online offset does not transfer across method classes.
N19 sharpens that warning: it does not even keep its SIGN.

Hypothesis for the mechanism: our local folds are dev-2023 Q2/Q3/Q4, i.e. April-December.
They contain NO WINTER.  The external lane records an independent participant measurement that
at the same wind speed winter generation is 1.37x summer (air density), and that a team using a
seasonally skewed validation window saw its local ranking invert on the leaderboard until they
switched to seasonally representative folds.  If our margin over the baseline decays as the
months get colder, the missing season is the mechanism and the local protocol is blind to it.

This node tests that with no new data: decompose both the organiser baseline and our incumbent
by calendar month over the three existing folds and compare the margin.
"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')
from loop_lib import *

N = '/Users/um-yunsang/BARAM2026/research/nodes/'


def nmae_by(df, pred_col, by):
    out = []
    for k, s in df.groupby(by, observed=True):
        cap = s.group_id.map(CAPS)
        v = s[s.actual_kwh >= 0.1 * cap]
        if len(v) < 40:
            continue
        capv = v.group_id.map(CAPS)
        err = (v[pred_col] - v.actual_kwh).abs() / capv
        u = np.select([err <= 0.06, err <= 0.08], [4.0, 3.0], 0.0)
        out.append(dict(key=k, n=len(v), one_minus_nmae=1 - float(err.mean()),
                        hit6=float((err <= 0.06).mean()),
                        ficr=float((v.actual_kwh * u).sum() / (v.actual_kwh * 4).sum())))
    return pd.DataFrame(out)


if __name__ == '__main__':
    BL = pd.read_parquet(N + 'S12-N19_baseline_local.parquet')
    R = canonical_keys(); dep = load_depavg()
    Dm, _, _ = fo_policy(utility_frames(align_prob('D', R), R), R)
    J = Dm[KEY + ['actual_kwh', 'prediction_kwh']].rename(columns={'prediction_kwh': 'D'}).merge(dep, on=KEY)
    J['OURS'] = 0.30 * J.D + 0.70 * J.DEPAVG
    # our best local POINT forecast (no decision layer) for a like-for-like accuracy contrast
    C26 = (np.arange(26) + 0.5) * W
    pt = pd.DataFrame({'fold_id': R.fold_id, 'group_id': R.group_id,
                       'forecast_kst_dtm': R.forecast_kst_dtm,
                       'DPOINT': (align_prob('D', R) * C26[None, :]).sum(axis=1)
                                 * R.group_id.map(CAPS).to_numpy()})
    J = J.merge(pt, on=KEY)
    M = J.merge(BL[KEY + ['prediction_kwh']].rename(columns={'prediction_kwh': 'BASE'}), on=KEY)
    M['month'] = pd.to_datetime(M.forecast_kst_dtm).dt.month
    print(f'joined rows {len(M)}')

    tabs = {c: nmae_by(M, c, 'month').set_index('key') for c in ['BASE', 'OURS', 'DPOINT']}
    T = pd.DataFrame({'n': tabs['BASE'].n,
                      'base_1mnmae': tabs['BASE'].one_minus_nmae,
                      'ours_1mnmae': tabs['OURS'].one_minus_nmae,
                      'dpoint_1mnmae': tabs['DPOINT'].one_minus_nmae})
    T['margin_ours'] = T.ours_1mnmae - T.base_1mnmae
    T['margin_point'] = T.dpoint_1mnmae - T.base_1mnmae
    T['base_ficr'] = tabs['BASE'].ficr
    T['ours_ficr'] = tabs['OURS'].ficr
    print('\n--- 1-NMAE by calendar month (all within dev-2023 Q2-Q4) ---')
    print(T.round(5).to_string())

    print('\n--- margin over the organiser baseline, by season block ---')
    M['season'] = pd.cut(M.month, [3, 5, 8, 10, 12],
                         labels=['spring(4-5)', 'summer(6-8)', 'autumn(9-10)', 'early-winter(11-12)'])
    for c in ['BASE', 'OURS', 'DPOINT']:
        tabs[c] = nmae_by(M, c, 'season').set_index('key')
    S = pd.DataFrame({'n': tabs['BASE'].n,
                      'base': tabs['BASE'].one_minus_nmae,
                      'ours': tabs['OURS'].one_minus_nmae,
                      'dpoint': tabs['DPOINT'].one_minus_nmae})
    S['margin_ours'] = S.ours - S.base
    S['margin_point'] = S.dpoint - S.base
    print(S.round(5).to_string())

    # correlate margin with a proxy for air density / temperature season
    corr = float(np.corrcoef(T.index.to_numpy(float), T.margin_point.to_numpy())[0, 1])
    print(f'\n  corr(month, point-forecast margin over baseline) = {corr:+.4f}')
    print(f'  margin range across months: {T.margin_point.min():+.5f} .. {T.margin_point.max():+.5f}')
    json.dump({'by_month': T.reset_index().to_dict('records'),
               'by_season': S.reset_index().to_dict('records'),
               'corr_month_margin': corr},
              open(N + 'S12-N20_transfer_by_month.json', 'w'), indent=1, default=str)
