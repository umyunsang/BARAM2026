"""S14-N1 * does the deployment distribution actually differ from the one we select on?

Motivation.  Two independent measurements say our local protocol and the deployment period are
not the same distribution:
  * S12-N19: the organiser's own RandomForest GAINS +0.012925 of 1-NMAE moving from our window to
    the graded period, while our pipeline LOSES -0.003091 on the same move -- a relative transfer
    disadvantage of 0.016016, larger than the 0.014839 we need to hit the target.
  * The engine's data audit: train inputs span 2022-01-01..2025-01-01, the graded TEST inputs span
    2025-01-01..2026-01-01 (8760 hours x 3 groups, all 830 columns shared and SUPPLIED), while our
    fold-outside protocol selects on dev-2023 Q2/Q3/Q4 only -- April to December, NO WINTER.

Reading test-period INPUTS is not a lockbox action: the lockbox is the 2024 outcome, and nothing
here touches any label or any score.  This node uses only the supplied feature matrix.

Method (domain-classifier two-sample test, the standard covariate-shift diagnostic):
train a classifier to tell train-period rows from test-period rows using the shared columns.
AUC 0.5 means the two are indistinguishable and importance weighting is pointless; AUC well above
0.5 means the deployment inputs are systematically different and w(x) = p_te(x)/p_tr(x) carries
real information.  Reported in three nested variants so a trivial explanation cannot masquerade
as a finding:
   ALL      every shared column, including explicit calendar/time indices
   NOTIME   calendar and time-index columns removed -- genuine METEOROLOGICAL shift only
   NOTIME_SEASONMATCHED  as NOTIME, but the test set is restricted to the months our folds
                         actually contain (Apr-Dec), which isolates "different weather" from
                         "we never validate on winter"
"""
import sys, json, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/scratch')
sys.path.insert(0, '/Users/um-yunsang/BARAM2026/research/nodes')

C = ('/Users/um-yunsang/BARAM2026/artifacts/cache/'
     '920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b/')
N = '/Users/um-yunsang/BARAM2026/research/nodes/'
TIME_LIKE = ('forecast_kst_dtm', 'data_available_kst_dtm', 'forecast_id', 'operating_day',
             'operating_year', 'operating_quarter', 'issuance_batch', 'lead_hour',
             'hour', 'month', 'day_of_year', 'cal__hour_sin', 'cal__hour_cos',
             'cal__doy_sin', 'cal__doy_cos')
CLF = dict(objective='binary', n_estimators=400, learning_rate=0.05, num_leaves=63,
           min_child_samples=60, subsample=0.85, subsample_freq=1, colsample_bytree=0.5,
           reg_lambda=5.0, random_state=20260807, n_jobs=6, verbose=-1)


def load():
    tr = pd.read_parquet(C + 'train_features.parquet')
    te = pd.read_parquet(C + 'test_features.parquet')
    return tr, te


def domain_auc(tr, te, cols, tag, folds=3):
    X = pd.concat([tr[cols], te[cols]], ignore_index=True)
    y = np.r_[np.zeros(len(tr)), np.ones(len(te))]
    X = X.apply(pd.to_numeric, errors='coerce')
    rng = np.random.default_rng(7)
    part = rng.integers(0, folds, len(X))
    auc, imp = [], None
    for k in range(folds):
        m = lgb.LGBMClassifier(**CLF)
        m.fit(X[part != k], y[part != k])
        p = m.predict_proba(X[part == k])[:, 1]
        auc.append(roc_auc_score(y[part == k], p))
        if imp is None:
            imp = pd.Series(m.booster_.feature_importance('gain'), index=cols)
    a = float(np.mean(auc))
    print(f'  {tag:24s} n_tr={len(tr):6d} n_te={len(te):6d} ncol={len(cols):4d}  '
          f'domain AUC = {a:.4f}  (0.5 = indistinguishable)')
    return a, imp.sort_values(ascending=False)


if __name__ == '__main__':
    tr, te = load()
    shared = [c for c in te.columns if c in tr.columns]
    num = [c for c in shared if pd.api.types.is_numeric_dtype(tr[c]) or
           pd.api.types.is_numeric_dtype(te[c])]
    notime = [c for c in num if c not in TIME_LIKE]
    print(f'shared {len(shared)}, numeric {len(num)}, non-time {len(notime)}')
    print(f'train {tr.forecast_kst_dtm.min()} -> {tr.forecast_kst_dtm.max()}')
    print(f'test  {te.forecast_kst_dtm.min()} -> {te.forecast_kst_dtm.max()}')

    out = {}
    print('\n--- domain-classifier two-sample test ---')
    out['ALL'], impa = domain_auc(tr, te, num, 'ALL (incl. time)')
    out['NOTIME'], impn = domain_auc(tr, te, notime, 'NOTIME (meteo only)')

    # season-matched: keep only the months our folds actually contain (Apr-Dec)
    trm = tr[tr.forecast_kst_dtm.dt.month.between(4, 12)]
    tem = te[te.forecast_kst_dtm.dt.month.between(4, 12)]
    out['NOTIME_SEASONMATCHED'], _ = domain_auc(trm, tem, notime, 'NOTIME season-matched')

    # our actual selection window (dev-2023 Q2-Q4) vs the graded period
    trd = tr[(tr.forecast_kst_dtm >= '2023-04-01 01:00') & (tr.forecast_kst_dtm <= '2024-01-01')]
    out['DEVFOLDS_vs_TEST'], impd = domain_auc(trd, te, notime, 'dev-2023 folds vs test')

    print('\n--- what drives the separation (top gain, meteo-only) ---')
    for c, v in impn.head(15).items():
        a_tr, a_te = tr[c].mean(), te[c].mean()
        print(f'  {c:52s} train_mean={a_tr:12.4f}  test_mean={a_te:12.4f}  '
              f'ratio={a_te/a_tr if a_tr else float("nan"):7.3f}')

    print('\n--- implied importance weights w(x)=p_te/p_tr on the meteo-only model ---')
    X = pd.concat([tr[notime], te[notime]], ignore_index=True).apply(pd.to_numeric, errors='coerce')
    y = np.r_[np.zeros(len(tr)), np.ones(len(te))]
    m = lgb.LGBMClassifier(**CLF); m.fit(X, y)
    p = np.clip(m.predict_proba(X[:len(tr)])[:, 1], 1e-4, 1 - 1e-4)
    w = (p / (1 - p)) * (len(tr) / len(te))
    q = np.quantile(w, [.01, .1, .25, .5, .75, .9, .99])
    print(f'  quantiles 1/10/25/50/75/90/99: {np.round(q,3)}')
    ess = w.sum() ** 2 / (w ** 2).sum()
    print(f'  effective sample size under weighting: {ess:.0f} / {len(tr)} ({ess/len(tr):.1%})')
    out['weight_quantiles'] = [float(x) for x in q]
    out['ess_fraction'] = float(ess / len(tr))
    json.dump(out, open(N + 'S14-N1_covariate_shift.json', 'w'), indent=1, default=str)
    np.save(N + 'S14-N1_train_weights.npy', w)
