
"""M275 — 밴드 캘리브레이션 지표 정의 및 분포 후보 8종 재평가 (적합 0회, 파생 전용).

문제: `V2_DISTRIBUTION` 게이트는 q10-q90 피복률로 캘리브레이션을 판정한다.
그러나 본 대회 목적함수는 계단형 정산보상이며, 필요한 것은 "80% 중심구간의 피복"이 아니라
"±6% 밴드 적중확률"의 캘리브레이션이다 (Sahoo et al., NeurIPS 2021, threshold calibration).

정의: 각 행에서 점예측 yhat=q50 일 때
  예측 밴드적중확률  P_hat = F(yhat + 0.06C) - F(yhat - 0.06C)      (7분위수 선형보간 F)
  실제 밴드적중       hit   = 1[|yhat - y| <= 0.06C]
평가는 공식 적격모집단(y >= 0.10C)에서만 수행한다.
락박스 미접근 / 제출물 없음 / 외부행위 없음.
"""
from __future__ import annotations
import json, sys, glob
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from baram.constants import CAPACITIES_KWH  # noqa: E402
from baram.evaluation.official import evaluate_official  # noqa: E402

LEVELS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
BAND_HIT, BAND_PARTIAL, ELIGIBLE = 0.06, 0.08, 0.10
SRC = ROOT / "artifacts/backtests/distribution-v2/baram-v2-20260801-01"


def band_metrics(path: Path):
    long = pd.read_parquet(path)
    wide = long.pivot_table(index=["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh", "fold_id"],
                            columns="quantile", values="prediction_kwh").reset_index()
    qcols = [c for c in wide.columns if isinstance(c, float)]
    qcols = sorted(qcols)
    Q = wide[qcols].to_numpy(float)
    Q = np.maximum.accumulate(Q, axis=1)            # 교차 복구(이미 0이어야 함)
    cap = wide["group_id"].map(CAPACITIES_KWH).to_numpy(float)
    y = wide["actual_kwh"].to_numpy(float)
    yhat = Q[:, list(qcols).index(0.50)]
    lo, hi = yhat - BAND_HIT * cap, yhat + BAND_HIT * cap

    def F(x):                                        # 행별 선형보간 CDF (양끝 clamp)
        return np.array([np.interp(x[i], Q[i], LEVELS) for i in range(len(x))])

    p_hat = np.clip(F(hi) - F(lo), 0.0, 1.0)
    hit = (np.abs(yhat - y) <= BAND_HIT * cap).astype(float)
    valid = y >= ELIGIBLE * cap
    p, h = p_hat[valid], hit[valid]

    # ECE (동일폭 10구간)
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = 0.0
    for b in range(10):
        m = bins == b
        if m.sum(): ece += m.sum() / len(p) * abs(p[m].mean() - h[m].mean())
    brier = float(np.mean((p - h) ** 2))

    # q10-q90 피복 (교차확인)
    q10, q90 = Q[:, list(qcols).index(0.10)], Q[:, list(qcols).index(0.90)]
    cover = float(np.mean((y >= q10) & (y <= q90)))

    # 공식 q50 점수 (교차확인)
    frame = wide.loc[valid | ~valid, ["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
    frame["prediction_kwh"] = yhat
    sc = evaluate_official(frame, CAPACITIES_KWH)

    return dict(n_valid=int(valid.sum()), pred_band=float(p.mean()), actual_band=float(h.mean()),
                bias=float(p.mean() - h.mean()), ece=float(ece), brier=brier,
                q10_q90_cover=cover, q50_total=float(sc.total), q50_ficr=float(sc.ficr))


def main() -> int:
    rows = {}
    for f in sorted(SRC.glob("*-oof.parquet")):
        cid = f.name.replace("-oof.parquet", "")
        rows[cid] = band_metrics(f)
        print(f"  {cid} 완료")
    t = pd.DataFrame(rows).T
    t.index.name = "candidate"
    print("\n=== 밴드 캘리브레이션 재평가 (적격행만) ===")
    cols = ["n_valid", "pred_band", "actual_band", "bias", "ece", "brier", "q10_q90_cover", "q50_total"]
    print(t[cols].to_string(float_format=lambda x: f"{x:.6f}"))

    from scipy.stats import spearmanr
    r1 = spearmanr(t["ece"].astype(float), t["q50_total"].astype(float))
    r2 = spearmanr((t["q10_q90_cover"].astype(float) - 0.80).abs(), t["q50_total"].astype(float))
    print(f"\n밴드ECE      vs q50_total  rho={r1.statistic:+.3f} p={r1.pvalue:.3f}  (정합하면 음수)")
    print(f"|q10q90-0.8| vs q50_total  rho={r2.statistic:+.3f} p={r2.pvalue:.3f}  (기존 게이트 전제)")
    best_band = t["ece"].astype(float).idxmin(); best_off = t["q50_total"].astype(float).idxmax()
    print(f"\n밴드캘리브 최선: {best_band}   공식지표 최선: {best_off}")
    print(f"D1(채택) 밴드ECE={t.loc['D1_LGBM_SHARED_BASE','ece']:.6f}  D2(기각) 밴드ECE={t.loc['D2_LGBM_SHARED_LEAF63','ece']:.6f}")

    out = {"node": "M275_BAND_CALIBRATION", "definition": "P_hat=F(q50+0.06C)-F(q50-0.06C) vs 1[|q50-y|<=0.06C], 적격행만",
           "records": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in rows.items()},
           "spearman_band_ece_vs_total": [float(r1.statistic), float(r1.pvalue)],
           "spearman_q1090dev_vs_total": [float(r2.statistic), float(r2.pvalue)]}
    (ROOT / "reports/m275_band_calibration_receipt.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("\n영수증 -> reports/m275_band_calibration_receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
