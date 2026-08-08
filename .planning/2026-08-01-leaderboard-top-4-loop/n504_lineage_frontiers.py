
"""N504 — M115 계보의 달성 가능 프론티어 (적합 0회). N503 의 계보 일반화 검정.

N503 은 배포 계보(M269_PROBE) 에서 1-NMAE 상한 0.86153 을 구했다. 그러나 현재 로컬 챔피언은
`M115_XGBOOST`(1-NMAE 0.859040 / FICR 0.417780)로 **두 성분 모두 더 높다**. 계보가 다르면
프론티어도 다르므로 N503 의 결론이 일반화되는지 확인한다.

`*-policies.parquet` 에 63 개 정책 예측이 이미 저장되어 있어 적합이 필요 없다.
락박스 미접근 / 적합 0회 / 제출 없음.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from baram.evaluation.official import evaluate_official  # noqa: E402
from baram.constants import CAPACITIES_KWH  # noqa: E402

SRC = ROOT / "artifacts/backtests/metric-aligned-probe"
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
OFF_NMAE, OFF_FICR = 0.001080, 0.012027
LB = {"rank20": (0.87991, 0.43952), "rank10": (0.87790, 0.44488), "rank1": (0.87964, 0.46767)}
MODELS = ("M115_XGBOOST", "M113_LGBM_DART", "M129_GROUP_FINETUNE", "M102_TOP100")


def frontier_for(model):
    frames = []
    for f in FOLDS:
        p = SRC / f"{model}-{f}-policies.parquet"
        if not p.exists():
            return None, f"missing {p.name}"
        frames.append(pd.read_parquet(p))
    df = pd.concat(frames, ignore_index=True)
    pol_cols = [c for c in df.columns if c.startswith("T")]
    base = df[["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]]
    out = {}
    for c in pol_cols:
        d = base.copy(); d["prediction_kwh"] = df[c].to_numpy(float)
        s = evaluate_official(d, CAPACITIES_KWH)
        out[c] = (float(s.one_minus_nmae), float(s.ficr), float(s.total))
    return out, None


def pareto(P):
    keep = []
    for i, p in enumerate(P):
        if not np.any((P[:, 0] >= p[0]) & (P[:, 1] >= p[1]) & ((P[:, 0] > p[0]) | (P[:, 1] > p[1]))):
            keep.append(i)
    return P[sorted(keep, key=lambda i: P[i, 0])]


def main() -> int:
    summary = {}
    for model in MODELS:
        pts, err = frontier_for(model)
        if pts is None:
            print(f"{model:22s} SKIP ({err})"); continue
        arr = np.array([[a, b] for a, b, _ in pts.values()])
        onl = arr + np.array([OFF_NMAE, OFF_FICR])
        F = pareto(onl)
        best_pol = max(pts, key=lambda k: pts[k][2])
        max_nmae = float(onl[:, 0].max()); max_total = float((0.5 * onl[:, 0] + 0.5 * onl[:, 1]).max())
        summary[model] = dict(policies=len(pts), best_local_policy=best_pol,
                              best_local=pts[best_pol], online_equiv_max_nmae=max_nmae,
                              online_equiv_max_total=max_total,
                              frontier=[[float(a), float(b)] for a, b in F])
        print(f"\n=== {model} ===  정책 {len(pts)}개")
        print(f"  로컬 최적 {best_pol}: 1-NMAE {pts[best_pol][0]:.6f} / FICR {pts[best_pol][1]:.6f} / Total {pts[best_pol][2]:.6f}")
        print(f"  온라인등가 1-NMAE 상한 {max_nmae:.6f}   Total 상한 {max_total:.6f}")
        for name, (n, fi) in LB.items():
            reach = F[F[:, 0] >= n]
            if len(reach):
                slack = float(reach[:, 1].max()) - fi
                print(f"    {name:8s} ({n:.5f},{fi:.5f})  안쪽여부 {'O' if slack >= 0 else 'X'}  slack {slack:+.5f}")
            else:
                print(f"    {name:8s} ({n:.5f},{fi:.5f})  **NMAE 도달 불가** (상한 {max_nmae:.5f}, 차 {n-max_nmae:+.5f})")

    best_model = max(summary, key=lambda m: summary[m]["online_equiv_max_total"])
    print(f"\n>>> 전 계보 중 온라인등가 Total 상한 최대: {best_model} = {summary[best_model]['online_equiv_max_total']:.6f}")
    print(f">>> 목표 0.66 까지 {0.66 - summary[best_model]['online_equiv_max_total']:+.6f}")
    (ROOT / "reports/n504_lineage_frontiers_receipt.json").write_text(json.dumps(dict(
        node="N504_LINEAGE_FRONTIERS", offsets=dict(nmae=OFF_NMAE, ficr=OFF_FICR),
        leaderboard=LB, summary=summary, best_model=best_model,
        gap_to_066=0.66 - summary[best_model]["online_equiv_max_total"],
        model_fits=0, lockbox_reopened=False, dacon_upload=False, external_actions=[]),
        indent=1, ensure_ascii=False))
    print("영수증 -> reports/n504_lineage_frontiers_receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
