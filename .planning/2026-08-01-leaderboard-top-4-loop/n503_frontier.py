
"""N503 — 달성 가능 (1-NMAE, FICR) 프론티어와 상위권 좌표의 기하 (적합 0회).

배포 계보 확률면에서 정책 (T,G) 를 전 격자로 훑어 달성 가능한 성분 조합의 **파레토 프론티어**를
구한다. 상위권 좌표가 그 프론티어의 안쪽인지 바깥쪽인지가 격차의 성격을 결정한다.

- 안쪽이면: 우리 모델로 도달 가능하며 정책 선택 문제다
- 바깥쪽이면: 정책으로는 불가능하고 모델(표현)이 바뀌어야 한다

락박스 미접근 / 적합 0회 / 제출 없음.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from baram.evaluation.official import evaluate_official  # noqa: E402
from baram.constants import CAPACITIES_KWH  # noqa: E402

SRC = ROOT / "artifacts/backtests/m269-probe"
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
ACTIONS = np.round(np.arange(0.075, 1.076, 0.0025), 6)
TEMPS = (0.3, 0.4, 0.5, 0.6, 0.75, 0.85, 1.0, 1.2, 1.5, 2.0)
GAMMAS = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 8.0)
# 온라인 앵커 오프셋 (M261, 분류기 계보)
OFF_NMAE, OFF_FICR = 0.001080, 0.012027
LB = {"rank20": (0.87991, 0.43952), "rank10": (0.87790, 0.44488), "rank1": (0.87964, 0.46767),
      "M261_online": (0.857885, 0.415169)}


def load():
    st = {}
    for f in FOLDS:
        z = np.load(SRC / f"M269_PROBE-{f}-probability.npz", allow_pickle=True)
        cap = np.array([CAPACITIES_KWH[int(g)] for g in z["group_id"]], float)
        st[f] = dict(prob=z["probability"], centers=z["centers"], group=z["group_id"].astype(int),
                     cap=cap, actual=z["actual_kwh"].astype(float), dtm=pd.to_datetime(z["forecast_kst_dtm"]))
    return st


def comp(st, T, G):
    parts = []
    for f in FOLDS:
        c = st[f]; C = c["centers"]
        err = np.abs(ACTIONS[:, None] - C[None, :])
        units = np.select([err <= 0.06, err <= 0.08], [4.0, 3.0], default=0.0)
        rate = c["actual"] / c["cap"]
        norms = {int(g): float(np.mean(rate[c["group"] == g])) for g in np.unique(c["group"])}
        cal = np.power(np.clip(c["prob"], 1e-12, None), 1.0 / T); cal /= cal.sum(1, keepdims=True)
        base, settle = -(cal @ err.T), cal @ (C[None, :] * units).T
        yh = np.empty(len(cal))
        for gid in np.unique(c["group"]):
            m = c["group"] == gid
            yh[m] = ACTIONS[np.argmax(base[m] + G * settle[m] / (4.0 * norms[int(gid)]), axis=1)]
        parts.append(pd.DataFrame(dict(forecast_id=[f"{f}-{i}" for i in range(len(cal))],
                                       forecast_kst_dtm=c["dtm"], group_id=c["group"],
                                       actual_kwh=c["actual"], prediction_kwh=yh * c["cap"])))
    s = evaluate_official(pd.concat(parts, ignore_index=True), CAPACITIES_KWH)
    return float(s.one_minus_nmae), float(s.ficr)


def main() -> int:
    st = load()
    pts = {}
    for T in TEMPS:
        for G in GAMMAS:
            pts[(T, G)] = comp(st, T, G)
    arr = np.array(list(pts.values()))
    # 온라인 등가 좌표 (앵커 오프셋 적용)
    onl = arr + np.array([OFF_NMAE, OFF_FICR])

    # 파레토 프론티어 (두 성분 모두 클수록 좋음)
    def pareto(P):
        keep = []
        for i, p in enumerate(P):
            if not np.any((P[:, 0] >= p[0]) & (P[:, 1] >= p[1]) & ((P[:, 0] > p[0]) | (P[:, 1] > p[1]))):
                keep.append(i)
        return P[sorted(keep, key=lambda i: P[i, 0])]

    F = pareto(onl)
    print(f"정책 격자 {len(pts)}개 → 온라인등가 파레토 프론티어 {len(F)}점\n")
    print(f"{'1-NMAE':>9} {'FICR':>9} {'Total':>9}")
    for a, b in F:
        print(f"{a:9.5f} {b:9.5f} {0.5*a+0.5*b:9.5f}")

    print(f"\n{'좌표':14s} {'1-NMAE':>9} {'FICR':>9} {'Total':>9}  프론티어 대비")
    out = {}
    for name, (n, fi) in LB.items():
        # 같은 1-NMAE 에서 우리가 낼 수 있는 최대 FICR
        reach = F[F[:, 0] >= n]
        best_ficr = float(reach[:, 1].max()) if len(reach) else float("nan")
        slack = best_ficr - fi if np.isfinite(best_ficr) else float("nan")
        inside = np.isfinite(best_ficr) and best_ficr >= fi
        out[name] = dict(one_minus_nmae=n, ficr=fi, total=0.5*n+0.5*fi,
                         our_max_ficr_at_that_nmae=best_ficr, slack=slack, inside=bool(inside))
        s = f"{slack:+.5f}" if np.isfinite(slack) else "도달불가(NMAE)"
        print(f"{name:14s} {n:9.5f} {fi:9.5f} {0.5*n+0.5*fi:9.5f}  {'안쪽' if inside else '바깥쪽':6s} slack {s}")

    best_total = max(0.5*a+0.5*b for a, b in onl)
    print(f"\n우리 온라인등가 최대 Total = {best_total:.6f}   목표 0.66 까지 {0.66-best_total:+.6f}")
    (ROOT / "reports/n503_frontier_receipt.json").write_text(json.dumps(dict(
        node="N503_FRONTIER", grid=len(pts), frontier=[[float(a), float(b)] for a, b in F],
        offsets=dict(nmae=OFF_NMAE, ficr=OFF_FICR), leaderboard=out,
        our_max_online_equivalent_total=best_total, gap_to_066=0.66-best_total,
        model_fits=0, lockbox_reopened=False, dacon_upload=False, external_actions=[]),
        indent=1, ensure_ascii=False))
    print("영수증 -> reports/n503_frontier_receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
