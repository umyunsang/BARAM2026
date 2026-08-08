
"""N511 — 필요 예측오차 감소율 직접 측정 (적합 0회, 요구치 산정 전용).

폐쇄 결론 전체가 "필요 예측오차 감소 약 11%" 에 걸려 있다. 그 수치는 m270_revised_verdicts
(k=0.890) 인용이며, 본 세션에서 인용 사슬의 오류를 여러 번 발견했으므로 **직접 잰다**.

장치: 예측오차를 실측 쪽으로 k 배 수축 (pred' = actual + k*(pred - actual)).
실측을 쓰므로 **요구치 산정 전용**이며 달성 가능 결과가 아니다(프로젝트가 승인한 관행).

목표: 온라인 0.66 의 로컬 등가 = 0.66 - 0.006554(M261 오프셋) = 0.653446
       및 2 차평가 진출선 0.65788 의 로컬 등가 = 0.651326

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
DEP = (0.5, 1.5)
OFFSET = 0.006554
TARGETS = {"online_0.66": 0.66 - OFFSET, "rank30_0.65788": 0.65788 - OFFSET}


def load():
    st = {}
    for f in FOLDS:
        z = np.load(SRC / f"M269_PROBE-{f}-probability.npz", allow_pickle=True)
        cap = np.array([CAPACITIES_KWH[int(g)] for g in z["group_id"]], float)
        st[f] = dict(prob=z["probability"], centers=z["centers"], group=z["group_id"].astype(int),
                     cap=cap, actual=z["actual_kwh"].astype(float), dtm=pd.to_datetime(z["forecast_kst_dtm"]))
    return st


def base_frames(st, T, G):
    out = []
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
        out.append(pd.DataFrame(dict(forecast_id=[f"{f}-{i}" for i in range(len(cal))],
                                     forecast_kst_dtm=c["dtm"], group_id=c["group"],
                                     actual_kwh=c["actual"], prediction_kwh=yh * c["cap"])))
    return pd.concat(out, ignore_index=True)


def score_k(df, k):
    d = df.copy()
    d["prediction_kwh"] = d["actual_kwh"] + k * (d["prediction_kwh"] - d["actual_kwh"])
    s = evaluate_official(d, CAPACITIES_KWH)
    return float(s.total), float(s.one_minus_nmae), float(s.ficr)


def main() -> int:
    st = load()
    df = base_frames(st, *DEP)
    base = score_k(df, 1.0)
    print(f"기준(k=1) Total {base[0]:.6f} / 1-NMAE {base[1]:.6f} / FICR {base[2]:.6f}")
    print(f"\n{'k':>6} {'감소율':>7} {'Total':>9} {'1-NMAE':>9} {'FICR':>9}")
    curve = {}
    for k in [1.0, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50]:
        v = score_k(df, k); curve[k] = v
        print(f"{k:6.2f} {1-k:7.1%} {v[0]:9.6f} {v[1]:9.6f} {v[2]:9.6f}")

    ks = np.array(sorted(curve)); tot = np.array([curve[k][0] for k in ks])
    print()
    out = {}
    for name, tgt in TARGETS.items():
        if tot.max() >= tgt:
            kstar = float(np.interp(tgt, tot[::-1], ks[::-1]))
            red = 1 - kstar
            out[name] = dict(target_local=tgt, k=kstar, reduction=red)
            print(f"{name}: 로컬 목표 {tgt:.6f} -> k* = {kstar:.4f}  필요 오차감소 **{red:.1%}**")
        else:
            out[name] = dict(target_local=tgt, k=None, reduction=None)
            print(f"{name}: 로컬 목표 {tgt:.6f} -> 격자 범위 내 미도달")

    print(f"\n참조: 다중소스 결합으로 얻을 수 있는 풍속 sigma 감소 상한 = 4.6% (C1N52)")
    print(f"      단 이는 풍속 sigma 이고 위 수치는 예측오차이므로 직접 비교는 아니다")
    (ROOT / "reports/n511_required_reduction_receipt.json").write_text(json.dumps(dict(
        node="N511_REQUIRED_REDUCTION", policy=list(DEP), offset=OFFSET,
        base=list(base), curve={str(k): list(v) for k, v in curve.items()}, targets=out,
        device="oracle k-shrink toward actual; requirement sizing only, not an achievable result",
        model_fits=0, lockbox_reopened=False, dacon_upload=False, external_actions=[]),
        indent=1, ensure_ascii=False))
    print("영수증 -> reports/n511_required_reduction_receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
