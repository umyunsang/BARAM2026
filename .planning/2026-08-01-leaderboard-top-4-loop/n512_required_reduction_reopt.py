
"""N512 — 필요 오차감소율 재측정: 각 k 에서 정책 재최적화 (적합 0회).

N511 은 배포정책(T0.5_G1.5) 예측에 k-축소를 적용했다. 그러나 그 정책은 밴드를 쫓느라
MAE 를 이미 3.7% 부풀린다(N507: gamma 1.5 대 0 에서 1-NMAE 0.854394 대 0.859562).
오차가 줄면 최적 정책도 이동하므로, **각 k 에서 (T,G) 를 재최적화**해야 요구치가 정확하다.

장치: 확률면의 각 구간 중심을 실측 쪽으로 k 배 수축한 뒤 정책 격자를 다시 훑는다.
실측을 쓰므로 **요구치 산정 전용**이며 달성 가능 결과가 아니다.

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
OFFSET = 0.006554
TEMPS = (0.4, 0.5, 0.6, 0.75, 0.85, 1.0, 1.2)
GAMMAS = (0.0, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
TARGETS = {"online_0.66": 0.66 - OFFSET, "rank30": 0.65788 - OFFSET}


def load():
    st = {}
    for f in FOLDS:
        z = np.load(SRC / f"M269_PROBE-{f}-probability.npz", allow_pickle=True)
        cap = np.array([CAPACITIES_KWH[int(g)] for g in z["group_id"]], float)
        st[f] = dict(prob=z["probability"], centers=z["centers"], group=z["group_id"].astype(int),
                     cap=cap, actual=z["actual_kwh"].astype(float), dtm=pd.to_datetime(z["forecast_kst_dtm"]))
    return st


def score(st, T, G, k):
    parts = []
    for f in FOLDS:
        c = st[f]
        rate = c["actual"] / c["cap"]
        # 구간 중심을 실측 쪽으로 k 배 수축 (행별)
        C = c["centers"][None, :] if k == 1.0 else rate[:, None] + k * (c["centers"][None, :] - rate[:, None])
        err = np.abs(ACTIONS[None, :, None] - C[:, None, :])
        units = np.select([err <= 0.06, err <= 0.08], [4.0, 3.0], default=0.0)
        cal = np.power(np.clip(c["prob"], 1e-12, None), 1.0 / T); cal /= cal.sum(1, keepdims=True)
        base = -(cal[:, None, :] * err).sum(2)
        settle = (cal[:, None, :] * C[:, None, :] * units).sum(2)
        norms = {int(g): float(np.mean(rate[c["group"] == g])) for g in np.unique(c["group"])}
        yh = np.empty(len(cal))
        for gid in np.unique(c["group"]):
            m = c["group"] == gid
            yh[m] = ACTIONS[np.argmax(base[m] + G * settle[m] / (4.0 * norms[int(gid)]), axis=1)]
        parts.append(pd.DataFrame(dict(forecast_id=[f"{f}-{i}" for i in range(len(cal))],
                                       forecast_kst_dtm=c["dtm"], group_id=c["group"],
                                       actual_kwh=c["actual"], prediction_kwh=yh * c["cap"])))
    s = evaluate_official(pd.concat(parts, ignore_index=True), CAPACITIES_KWH)
    return float(s.total), float(s.one_minus_nmae), float(s.ficr)


def main() -> int:
    st = load()
    print(f"{'k':>6} {'감소':>6} {'최적정책':>12} {'Total':>9} {'1-NMAE':>9} {'FICR':>9}")
    curve = {}
    for k in [1.0, 0.96, 0.92, 0.90, 0.88, 0.85, 0.82, 0.80]:
        best, bp = None, None
        for T in TEMPS:
            for G in GAMMAS:
                v = score(st, T, G, k)
                if best is None or v[0] > best[0]: best, bp = v, (T, G)
        curve[k] = (best, bp)
        print(f"{k:6.2f} {1-k:6.1%} {f'T{bp[0]:g}_G{bp[1]:g}':>12} {best[0]:9.6f} {best[1]:9.6f} {best[2]:9.6f}")

    ks = np.array(sorted(curve)); tot = np.array([curve[k][0][0] for k in ks])
    print()
    out = {}
    for name, tgt in TARGETS.items():
        if tot.max() >= tgt:
            kstar = float(np.interp(tgt, tot[::-1], ks[::-1])); red = 1 - kstar
            out[name] = dict(target=tgt, k=kstar, reduction=red)
            print(f"{name}: 로컬목표 {tgt:.6f} -> k* {kstar:.4f}  **필요 감소 {red:.1%}**")
        else:
            out[name] = dict(target=tgt, k=None, reduction=None)
            print(f"{name}: 격자 범위 내 미도달")
    print(f"\nN511(정책 고정) 대비: online_0.66 요구가 10.8% -> {out['online_0.66']['reduction']:.1%}")
    (ROOT / "reports/n512_required_reduction_reopt_receipt.json").write_text(json.dumps(dict(
        node="N512_REQUIRED_REDUCTION_REOPT", offset=OFFSET,
        curve={str(k): dict(total=v[0][0], one_minus_nmae=v[0][1], ficr=v[0][2], policy=list(v[1]))
               for k, v in curve.items()},
        targets=out, n511_comparison=0.108,
        device="oracle k-shrink of bin centres toward actual, policy re-optimised at each k",
        model_fits=0, lockbox_reopened=False, dacon_upload=False, external_actions=[]),
        indent=1, ensure_ascii=False))
    print("영수증 -> reports/n512_required_reduction_reopt_receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
