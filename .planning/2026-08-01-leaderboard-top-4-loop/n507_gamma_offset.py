
"""N507 — 오프셋의 gamma 의존성 부분 검정 + M266 사전확약 동결 (적합 0회).

## 사전확약 (M266 업로드 결과가 나오기 전에 동결)
가정: 온라인 오프셋은 gamma 에 불변 (1-NMAE +0.001080, FICR +0.012027).
그 가정 하 예측: **M266(T0.6_G0.5) 의 온라인 Total < 0.6365274 (M261)**.
빗나가면 N503/N504 의 상한 추정(0.637653)을 전면 재작성한다.

## 본 실험 (업로드 없이 가능한 부분)
로컬 표면에서 gamma 궤적을 재고, 온라인 앵커 2 개가 함의하는 궤적과 비교한다.
- 로컬 dTotal/dgamma 와 온라인 dTotal/dgamma 가 크게 다르면 오프셋의 gamma 의존성이 시사된다
- 비슷하면 gamma 불변 가정이 보강된다

락박스 미접근 / 적합 0회 / 제출 없음 / 업로드 없음.
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
ONLINE = {"pre_repo": (0.87305, 0.37426, 0.62366), "M261": (0.857885, 0.415169, 0.6365274)}


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
    return float(s.one_minus_nmae), float(s.ficr), float(s.total)


def main() -> int:
    st = load()
    print("로컬 gamma 궤적 (T=0.5 고정, M269_PROBE)")
    print(f"{'gamma':>7} {'1-NMAE':>9} {'FICR':>9} {'Total':>9}")
    traj = {}
    for G in (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
        traj[G] = comp(st, 0.5, G)
        print(f"{G:7.2f} {traj[G][0]:9.6f} {traj[G][1]:9.6f} {traj[G][2]:9.6f}")

    l0, l15 = traj[0.0], traj[1.5]
    dn_local, df_local = l0[0] - l15[0], l0[1] - l15[1]
    a, b = ONLINE["pre_repo"], ONLINE["M261"]
    dn_onl, df_onl = a[0] - b[0], a[1] - b[1]
    print(f"\n로컬  gamma 1.5 -> 0 :  d(1-NMAE) {dn_local:+.6f}   dFICR {df_local:+.6f}")
    print(f"온라인 M261 -> pre_repo:  d(1-NMAE) {dn_onl:+.6f}   dFICR {df_onl:+.6f}")
    print(f"비율                    :  NMAE {dn_onl/dn_local if dn_local else float('nan'):.2f}x   FICR {df_onl/df_local if df_local else float('nan'):.2f}x")
    print("\n주의: pre_repo 는 우리 모델의 gamma=0 점이 아니라 **다른 모델**이다.")
    print("      따라서 이 비율은 gamma 민감도 차이와 모델 차이가 교락돼 있다 — 시사적일 뿐이다.")

    g05 = comp(st, 0.6, 0.5)
    pred = 0.5 * (g05[0] + 0.001080) + 0.5 * (g05[1] + 0.012027)
    print(f"\nM266 사전확약: T0.6_G0.5 로컬 = 1-NMAE {g05[0]:.6f} / FICR {g05[1]:.6f} / Total {g05[2]:.6f}")
    print(f"  gamma 불변 오프셋 적용 예측 온라인 Total = {pred:.6f}")
    print(f"  M261 실측 0.6365274 대비 {pred-0.6365274:+.6f}  -> 예측: {'하회' if pred < 0.6365274 else '상회'}")
    print("  (주: M266 은 다른 아키텍처(dart_group3_augmentation)이므로 이 로컬값은 대리치다)")

    (ROOT / "reports/n507_gamma_offset_predeclaration.json").write_text(json.dumps(dict(
        node="N507_GAMMA_OFFSET", local_trajectory={str(k): list(v) for k, v in traj.items()},
        online_anchors=ONLINE, local_delta=[dn_local, df_local], online_delta=[dn_onl, df_onl],
        m266_proxy_local=list(g05), m266_predicted_online_total=pred,
        predeclaration="M266 online Total < 0.6365274 under gamma-invariant offset",
        falsifier="if M266 exceeds 0.6365274, N503/N504 ceiling estimates are rewritten",
        model_fits=0, lockbox_reopened=False, dacon_upload=False, external_actions=[]),
        indent=1, ensure_ascii=False))
    print("\n영수증 -> reports/n507_gamma_offset_predeclaration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
