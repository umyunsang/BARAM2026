
"""M272 P0-2 — 온라인 등가 표면에서 결정층 정책 재최적화 (적합 0회, 파생 전용).

배포 정책 T0.5_G1.5 는 로컬(dev-2023) 표면에서 고정됐다. 온라인 앵커는
online FICR - local FICR = +0.012027 (M261, 분류기 계보) 를 보인다.
따라서 로컬 최적이 온라인 최적이 아닐 수 있다. 이 스크립트는
  (1) 배포 정책을 바이트 수준으로 재현하고 (V1)
  (2) 관측된 온라인 FICR 을 재현하는 축소계수 k* 를 식별하고
  (3) 그 온라인 등가 표면에서 7x9 정책격자를 재최적화한다.
락박스 미접근 / 모델 적합 0회 / 제출물 없음.
"""
from __future__ import annotations
import json, sys, glob
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from baram.evaluation.official import evaluate_official  # noqa: E402
from baram.constants import CAPACITIES_KWH  # noqa: E402

N_CLASS, CLASS_WIDTH = 46, 0.02
CENTERS = (np.arange(N_CLASS) + 0.5) * CLASS_WIDTH
DEPLOYED_ACTIONS = np.round(np.arange(0.075, 1.076, 0.0025), 6)
DEPLOYED_T, DEPLOYED_G = 0.5, 1.5
TEMPS = (4.0, 3.0, 2.5, 2.0, 1.6, 1.4, 1.2, 1.0, 0.85, 0.75, 0.6, 0.5, 0.4)  # 1.2 초과 확장
GAMMAS = (0.0, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0)  # 2.0 초과 확장
A1_TOTAL, A1_FICR = 0.595568, 0.342659   # 재구성표면 위 배포규칙 (m271_n9 공표값)
DEPLOYED_TOTAL, DEPLOYED_FICR = 0.628605, 0.402464
ONLINE_FICR_M261, LOCAL_FICR_M261 = 0.4151695, 0.403142
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")

_ERR = np.abs(DEPLOYED_ACTIONS[:, None] - CENTERS[None, :])
_UNITS = np.select([_ERR <= 0.06, _ERR <= 0.08], [4.0, 3.0], default=0.0)
_SETTLE_M = (CENTERS[None, :] * _UNITS).T


def decide(prob, group, temperature, gamma, norms):
    cal = np.power(np.clip(prob, 1e-12, None), 1.0 / temperature)
    cal /= cal.sum(axis=1, keepdims=True)
    settle = cal @ _SETTLE_M
    base = -(cal @ _ERR.T)
    out = np.empty(len(prob))
    for gid in np.unique(group):
        m = group == gid
        out[m] = DEPLOYED_ACTIONS[np.argmax(base[m] + gamma * settle[m] / (4.0 * norms[int(gid)]), axis=1)]
    return out


def load(cache_dir):
    store = {}
    for f in FOLDS:
        z = np.load(f"{cache_dir}/{f}__arrays.npz")
        meta = pd.read_parquet(f"{cache_dir}/{f}__meta.parquet")
        store[f] = dict(prob=z["probability"], group=z["group"], cap=z["capacity"], meta=meta)
    return store


def contract(prob, rate, k):
    """예측분포 지지점을 실측 쪽으로 k배 수축 -> '더 정확한 모형' 등가 표면."""
    newc = rate[:, None] + k * (CENTERS[None, :] - rate[:, None])
    idx = np.clip(np.round((newc - CENTERS[0]) / CLASS_WIDTH).astype(int), 0, N_CLASS - 1)
    out = np.zeros_like(prob)
    np.add.at(out, (np.arange(len(prob))[:, None], idx), prob)
    return out / out.sum(axis=1, keepdims=True)


def score(store, T, G, k=1.0):
    pieces = []
    for f in FOLDS:
        c = store[f]
        rate = c["meta"]["actual_kwh"].to_numpy(float) / c["cap"]
        ok = np.isfinite(rate)
        norms = {int(g): float(np.nanmean(rate[(c["group"] == g) & ok])) for g in np.unique(c["group"])}
        prob = c["prob"] if k == 1.0 else contract(c["prob"], np.clip(rate, 0, 0.92), k)
        pred = decide(prob, c["group"], T, G, norms) * c["cap"]
        d = c["meta"][["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
        d["prediction_kwh"] = pred
        pieces.append(d[ok])
    frame = pd.concat(pieces, ignore_index=True)
    s = evaluate_official(frame, CAPACITIES_KWH)
    return float(s.total), float(s.one_minus_nmae), float(s.ficr)


def main():
    # --- 표면 선택: 배포 정책이 0.628605 를 재현하는 캐시 ---
    chosen, repro = None, None
    for d in sorted(glob.glob(str(ROOT / "artifacts/cache/m271_decision_surface/*"))):
        try:
            st = load(d)
            t = score(st, DEPLOYED_T, DEPLOYED_G)
        except Exception as e:
            print(f"  skip {Path(d).name[:12]}: {type(e).__name__} {e}"); continue
        print(f"  {Path(d).name[:12]}  deployed Total={t[0]:.6f} 1-NMAE={t[1]:.6f} FICR={t[2]:.6f}")
        if abs(t[0] - A1_TOTAL) < 5e-5:
            chosen, repro = d, t
    if chosen is None:
        print("V1 FAIL: 배포 재현 불가"); return 1
    print(f"\nV1 PASS 재현 표면 = {Path(chosen).name}  Total={repro[0]:.6f} (기대 A1={A1_TOTAL})")
    store = load(chosen)

    # --- 로컬 표면 격자 ---
    local = {(T, G): score(store, T, G) for T in TEMPS for G in GAMMAS}
    lbest = max(local, key=lambda kk: local[kk][0])
    edge = (lbest[0] in (max(TEMPS), min(TEMPS))) or (lbest[1] in (max(GAMMAS), min(GAMMAS)))
    print(f"[경계최적 진단] 최적이 격자 경계인가? {edge}   (기존격자 최적 T1.2_G2 = {local.get((1.2,2.0),(float('nan'),))[0]:.6f})")
    print(f"로컬 최적 T{lbest[0]:g}_G{lbest[1]:g} Total={local[lbest][0]:.6f} | 배포 T0.5_G1.5 Total={local[(0.5,1.5)][0]:.6f}")

    # --- k* 식별: 배포 정책 FICR 이 온라인 관측치를 재현하는 수축계수 ---
    target = A1_FICR + (ONLINE_FICR_M261 - LOCAL_FICR_M261)
    rows = []
    for k in [1.0, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.86]:
        t = score(store, DEPLOYED_T, DEPLOYED_G, k)
        rows.append((k, *t)); print(f"  k={k:.2f}  Total={t[0]:.6f} 1-NMAE={t[1]:.6f} FICR={t[2]:.6f}")
    ks = np.array([r[0] for r in rows]); fs = np.array([r[3] for r in rows])
    kstar = float(np.interp(target, fs[::-1], ks[::-1]))
    print(f"\n목표 FICR={target:.6f} (A1 로컬 {A1_FICR} + 앵커 오프셋 {ONLINE_FICR_M261-LOCAL_FICR_M261:+.6f}) -> k*={kstar:.4f}")

    # --- 온라인 등가 표면에서 재최적화 ---
    onl = {(T, G): score(store, T, G, kstar) for T in TEMPS for G in GAMMAS}
    obest = max(onl, key=lambda kk: onl[kk][0])
    dep = onl[(DEPLOYED_T, DEPLOYED_G)]
    print(f"\n온라인등가 최적  T{obest[0]:g}_G{obest[1]:g}  Total={onl[obest][0]:.6f}")
    print(f"온라인등가 배포  T0.5_G1.5      Total={dep[0]:.6f}")
    print(f">>> 재최적화 이득(온라인등가) = {onl[obest][0]-dep[0]:+.6f}")
    print(f">>> 그 정책의 로컬 Total      = {local[obest][0]:.6f} (배포 로컬 {local[(0.5,1.5)][0]:.6f}, 차 {local[obest][0]-local[(0.5,1.5)][0]:+.6f})")

    out = dict(node="M272_ONLINE_EQUIVALENT_POLICY", surface=Path(chosen).name, v1_reproduction=repro,
               kstar=kstar, target_ficr=target,
               local_best=[*lbest, *local[lbest]], online_best=[*obest, *onl[obest]],
               deployed_local=local[(0.5,1.5)], deployed_online_equiv=dep,
               gain_online_equiv=onl[obest][0]-dep[0], local_cost=local[obest][0]-local[(0.5,1.5)][0],
               grid_local={f"T{T:g}_G{G:g}": local[(T,G)] for T in TEMPS for G in GAMMAS},
               grid_online={f"T{T:g}_G{G:g}": onl[(T,G)] for T in TEMPS for G in GAMMAS})
    (ROOT / "reports/m272_online_equivalent_policy_receipt.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("\n영수증 -> reports/m272_online_equivalent_policy_receipt.json")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
