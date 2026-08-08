
"""M279 — M272 정책격자 확장이 배포 계보(M269_PROBE / M102_TOP100)에 전이되는가.

## 배경
M272: 재구성 표면에서 정책격자가 T<=1.2, G<=2.0 로 잘려 있었고, 확장하니 내부 최적 T2_G4 (+0.010483 vs 배포규칙).
M276: 결정온도는 캘리브레이션 수리다 -> 최적 T 는 **모델별로 다르다**. 전이 보장 없음.
m269 가 배포 표면에서 쓴 격자: T in (0.4..1.2), G in (0..2.0). **동일한 절단**이 걸려 있다.

## 사전확약 (실행 전 동결)
- V1  배포 정책 T0.5_G1.5 가 pooled 0.628605 를 재현한다 (허용 5e-4)
- H1  확장격자의 pooled 최적이 기존격자(T<=1.2,G<=2.0) 최적보다 높다
- H2  **fold-외** 선택 정책이 배포(0.628605)를 넘는다        <- 배포 가능성 판정
- H3  fold-외 선택 정책이 3/3 폴드에서 배포 대비 개선        <- 견고성
- H4  배포 표면 최적 T 가 재구성 표면 최적(2.0)과 다르다     <- M276 의 '모델별' 예측

**반증**: H2 가 거짓이면 M272 이득은 재구성 표면 국한이며 배포 개선 경로가 아니다.

락박스 미접근 / 모델 적합 0회 / 제출물 없음 / 외부행위 없음.
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
DEPLOYED_T, DEPLOYED_G, DEPLOYED_TOTAL = 0.5, 1.5, 0.628605
OLD_T = (0.4, 0.5, 0.6, 0.75, 0.85, 1.0, 1.2)
OLD_G = (0.0, 0.2, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
NEW_T = OLD_T + (1.4, 1.6, 1.8, 2.0, 2.5, 3.0)
NEW_G = OLD_G + (2.5, 3.0, 4.0, 6.0)


def load():
    st = {}
    for f in FOLDS:
        z = np.load(SRC / f"M269_PROBE-{f.replace('dev-2023-','dev-2023-')}-probability.npz", allow_pickle=True)
        C = z["centers"]
        cap = np.array([CAPACITIES_KWH[int(g)] for g in z["group_id"]], dtype=float)
        st[f] = dict(prob=z["probability"], centers=C, group=z["group_id"].astype(int), cap=cap,
                     actual=z["actual_kwh"].astype(float), dtm=pd.to_datetime(z["forecast_kst_dtm"]))
    return st


def make_mats(C):
    err = np.abs(ACTIONS[:, None] - C[None, :])
    units = np.select([err <= 0.06, err <= 0.08], [4.0, 3.0], default=0.0)
    return err, (C[None, :] * units).T


def score(store, T, G, folds=FOLDS):
    parts = []
    for f in folds:
        c = store[f]
        err, settle_m = make_mats(c["centers"])
        rate = c["actual"] / c["cap"]
        norms = {int(g): float(np.mean(rate[c["group"] == g])) for g in np.unique(c["group"])}
        cal = np.power(np.clip(c["prob"], 1e-12, None), 1.0 / T); cal /= cal.sum(1, keepdims=True)
        settle, base = cal @ settle_m, -(cal @ err.T)
        yhat = np.empty(len(cal))
        for gid in np.unique(c["group"]):
            m = c["group"] == gid
            yhat[m] = ACTIONS[np.argmax(base[m] + G * settle[m] / (4.0 * norms[int(gid)]), axis=1)]
        parts.append(pd.DataFrame(dict(forecast_id=[f"{f}-{i}" for i in range(len(cal))],
                                       forecast_kst_dtm=c["dtm"], group_id=c["group"],
                                       actual_kwh=c["actual"], prediction_kwh=yhat * c["cap"])))
    s = evaluate_official(pd.concat(parts, ignore_index=True), CAPACITIES_KWH)
    return float(s.total), float(s.one_minus_nmae), float(s.ficr)


def main() -> int:
    st = load()
    dep = score(st, DEPLOYED_T, DEPLOYED_G)
    v1 = abs(dep[0] - DEPLOYED_TOTAL) < 5e-4
    print(f"V1 배포 재현 T0.5_G1.5 -> {dep[0]:.6f} (기대 {DEPLOYED_TOTAL}) : {v1}")
    if not v1:
        print("V1 실패 — 표면이 배포와 다르다. 이후 결론은 이 표면 한정으로만 읽어야 한다.")

    grid = {(T, G): score(st, T, G)[0] for T in NEW_T for G in NEW_G}
    old_best = max(((T, G) for T in OLD_T for G in OLD_G), key=lambda k: grid[k])
    new_best = max(grid, key=grid.get)
    print(f"기존격자 최적 T{old_best[0]:g}_G{old_best[1]:g} = {grid[old_best]:.6f}")
    print(f"확장격자 최적 T{new_best[0]:g}_G{new_best[1]:g} = {grid[new_best]:.6f}  (차 {grid[new_best]-grid[old_best]:+.6f})")

    # fold-외 선택
    fo, fold_tot = {}, {}
    for f in FOLDS:
        others = [o for o in FOLDS if o != f]
        best = max(grid, key=lambda k: score(st, k[0], k[1], folds=others)[0])
        fo[f] = best
        fold_tot[f] = (score(st, best[0], best[1], folds=[f])[0], score(st, DEPLOYED_T, DEPLOYED_G, folds=[f])[0])
    parts = []
    for f in FOLDS:
        T, G = fo[f]
        parts.append(score(st, T, G, folds=[f]))
    # pooled with per-fold policies
    frames = []
    for f in FOLDS:
        T, G = fo[f]
        c = st[f]; err, settle_m = make_mats(c["centers"]); rate = c["actual"] / c["cap"]
        norms = {int(g): float(np.mean(rate[c["group"] == g])) for g in np.unique(c["group"])}
        cal = np.power(np.clip(c["prob"], 1e-12, None), 1.0 / T); cal /= cal.sum(1, keepdims=True)
        settle, base = cal @ settle_m, -(cal @ err.T)
        yhat = np.empty(len(cal))
        for gid in np.unique(c["group"]):
            m = c["group"] == gid
            yhat[m] = ACTIONS[np.argmax(base[m] + G * settle[m] / (4.0 * norms[int(gid)]), axis=1)]
        frames.append(pd.DataFrame(dict(forecast_id=[f"{f}-{i}" for i in range(len(cal))],
                                        forecast_kst_dtm=c["dtm"], group_id=c["group"],
                                        actual_kwh=c["actual"], prediction_kwh=yhat * c["cap"])))
    s_fo = evaluate_official(pd.concat(frames, ignore_index=True), CAPACITIES_KWH)
    fo_total = float(s_fo.total)

    h2 = fo_total > dep[0]
    h3 = all(fold_tot[f][0] > fold_tot[f][1] for f in FOLDS)
    h4 = new_best[0] != 2.0
    print(f"\nfold-외 선택 정책 {[(f, fo[f]) for f in FOLDS]}")
    print(f"fold-외 pooled = {fo_total:.6f}  vs 배포 {dep[0]:.6f}  ({fo_total-dep[0]:+.6f})")
    for f in FOLDS:
        print(f"   {f}: 선택 {fold_tot[f][0]:.6f} vs 배포 {fold_tot[f][1]:.6f}  ({fold_tot[f][0]-fold_tot[f][1]:+.6f})")
    print(f"\nH1 확장>기존              -> {grid[new_best] > grid[old_best]}")
    print(f"H2 fold-외 > 배포 [핵심]  -> {h2}")
    print(f"H3 3/3 폴드 개선          -> {h3}")
    print(f"H4 배포최적T != 재구성 2.0 -> {h4}  (배포최적 T={new_best[0]})")
    verdict = "EXTENSION_TRANSFERS_TO_DEPLOYED" if h2 and h3 else "EXTENSION_DOES_NOT_TRANSFER"
    print(f"\n판정: {verdict}")

    (ROOT / "reports/m279_deployed_grid_transfer_receipt.json").write_text(json.dumps(dict(
        node="M279_DEPLOYED_GRID_TRANSFER", v1_reproduction=v1, deployed=dep,
        old_best=[*old_best, grid[old_best]], new_best=[*new_best, grid[new_best]],
        fold_outside={f: [*fo[f], *fold_tot[f]] for f in FOLDS}, fold_outside_pooled=fo_total,
        hypotheses=dict(H1=bool(grid[new_best] > grid[old_best]), H2=bool(h2), H3=bool(h3), H4=bool(h4)),
        verdict=verdict, model_fits=0, lockbox_reopened=False, dacon_upload=False, external_actions=[],
        grid={f"T{T:g}_G{G:g}": v for (T, G), v in grid.items()}), indent=1, ensure_ascii=False))
    print("영수증 -> reports/m279_deployed_grid_transfer_receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
