
"""M277 — 조건부 밴드 캘리브레이션이 전역 온도를 이기는가 (적합 0회, 파생 전용).

## 배경
M276 판정: `DECISION_TEMPERATURE_IS_CALIBRATION` (T_cal=1.731 vs T*=2.00, 차이 0.269).
결정층 온도의 대부분은 결정이론이 아니라 **상류 캘리브레이션 오차를 수리**하는 데 쓰이고 있다.
전역 스칼라 온도는 **평균 캘리브레이션만** 고칠 수 있고 조건부(그룹·수준별) 오차는 못 고친다.

## 사전확약 (실행 전 동결)
- 팔 A: 전역 온도 1개를 fold-외에서 밴드bias=0 이 되도록 적합
- 팔 B: 그룹별 온도 3개
- 팔 C: 그룹 x 예측수준 3구간 = 온도 9개
- 팔 D(참조): 전역 온도를 fold-외 Total 최대로 적합 (캘리브레이션이 아닌 직접 최적화)

- H1  B > A                      (그룹별 조건부가 전역을 이긴다)
- H2  C > B                      (수준별까지 나누면 더 낫다)
- H3  최선팔 - A > 0.001013      (C1N90 검출문턱 초과)
- H4  전 팔이 fold 3/3 에서 A 대비 개선 (견고성)

**반증**: H1 이 거짓이면 "조건부 캘리브레이션이 남은 레버" 가설은 기각되고,
캘리브레이션 축은 전역 온도로 이미 소진된 것으로 판정한다.

온도는 **fold-외**(해당 fold 를 제외한 나머지)에서 적합한다 — 프로젝트의 `A0_corrected_control` 정의와 동일.
락박스 미접근 / 모델 적합 0회 / 제출물 없음.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from baram.evaluation.official import evaluate_official  # noqa: E402
from baram.constants import CAPACITIES_KWH  # noqa: E402

N_CLASS, CLASS_WIDTH = 46, 0.02
CENTERS = (np.arange(N_CLASS) + 0.5) * CLASS_WIDTH
ACTIONS = np.round(np.arange(0.075, 1.076, 0.0025), 6)
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
SURFACE = ROOT / "artifacts/cache/m271_decision_surface/994ae6dff5796332daf21a6f"
G, BAND, ELIGIBLE = 4.0, 0.06, 0.10
TGRID = np.round(np.arange(0.6, 4.01, 0.05), 2)
LEVEL_EDGES = (0.25, 0.55)          # 예측수준 3구간

_ERR = np.abs(ACTIONS[:, None] - CENTERS[None, :])
_UNITS = np.select([_ERR <= 0.06, _ERR <= 0.08], [4.0, 3.0], default=0.0)
_SETTLE_M = (CENTERS[None, :] * _UNITS).T
_INBAND = _ERR <= BAND


def load():
    st = {}
    for f in FOLDS:
        z = np.load(SURFACE / f"{f}__arrays.npz")
        m = pd.read_parquet(SURFACE / f"{f}__meta.parquet")
        rate = m["actual_kwh"].to_numpy(float) / z["capacity"]
        ok = np.isfinite(rate)
        st[f] = dict(prob=z["probability"][ok], group=z["group"][ok], cap=z["capacity"][ok],
                     meta=m[ok].reset_index(drop=True), rate=rate[ok],
                     level=(z["probability"][ok] @ CENTERS))     # 예측 발전율 수준
    return st


def decide(prob, group, T, norms):
    cal = np.power(np.clip(prob, 1e-12, None), 1.0 / np.asarray(T).reshape(-1, 1))
    cal /= cal.sum(axis=1, keepdims=True)
    settle, base = cal @ _SETTLE_M, -(cal @ _ERR.T)
    idx = np.empty(len(cal), dtype=int)
    for gid in np.unique(group):
        m = group == gid
        idx[m] = np.argmax(base[m] + G * settle[m] / (4.0 * norms[int(gid)]), axis=1)
    return ACTIONS[idx], cal, idx


def cell_ids(cell, arm):
    g, lv = cell["group"], cell["level"]
    if arm == "A": return np.zeros(len(g), dtype=int)
    if arm in ("B", "D"): return g.astype(int)
    b = np.digitize(lv, LEVEL_EDGES)
    return g.astype(int) * 10 + b


def bias_or_total(cells, T_of_cell, arm, want):
    """fold-외 적합용 목적: want='bias' -> |밴드bias|, want='total' -> -Total"""
    frames, ps, hs = [], [], []
    for cell in cells:
        cid = cell_ids(cell, arm)
        Tv = np.array([T_of_cell[c] for c in cid])
        norms = {int(g): float(np.mean(cell["rate"][cell["group"] == g])) for g in np.unique(cell["group"])}
        yhat, cal, idx = decide(cell["prob"], cell["group"], Tv, norms)
        elig = cell["rate"] >= ELIGIBLE
        ps.append((cal * _INBAND[idx]).sum(1)[elig])
        hs.append((np.abs(yhat - cell["rate"]) <= BAND).astype(float)[elig])
        d = cell["meta"][["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
        d["prediction_kwh"] = yhat * cell["cap"]
        frames.append(d)
    if want == "bias":
        p, h = np.concatenate(ps), np.concatenate(hs)
        return abs(float(p.mean() - h.mean()))
    return -float(evaluate_official(pd.concat(frames, ignore_index=True), CAPACITIES_KWH).total)


def fit(cells, arm):
    """셀별로 독립 1D 탐색. A/B/C 는 밴드bias=0, D 는 Total 최대."""
    want = "total" if arm == "D" else "bias"
    ids = np.unique(np.concatenate([cell_ids(c, arm) for c in cells]))
    T_of = {int(i): 2.0 for i in ids}
    for _ in range(2):                                   # 좌표하강 2회
        for i in ids:
            best, bT = np.inf, T_of[int(i)]
            for T in TGRID:
                trial = dict(T_of); trial[int(i)] = float(T)
                v = bias_or_total(cells, trial, arm, want)
                if v < best: best, bT = v, float(T)
            T_of[int(i)] = bT
    return T_of


def score(cells, T_of, arm):
    frames = []
    for cell in cells:
        cid = cell_ids(cell, arm)
        Tv = np.array([T_of.get(int(c), 2.0) for c in cid])
        norms = {int(g): float(np.mean(cell["rate"][cell["group"] == g])) for g in np.unique(cell["group"])}
        yhat, _, _ = decide(cell["prob"], cell["group"], Tv, norms)
        d = cell["meta"][["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
        d["prediction_kwh"] = yhat * cell["cap"]
        frames.append(d)
    s = evaluate_official(pd.concat(frames, ignore_index=True), CAPACITIES_KWH)
    return float(s.total), float(s.one_minus_nmae), float(s.ficr)


def main() -> int:
    st = load()
    out, per_fold = {}, {}
    for arm in ("A", "B", "C", "D"):
        tot_frames, fold_tot = [], {}
        for f in FOLDS:
            others = [st[o] for o in FOLDS if o != f]
            T_of = fit(others, arm)                       # fold-외 적합
            t, n, fi = score([st[f]], T_of, arm)
            fold_tot[f] = t
            d = st[f]["meta"][["forecast_id", "forecast_kst_dtm", "group_id", "actual_kwh"]].copy()
            cid = cell_ids(st[f], arm)
            Tv = np.array([T_of.get(int(c), 2.0) for c in cid])
            norms = {int(g): float(np.mean(st[f]["rate"][st[f]["group"] == g])) for g in np.unique(st[f]["group"])}
            yh, _, _ = decide(st[f]["prob"], st[f]["group"], Tv, norms)
            d["prediction_kwh"] = yh * st[f]["cap"]; tot_frames.append(d)
        s = evaluate_official(pd.concat(tot_frames, ignore_index=True), CAPACITIES_KWH)
        out[arm] = dict(pooled=float(s.total), one_minus_nmae=float(s.one_minus_nmae), ficr=float(s.ficr), **fold_tot)
        per_fold[arm] = fold_tot
        print(f"  팔 {arm}: pooled={s.total:.6f}  " + "  ".join(f"{k[-2:]}={v:.6f}" for k, v in fold_tot.items()))

    A, B, C, D = (out[x]["pooled"] for x in "ABCD")
    best = max("ABCD", key=lambda x: out[x]["pooled"])
    h4 = all(out[x][f] > out["A"][f] for x in ("B", "C") for f in FOLDS)
    print(f"\nH1 B > A                  -> {B > A}  ({B - A:+.6f})")
    print(f"H2 C > B                  -> {C > B}  ({C - B:+.6f})")
    print(f"H3 최선-A > 0.001013      -> {out[best]['pooled'] - A > 0.001013}  (최선={best}, {out[best]['pooled'] - A:+.6f})")
    print(f"H4 B,C 가 3/3 fold 개선   -> {h4}")
    print(f"참조 팔 D(Total 직접최적화) = {D:.6f}  vs 캘리브레이션 팔 A = {A:.6f}  ({D - A:+.6f})")
    verdict = "CONDITIONAL_CALIBRATION_ADDS" if (B > A and out[best]["pooled"] - A > 0.001013) \
        else "CONDITIONAL_CALIBRATION_EXHAUSTED"
    print(f"\n판정: {verdict}")
    (ROOT / "reports/m277_conditional_calibration_receipt.json").write_text(json.dumps(
        dict(node="M277_CONDITIONAL_CALIBRATION", arms=out, verdict=verdict,
             predeclared=dict(H1="B>A", H2="C>B", H3="best-A>0.001013", H4="B,C improve 3/3 folds")),
        indent=1, ensure_ascii=False))
    print("영수증 -> reports/m277_conditional_calibration_receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
