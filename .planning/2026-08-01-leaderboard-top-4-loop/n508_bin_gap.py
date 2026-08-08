
"""N508 — 구간별 적중률: 실측 대 물리 천장 (적합 0회, 파생 전용).

C1N57 은 **완벽한 중앙 예보** 가정 하 FICR 천장을 냈다(g1 0.7586 / g2 0.9108 / g3 0.5770,
평균 0.7488, 요구치 0.4459 를 크게 상회). 그 천장의 g1 bin 별 6% 적중률도 공표돼 있다.
따라서 (천장 적중률 - 실측 적중률) 이 **예보오차가 치르는 비용**이고, 그 분포가 개입 지점을 정한다.

락박스 미접근 / 적합 0회 / 제출 없음.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from baram.constants import CAPACITIES_KWH  # noqa: E402

SRC = ROOT / "artifacts/backtests/m269-probe"
FOLDS = ("dev-2023-Q2", "dev-2023-Q3", "dev-2023-Q4")
ACTIONS = np.round(np.arange(0.075, 1.076, 0.0025), 6)
DEP = (0.5, 1.5)
# C1N57 공표: group 1 의 풍속 bin 별 (평균출력, 발전량가중, 잔차sigma, 천장 6% 적중률)
CEILING_G1 = {
    5.0: (0.095, 0.005, 0.0224, 0.907), 6.0: (0.188, 0.028, 0.0347, 0.914),
    7.0: (0.307, 0.048, 0.0463, 0.880), 8.0: (0.458, 0.067, 0.0562, 0.796),
    9.0: (0.601, 0.077, 0.0545, 0.761), 10.0: (0.726, 0.082, 0.0679, 0.582),
    11.0: (0.812, 0.062, 0.0845, 0.476), 12.0: (0.876, 0.049, 0.0851, 0.431),
    13.0: (0.886, 0.034, 0.0753, 0.620), 14.0: (0.926, 0.024, 0.0484, 0.763),
    15.0: (0.939, 0.014, 0.0453, 0.918), 16.0: (0.938, 0.008, 0.0514, 0.846),
}


def load():
    st = {}
    for f in FOLDS:
        z = np.load(SRC / f"M269_PROBE-{f}-probability.npz", allow_pickle=True)
        cap = np.array([CAPACITIES_KWH[int(g)] for g in z["group_id"]], float)
        st[f] = dict(prob=z["probability"], centers=z["centers"], group=z["group_id"].astype(int),
                     cap=cap, actual=z["actual_kwh"].astype(float))
    return st


def predict(c, T, G):
    C = c["centers"]
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
    return yh, rate


def main() -> int:
    st = load()
    rows = []
    for f in FOLDS:
        c = st[f]
        yh, rate = predict(c, *DEP)
        rows.append(pd.DataFrame(dict(group=c["group"], rate=rate, pred=yh)))
    df = pd.concat(rows, ignore_index=True)
    df = df[np.isfinite(df.rate) & (df.rate >= 0.10)]
    df["err"] = (df.pred - df.rate).abs()
    df["hit6"] = (df.err <= 0.06).astype(float)
    df["hit8"] = (df.err <= 0.08).astype(float)
    df["unit"] = np.where(df.err <= 0.06, 4.0, np.where(df.err <= 0.08, 3.0, 0.0))

    # 실측 출력수준을 C1N57 의 bin 평균출력에 최근접 매핑
    lv = np.array(sorted(CEILING_G1)); mu = np.array([CEILING_G1[k][0] for k in lv])
    g1 = df[df.group == 1].copy()
    idx = np.abs(g1.rate.to_numpy()[:, None] - mu[None, :]).argmin(1)
    g1["bin"] = lv[idx]

    print("group 1 — 출력수준 bin 별 실측 대 물리천장")
    print(f"{'bin':>5} {'평균출력':>8} {'가중':>7} {'천장적중':>9} {'실측적중':>9} {'격차':>9} {'가중격차':>9} {'n':>6}")
    tot_gap = 0.0
    for b in lv:
        sub = g1[g1.bin == b]
        if len(sub) == 0: continue
        mu_b, w, sig, ceil = CEILING_G1[b]
        act = float(sub.hit6.mean())
        gap = ceil - act
        wg = w * gap
        tot_gap += wg
        print(f"{b:5.0f} {mu_b:8.3f} {w:7.3f} {ceil:9.3f} {act:9.3f} {gap:+9.3f} {wg:+9.4f} {len(sub):6d}")
    print(f"\n발전량가중 총 적중률 격차 (g1) = {tot_gap:+.4f}")
    print(f"→ 이 격차가 예보오차가 치르는 비용이다 (산포는 이미 천장에 반영됨)")

    print("\n전 그룹 요약")
    for g in (1, 2, 3):
        sub = df[df.group == g]
        w = sub.rate / sub.rate.sum()
        ficr = float((sub.rate * sub.unit).sum() / (sub.rate * 4.0).sum())
        print(f"  g{g}: 적격행 {len(sub):6d}  6%적중 {sub.hit6.mean():.3f}  8%적중 {sub.hit8.mean():.3f}  FICR {ficr:.4f}")

    (ROOT / "reports/n508_bin_gap_receipt.json").write_text(json.dumps(dict(
        node="N508_BIN_GAP", policy=list(DEP),
        g1_bins={str(b): dict(mu=CEILING_G1[b][0], weight=CEILING_G1[b][1],
                              sigma=CEILING_G1[b][2], ceiling_hit=CEILING_G1[b][3],
                              actual_hit=float(g1[g1.bin == b].hit6.mean()) if len(g1[g1.bin == b]) else None,
                              n=int((g1.bin == b).sum())) for b in lv},
        weighted_hit_gap_g1=tot_gap,
        model_fits=0, lockbox_reopened=False, dacon_upload=False, external_actions=[]),
        indent=1, ensure_ascii=False))
    print("\n영수증 -> reports/n508_bin_gap_receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
