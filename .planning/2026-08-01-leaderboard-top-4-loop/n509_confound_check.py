
"""N509 — N508 의 교락 점검: 실측 조건부 대 예측 조건부 (적합 0회).

N508 은 행을 **실측 출력**으로 구간화해 C1N57 의 **풍속** 구간 천장과 비교했다.
결과(중간 구간 적중률이 천장의 1/4)가 조건부 편향의 산물인지 확인한다.
통제: 동일 분석을 **예측 수준**(결과에 의존하지 않음)으로 재수행.
패턴이 유지되면 실재, 사라지면 N508 결론을 철회한다.

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
EDGES = [0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 1.10]


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
    exp_level = cal @ C          # 예측 기대수준 (결과 비의존)
    return yh, rate, exp_level


def table(df, key, label):
    df = df.copy()
    df["bin"] = pd.cut(df[key], EDGES, right=False)
    g = df.groupby("bin", observed=True).apply(
        lambda s: pd.Series({
            "n": len(s),
            "gen_weight": s.rate.sum(),
            "hit6": s.hit6.mean(),
            "hit8": s.hit8.mean(),
            "mean_abs_err": s.err.mean(),
        }), include_groups=False)
    g["gen_weight"] = g.gen_weight / g.gen_weight.sum()
    print(f"\n=== {label} 기준 구간화 ===")
    print(f"{'구간':>14} {'n':>6} {'발전량가중':>10} {'6%적중':>8} {'8%적중':>8} {'평균오차':>9}")
    for idx, r in g.iterrows():
        print(f"{str(idx):>14} {int(r.n):6d} {r.gen_weight:10.3f} {r.hit6:8.3f} {r.hit8:8.3f} {r.mean_abs_err:9.4f}")
    return g


def main() -> int:
    st = load()
    parts = []
    for f in FOLDS:
        c = st[f]
        yh, rate, lvl = predict(c, *DEP)
        parts.append(pd.DataFrame(dict(group=c["group"], rate=rate, pred=yh, level=lvl)))
    df = pd.concat(parts, ignore_index=True)
    df = df[np.isfinite(df.rate) & (df.rate >= 0.10)].copy()
    df["err"] = (df.pred - df.rate).abs()
    df["hit6"] = (df.err <= 0.06).astype(float)
    df["hit8"] = (df.err <= 0.08).astype(float)

    g_actual = table(df, "rate", "실측 출력 (N508 방식, 결과 조건부)")
    g_pred = table(df, "level", "예측 기대수준 (결과 비의존, 통제)")

    print("\n=== 패턴 비교 ===")
    a_mid = g_actual.loc[[i for i in g_actual.index if 0.30 <= i.left < 0.70], "hit6"].mean()
    a_hi = g_actual.loc[[i for i in g_actual.index if i.left >= 0.80], "hit6"].mean()
    p_mid = g_pred.loc[[i for i in g_pred.index if 0.30 <= i.left < 0.70], "hit6"].mean()
    p_hi = g_pred.loc[[i for i in g_pred.index if i.left >= 0.80], "hit6"].mean()
    print(f"  실측조건부: 중간(0.35~0.65) {a_mid:.3f}  고출력(>=0.80) {a_hi:.3f}  차 {a_hi-a_mid:+.3f}")
    print(f"  예측조건부: 중간(0.35~0.65) {p_mid:.3f}  고출력(>=0.80) {p_hi:.3f}  차 {p_hi-p_mid:+.3f}")
    survives = (p_hi - p_mid) > 0.05
    print(f"\n패턴 유지 (예측조건부에서도 중간이 열세) -> {survives}")
    print("  유지 -> N508 결론 실재 / 소멸 -> N508 은 결과 조건부 산물, 철회")

    (ROOT / "reports/n509_confound_check_receipt.json").write_text(json.dumps(dict(
        node="N509_CONFOUND_CHECK",
        actual_conditioned={str(k): {c: float(v) for c, v in r.items()} for k, r in g_actual.iterrows()},
        pred_conditioned={str(k): {c: float(v) for c, v in r.items()} for k, r in g_pred.iterrows()},
        actual_mid_hi=[float(a_mid), float(a_hi)], pred_mid_hi=[float(p_mid), float(p_hi)],
        pattern_survives=bool(survives),
        model_fits=0, lockbox_reopened=False, dacon_upload=False, external_actions=[]),
        indent=1, ensure_ascii=False))
    print("\n영수증 -> reports/n509_confound_check_receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
