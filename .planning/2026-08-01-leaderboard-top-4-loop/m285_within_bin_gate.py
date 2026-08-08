
"""M285 — M284(구간 내부 적분) 배포 자격: 동결 월별 게이트 판정 (적합 0회).

## 사전확약 (실행 전 동결)
게이트 `M270_MONTHLY_GATE_v1_frozen_2026-08-04` 를 **수정 없이** 적용한다.
부모 = 배포 정책 `T0.5_G1.5` K=1. 후보 = fold-외 선택 정책 K=20.

- V1  부모가 0.628337 재현
- G1  sign-test p <= 0.10
- G2  월별 델타 중앙값 > 0
- G3  블록 부트스트랩 q05 > 0
- G4  최악 월 >= -0.010

**배포 자격은 4 조건 전부 통과일 때만 부여한다.** 하나라도 실패하면 M284 는 닫는다.
(참조: `C1N90` 은 재구성 표면에서 +0.001890 을 얻고도 이 게이트에 5/9 월로 기각됐다)

락박스 미접근 / 적합 0회 / 제출 없음.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "src"))
from m270_gate import GATE_VERSION, evaluate_gate  # noqa: E402
from baram.evaluation.official import evaluate_official  # noqa: E402
from baram.constants import CAPACITIES_KWH  # noqa: E402

exec(open(HERE / "m284_within_bin_deployed.py").read().split("def main()")[0].replace("if __name__", "#"))

POL_K20 = {"dev-2023-Q2": (0.6, 1.25), "dev-2023-Q3": (0.5, 0.5), "dev-2023-Q4": (0.6, 0.35)}
POL_K1 = {f: (0.5, 1.5) for f in FOLDS}


def build(st, cache, pol, K):
    parts = []
    for f in FOLDS:
        T, G = pol[f]
        d = frame_for(st[f], T, G, K, cache)
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    out["month"] = out["forecast_kst_dtm"].dt.to_period("M").astype(str)
    return out


def main() -> int:
    st, cache = load(), {}
    parent = build(st, cache, POL_K1, 1)
    cand = build(st, cache, POL_K20, 20)
    ps = evaluate_official(parent.drop(columns=["month"]), CAPACITIES_KWH)
    cs = evaluate_official(cand.drop(columns=["month"]), CAPACITIES_KWH)
    v1 = abs(float(ps.total) - 0.628337) < 1e-5
    print(f"V1 부모(배포 T0.5_G1.5, K=1) -> {ps.total:.6f} : {v1}")
    print(f"   후보(fold-외 K=20)         -> {cs.total:.6f}  (델타 {cs.total-ps.total:+.6f})")
    print(f"게이트 버전: {GATE_VERSION}\n")

    res = evaluate_gate(cand, parent)
    for name, ok in res.conditions.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n게이트 통과: {res.passed}")
    verdict = "M284_DEPLOYMENT_ELIGIBLE" if res.passed else "M284_GATE_REJECTED"
    print(f"판정: {verdict}")

    ev = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in res.evidence.items()
          if not isinstance(v, (list, dict, pd.DataFrame))}
    (ROOT / "reports/m285_within_bin_gate_receipt.json").write_text(json.dumps(dict(
        node="M285_WITHIN_BIN_GATE", gate_version=GATE_VERSION, v1_reproduction=bool(v1),
        parent_total=float(ps.total), candidate_total=float(cs.total),
        delta=float(cs.total - ps.total), conditions={k: bool(v) for k, v in res.conditions.items()},
        evidence=ev, passed=bool(res.passed), verdict=verdict,
        model_fits=0, lockbox_reopened=False, dacon_upload=False, external_actions=[]),
        indent=1, ensure_ascii=False, default=str))
    print("영수증 -> reports/m285_within_bin_gate_receipt.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
