"""M271 P4 사이클 57b — C57 산출물의 재판독: 조건부 산포의 분산법칙.

새 실험이 아니다. `m271_cycle57_ficr_ceiling_receipt.json` 에 이미 저장된 구간표를
다시 읽는다. 학습·수집·2024 행·lockbox 없음. 순수 파생이므로 C57 의 `artifact_hash`
가 입력이고, 이 노드는 그 위의 **읽기**다.

C57 은 평균 천장(0.7488)만 판정했고 구간별 구조는 리포트 표로만 남겼다. 그 표를
출력수준으로 다시 묶으면 사이클 55 가 세운 "급경사가 병목" 그림이 한 번 더 무너진다.

    산포는 급경사에서 최대가 아니라 **출력수준을 따라 단조로** 커진다.

**① 방법 리서치**

  분산이 평균의 함수인 구조를 다루는 표준은 **분산함수 추정**이다.

  - Carroll & Ruppert(1988) 의 이분산 회귀에서 표준 모수형은 `Var = sigma^2 * mu^(2*theta)`.
    양변에 로그를 씌우면 `log sd = log sigma + theta * log mu` — **log-log 기울기가
    theta** 다. 이것이 여기서 재는 양이다.
  - theta 의 해석은 확립돼 있다. `theta=1` 순수 곱셈(상대오차 일정), `theta=0.5`
    분산 ∝ 평균(집계·계수 잡음), `theta=0` 순수 덧셈.
  - Tweedie 족(Jorgensen 1987)이 같은 지수를 분산멱 `p` 로 모수화한다 — `p=2` 감마
    (곱셈), `p=1` 포아송, `p=0` 정규. `theta = p/2` 로 대응한다.
  - **채택**: 발전량 가중 log-log 회귀로 theta 를 그룹별 추정. 구간 표본이 얇으면
    기울기가 흔들리므로 `rows >= MIN_ROWS` 로 자르고 그 사실을 남긴다.

**② 사양 동결**

  입력   C57 receipt 의 `per_group[g]["bins"]` (구간별 mean_power / sigma_resid /
         gen_weight / rows). 재계산 없음.
  절차   `rows >= 200` 구간만. `log(sigma_resid) ~ log(mean_power)` 를 gen_weight
         가중 최소제곱으로 적합. 기울기 theta 와 가중 r 를 보고.
  대역   C57 리포트와 같은 y 경계 (0.10, 0.25, 0.45, 0.70, 1.10) 로 질량가중 요약.

  사전확약:
    H1  theta 가 세 그룹 모두 **0 과 유의하게 다르다** (덧셈잡음 기각).
        판정: 가중 |r| >= 0.90 이고 theta >= 0.3.
    H2  theta 가 세 그룹 모두 **1 미만** (순수 곱셈도 기각).
    H3  그룹3 의 theta 가 최대. C57 이 g3 고출력대 unit/4 0.364 로 최악을 쟀으므로
        g3 의 수준의존이 가장 가팔라야 앞뒤가 맞는다.
    H4  변동계수 sigma/mu 가 그룹3 에서 최대.

  H1·H2 가 함께 참이면 잡음은 덧셈도 곱셈도 아닌 **중간**이고, 그러면 결정층의
  전역 상수 T 는 어느 수준에서도 맞을 수 없다. 그것이 C60 의 근거다.

**이 노드는 후보가 아니다.** 점수를 내지 않는다. 진단·근거 기록 전용.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
SOURCE = REPORTS / "m271_cycle57_ficr_ceiling_receipt.json"
REPORT_MD = REPORTS / "m271_cycle57b_variance_law.md"
RECEIPT = REPORTS / "m271_cycle57b_variance_law_receipt.json"

NODE_ID = "C1N57B_VARIANCE_LAW"
LANE = "L8"
PARENT_NODE = "C1N57_FICR_CEILING"
MIN_ROWS = 200
Y_EDGES = (0.0, 0.10, 0.25, 0.45, 0.70, 1.10)
LABELS = ("(0.00,0.10]", "(0.10,0.25]", "(0.25,0.45]", "(0.45,0.70]", "(0.70,1.10]")
R_FLOOR = 0.90
THETA_FLOOR = 0.3


def weighted_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """가중 최소제곱 기울기·절편과 가중 상관계수."""
    w = w / w.sum()
    mx, my = float(np.sum(w * x)), float(np.sum(w * y))
    vx = float(np.sum(w * (x - mx) ** 2))
    vy = float(np.sum(w * (y - my) ** 2))
    cxy = float(np.sum(w * (x - mx) * (y - my)))
    slope = cxy / vx if vx > 0 else float("nan")
    r = cxy / np.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")
    return float(slope), float(my - slope * mx), float(r)


def main() -> int:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    result = source["result"]
    source_digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:16]

    per_group: dict[str, Any] = {}
    for group in ("1", "2", "3"):
        block = result["per_group"][group]
        kept = [b for b in block["bins"] if b["rows"] >= MIN_ROWS]
        mu = np.array([b["mean_power"] for b in kept], dtype="float64")
        sd = np.array([b["sigma_resid"] for b in kept], dtype="float64")
        w = np.array([b["gen_weight"] for b in kept], dtype="float64")
        theta, intercept, r = weighted_fit(np.log(mu), np.log(sd), w)
        cv = sd / np.clip(mu, 1e-9, None)
        cv_mean = float(np.average(cv, weights=w))

        bands = []
        for i, label in enumerate(LABELS):
            sel = [b for b in block["bins"] if Y_EDGES[i] < b["mean_power"] <= Y_EDGES[i + 1]]
            if not sel:
                continue
            bw = np.array([b["gen_weight"] for b in sel], dtype="float64")
            bands.append({
                "band": label,
                "mass": float(bw.sum()),
                "unit_over_4": float(
                    np.average([b["unit_mean"] for b in sel], weights=bw) / 4.0
                ),
                "hit6": float(np.average([b["hit6"] for b in sel], weights=bw)),
                "sigma": float(np.average([b["sigma_resid"] for b in sel], weights=bw)),
                "rows": int(sum(b["rows"] for b in sel)),
            })
        per_group[group] = {
            "kept_bins": len(kept),
            "theta": theta,
            "intercept": intercept,
            "weighted_r": r,
            "cv_mean": cv_mean,
            "ficr_ceiling": block["ficr_ceiling"],
            "bands": bands,
        }

    thetas = {g: per_group[g]["theta"] for g in per_group}
    rs = {g: per_group[g]["weighted_r"] for g in per_group}
    cvs = {g: per_group[g]["cv_mean"] for g in per_group}

    h1 = bool(all(abs(rs[g]) >= R_FLOOR and thetas[g] >= THETA_FLOOR for g in thetas))
    h2 = bool(all(thetas[g] < 1.0 for g in thetas))
    h3 = bool(max(thetas, key=lambda g: thetas[g]) == "3")
    h4 = bool(max(cvs, key=lambda g: cvs[g]) == "3")

    if h1 and h2:
        verdict = "VARIANCE_LAW_INTERMEDIATE_GLOBAL_TEMPERATURE_CANNOT_FIT"
    elif h1:
        verdict = "VARIANCE_LAW_MULTIPLICATIVE"
    else:
        verdict = "VARIANCE_LAW_NOT_ESTABLISHED"

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "candidate": False,
        "derivation_only": True,
        "source_receipt": SOURCE.name,
        "source_digest": source_digest,
        "method": "VARIANCE_FUNCTION (Carroll & Ruppert 1988; Jorgensen 1987 Tweedie)",
        "min_rows": MIN_ROWS,
        "per_group": per_group,
        "theta": thetas,
        "weighted_r": rs,
        "cv_mean": cvs,
        "hypotheses": {
            "H1_not_additive": h1,
            "H2_not_pure_multiplicative": h2,
            "H3_group3_steepest": h3,
            "H4_group3_highest_cv": h4,
        },
        "verdict": verdict,
        "no_training": True,
        "no_collection": True,
        "uses_2024_rows": False,
        "lockbox_used": False,
        "dacon_upload": False,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    payload["digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    RECEIPT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [
        "# M271 P4 사이클 57b — 조건부 산포의 분산법칙",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}` / **파생 전용 (후보 아님)**",
        "",
        f"입력: `{SOURCE.name}` digest `{source_digest}`. 재계산·학습·수집 없음.",
        "",
        "## 1. 분산법칙  `log sd = log sigma + theta * log mu`",
        "",
        "| 그룹 | 구간수 | **theta** | 가중 r | 변동계수 | FICR 천장 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for g, blk in per_group.items():
        lines.append(
            f"| {g} | {blk['kept_bins']} | **{blk['theta']:+.3f}** | "
            f"{blk['weighted_r']:+.3f} | {blk['cv_mean']:.3f} | {blk['ficr_ceiling']:.4f} |"
        )
    lines += [
        "",
        "`theta=1` 순수 곱셈 / `theta=0.5` 분산 ∝ 평균 / `theta=0` 순수 덧셈 "
        "(Tweedie 분산멱 `p = 2*theta`).",
        "",
        "## 2. 대역별 (질량가중)",
        "",
        "| 그룹 | 대역 | 질량 | 천장 단위/4 | 적중률 | 산포 | 행 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for g, blk in per_group.items():
        for band in blk["bands"]:
            lines.append(
                f"| {g} | {band['band']} | {band['mass']:.3f} | {band['unit_over_4']:.3f} | "
                f"{band['hit6']:.3f} | {band['sigma']:.4f} | {band['rows']} |"
            )
    lines += [
        "",
        "## 3. 사전확약",
        "",
        f"- H1 덧셈잡음 기각 (|r| >= {R_FLOOR}, theta >= {THETA_FLOOR}) -> **{h1}**",
        f"- H2 순수 곱셈 기각 (theta < 1) -> **{h2}**",
        f"- H3 그룹3 theta 최대 -> **{h3}**",
        f"- H4 그룹3 변동계수 최대 -> **{h4}**",
        "",
        "## 4. 판정",
        "",
        f"**{verdict}**",
        "",
        "산포는 급경사 고유가 아니라 출력수준의 매끈한 멱함수다. 결정층은 fold 당 전역 T "
        "하나를 쓰므로(C44) 수준에 걸쳐 변하는 산포/계단폭 비를 맞출 수 없다. "
        "`C1N60_LEVEL_TEMPERATURE` 가 그 자유도를 검정한다.",
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== 완료 ===")
    for g, blk in per_group.items():
        print(f"[C57b] group {g}: theta {blk['theta']:+.3f} / 가중 r {blk['weighted_r']:+.3f} "
              f"/ CV {blk['cv_mean']:.3f} / 구간 {blk['kept_bins']}")
    print(f"[C57b] H1 {h1} / H2 {h2} / H3 {h3} / H4 {h4}")
    print(f"[C57b] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
