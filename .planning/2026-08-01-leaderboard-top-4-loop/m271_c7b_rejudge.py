"""M271 C7b — C7 산출물의 재판정: 올바른 기준으로 V1 을 다시 건다.

새 실험이 아니다. `m271_c7_teacher_scaleup_receipt.json` 에 이미 저장된 팔 값을 다시
읽는다. teacher 적합·표면 재구성 없음. C1N57B 가 C1N57 에 한 것과 같은 순수 파생이다.

**C7 은 자기 V1 가드로 VOID 였고 그 발화가 옳았다.** 다만 원인이 하네스가 아니라
**내가 비교 대상을 틀리게 동결한 것**이다.

    C1N58 shuffle KFold OOF (학습행)   1.0923   <- C7 의 base 와 같은 양
    C1N58 blocked                      1.6772
    C1N66 (평가 fold 의 test 행)        1.5866   <- 내가 V1 기준으로 잡은 값
    C7 base                            1.1037

C7 은 **학습행의 무작위 KFold OOF** 를 재는데 나는 **평가 fold 의 test 행** 잔차를
기준으로 걸었다. C1N58 의 shuffle 값과는 **0.0114 차이로 일치**한다 — 하네스는 멀쩡했다.

**두 값의 차(1.09 대 1.59)가 이 노드의 핵심 한계다.** C1N54 가 이미 "시간 인접 누출분
17.8~21.3%p" 로 기록했다. 무작위 KFold 는 이웃 시각이 학습에 들어가므로 잔차가 작게
나온다. 따라서 C7 이 본 개선은 **누출된 면에서의 개선**이고, 실제 평가면(시간 분할)에서
같은 크기일 보장이 없다.

**① 방법 리서치**

  새 방법 없음. C1N57B 와 같은 절차 — 저장된 산출물을 다시 읽어 판정만 교정한다.
  팔 값은 같은 표면·같은 KFold 시드·같은 행으로 **결정적**이므로 재실행해도 동일하다.
  참조 상수 하나만 바뀌었으므로 재계산이 불필요하고, 재계산하면 2.5 시간을 같은 숫자를
  다시 얻는 데 쓰게 된다.

**② 사양 동결**

  입력   `m271_c7_teacher_scaleup_receipt.json` 의 `arms` / `reductions`
  V1     `base` 가 **C1N58 shuffle 1.0923** 의 ±0.03 이내 (올바른 기준)
  C16    요구 감소율 = `MAGNITUDE_FLOOR * 남은격차 / C1N69 기울기` = 2.72%

  사전확약:
    H1  V1 이 올바른 기준에서 통과한다. 통과하면 C7 의 VOID 는 **기준 오지정**이었고
        팔 값은 유효하다는 뜻이다.
    H2  최선 팔이 C16 문턱(2.72%)을 넘는다.
    H3  `seq` 가 `base` 를 넘지 못한다 — 시계열 문맥이 teacher 에 값을 하지 않는다.
    H4  최선 팔이 세 그룹 모두에서 개선한다.
    H5  **누출면 한계를 명시한다.** 이 개선은 시간 분할에서 재확인되기 전까지
        승격 근거가 되지 못한다. 이것은 검정이 아니라 **선언**이며, 그래서 판정문에
        `PENDING_CHRONOLOGICAL_CONFIRMATION` 을 남긴다.

**진단 전용.** 후보 아님. 모델 미변경. 게이트·lockbox·외부데이터 미사용.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
SOURCE = REPORTS / "m271_c7_teacher_scaleup_receipt.json"
REPORT_MD = REPORTS / "m271_c7b_rejudge.md"
RECEIPT = REPORTS / "m271_c7b_rejudge_receipt.json"

NODE_ID = "C1N83_TEACHER_SCALEUP_REJUDGED"
LANE = "L6"
PARENT_NODE = "C1N82_TEACHER_SCALEUP"

C58_SHUFFLE_SIGMA = 1.0923
C66_TEST_SIGMA = 1.5866
V1_TOLERANCE = 0.03


def main() -> int:
    payload_in = json.loads(SOURCE.read_text(encoding="utf-8"))
    source_digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:16]

    arms = payload_in["arms"]
    reductions = payload_in["reductions"]
    required = float(payload_in["required_reduction_for_c16"])
    slope = float(payload_in["response_slope"])
    floor_total = float(payload_in["c16_floor_total"])

    base_sigma = float(arms["base"]["overall"])
    v1 = bool(abs(base_sigma - C58_SHUFFLE_SIGMA) <= V1_TOLERANCE)

    best_arm = max((a for a in arms if a != "base"), key=lambda a: reductions[a])
    best_reduction = float(reductions[best_arm])
    implied_total = best_reduction * slope

    h1 = v1
    h2 = bool(best_reduction >= required)
    h3 = bool(float(arms["seq"]["overall"]) >= base_sigma)
    h4 = bool(all(
        float(arms[best_arm][f"g{g}"]) < float(arms["base"][f"g{g}"]) for g in (1, 2, 3)
    ))

    if not v1:
        verdict = "REJUDGE_STILL_FAILS_HARNESS_SUSPECT"
    elif h2:
        verdict = "TEACHER_CAPACITY_CLEARS_GATE_PENDING_CHRONOLOGICAL_CONFIRMATION"
    else:
        verdict = "TEACHER_AXIS_BELOW_MAGNITUDE_GATE"

    payload: dict[str, Any] = {
        "node_id": NODE_ID,
        "lane": LANE,
        "parent": PARENT_NODE,
        "candidate": False,
        "derivation_only": True,
        "source_receipt": SOURCE.name,
        "source_digest": source_digest,
        "corrects": {
            "node": "C1N82_TEACHER_SCALEUP",
            "what": "V1 참조 상수",
            "was": f"C1N66 test 행 잔차 {C66_TEST_SIGMA}",
            "should_be": f"C1N58 shuffle KFold OOF {C58_SHUFFLE_SIGMA}",
            "why": (
                "C7 은 학습행의 무작위 KFold OOF 를 재는데 나는 평가 fold 의 test 행 "
                "잔차를 기준으로 걸었다. 다른 양이다. C1N58 의 shuffle 값과는 0.0114 "
                "차이로 일치하므로 하네스는 멀쩡했다."
            ),
        },
        "arms": arms,
        "reductions": reductions,
        "base_sigma": base_sigma,
        "best_arm": best_arm,
        "best_reduction": best_reduction,
        "implied_total_gain": implied_total,
        "required_reduction_for_c16": required,
        "c16_floor_total": floor_total,
        "sequence_columns_added": payload_in.get("sequence_columns_added"),
        "checks": {"V1_base_matches_c58_shuffle": v1, "V1_gap": abs(base_sigma - C58_SHUFFLE_SIGMA)},
        "hypotheses": {
            "H1_v1_passes_on_correct_reference": h1,
            "H2_clears_magnitude_gate": h2,
            "H3_sequence_does_not_help": h3,
            "H4_helps_all_groups": h4,
        },
        "limitation": (
            "**누출면에서 잰 값이다.** 무작위 KFold 는 이웃 시각이 학습에 들어가므로 "
            f"잔차가 작다(학습행 {C58_SHUFFLE_SIGMA} 대 test 행 {C66_TEST_SIGMA}, "
            "C1N54 가 잰 누출분 17.8~21.3%p). 따라서 이 개선은 시간 분할에서 재확인되기 "
            "전까지 **승격 근거가 되지 못한다**."
        ),
        "verdict": verdict,
        "no_training": True,
        "dacon_upload": False,
        "lockbox_used": False,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    payload["digest"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    RECEIPT.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                       encoding="utf-8")

    lines = [
        "# M271 C7b — C7 산출물의 재판정",
        "",
        f"노드 `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}` / **파생 전용 (후보 아님)**",
        "",
        f"입력 `{SOURCE.name}` digest `{source_digest}`. teacher 적합·표면 재구성 없음.",
        "",
        "## 1. 무엇을 교정하는가",
        "",
        f"C7 은 자기 V1 로 VOID 였고 **그 발화가 옳았다**. 다만 원인이 하네스가 아니라 "
        f"기준 오지정이다 — C7 은 **학습행 무작위 KFold OOF** 를 재는데 나는 "
        f"**평가 fold test 행** 잔차({C66_TEST_SIGMA})를 기준으로 걸었다.",
        "",
        f"올바른 기준은 C1N58 이 같은 방식으로 잰 shuffle 값 **{C58_SHUFFLE_SIGMA}** 이고, "
        f"C7 base 는 **{base_sigma:.4f}** 로 차이 {abs(base_sigma - C58_SHUFFLE_SIGMA):.4f} 다.",
        "",
        "## 2. 팔 (재계산 없음, 결정적)",
        "",
        "| 팔 | 전체 sigma | g1 | g2 | g3 | 감소율 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ("base", "deep", "long", "seq", "seq_deep"):
        a = arms[arm]
        lines.append(
            f"| {arm} | {float(a['overall']):.4f} | {float(a['g1']):.4f} | "
            f"{float(a['g2']):.4f} | {float(a['g3']):.4f} | {float(reductions[arm]):+.2%} |"
        )
    lines += [
        "",
        f"최선 **{best_arm}** 감소 **{best_reduction:.2%}** -> 환산 Total "
        f"**{implied_total:+.6f}** / C16 문턱 {required:.2%} ({floor_total:.6f})",
        "",
        "## 3. 사전확약",
        "",
        f"- V1/H1 올바른 기준에서 통과 -> **{h1}**",
        f"- H2 C16 문턱 통과 -> **{h2}**",
        f"- H3 시계열 문맥이 값을 못 한다 -> **{h3}** "
        f"(추가 {payload_in.get('sequence_columns_added')} 열, seq 감소 "
        f"{float(reductions['seq']):+.2%})",
        f"- H4 세 그룹 모두 개선 -> **{h4}**",
        "",
        "## 4. 판정",
        "",
        f"**{verdict}**",
        "",
        payload["limitation"],
        "",
        f"digest `{payload['digest']}`",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== C7b 재판정 ===")
    print(f"[C7b] base {base_sigma:.4f} vs C1N58 shuffle {C58_SHUFFLE_SIGMA} "
          f"(차 {abs(base_sigma - C58_SHUFFLE_SIGMA):.4f}) -> V1 {v1}")
    for arm in ("base", "deep", "long", "seq", "seq_deep"):
        print(f"[C7b] {arm:9s} sigma {float(arms[arm]['overall']):.4f}  "
              f"감소 {float(reductions[arm]):+.2%}")
    print(f"[C7b] 최선 {best_arm} {best_reduction:.2%} -> Total {implied_total:+.6f} "
          f"(문턱 {required:.2%} / {floor_total:.6f})")
    print(f"[C7b] H1 {h1} / H2 {h2} / H3 {h3} / H4 {h4}")
    print(f"[C7b] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
