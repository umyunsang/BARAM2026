"""M270 P0: bootstrap the experiment graph from the existing planning record.

Extracts every documented M-node from the task_plan decision table and the progress log,
assigns a status and a STRUCTURED CLOSURE PREMISE, then decides which closed lanes revive
once the technical scope widens from the project's self-imposed limits to the competition
rule boundary.

The premise classifier is deliberately CONSERVATIVE: a node is only marked revivable when
its recorded rationale contains an explicit capability blocker. Everything else defaults to
an evidence-based closure, which does not revive. Under-revival is recoverable by review;
over-revival would resurrect lanes that genuinely failed on their merits.

Read-only inputs. No model is fitted, no 2024 row is read, no submission is built.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN_DIR = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"

TASK_PLAN = PLAN_DIR / "task_plan.md"
PROGRESS = PLAN_DIR / "progress.md"
GRAPH_OUT = PLAN_DIR / "experiment_graph.json"

NODE_RE = re.compile(r"\bM(\d{1,3})\b")

# Verdict keywords, checked in order. First match wins for a given decision row.
VERDICTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("rejected", ("reject", "close ", "closed", "do not ", "stop ", "forbid")),
    ("promoted", ("promote", "pass the", "build and freeze", "make ", "keep ", "retain")),
    ("built", ("build ", "hand off")),
    ("predeclared", ("predeclare",)),
)

# Conservative capability-blocker phrases. Only these revive a closed node.
# Each entry: premise code -> phrases that must appear verbatim (lowercased).
REVIVABLE_PREMISES: dict[str, tuple[str, ...]] = {
    "NO_SELECTABLE_NWP": (
        "selectable nwp",
        "multiple nwp configurations",
        "unavailable selectable",
    ),
    "NO_LIVE_OBSERVATION": (
        "live observations",
        "unavailable at inference",
        "contemporaneous scada",
    ),
    "NO_EXTERNAL_DATA": (
        "external data",
        "requires unavailable",
        "unavailable inputs",
        "external input",
    ),
    "NO_NEW_DEPENDENCY": (
        "tft",
        "ngboost",
        "deep tier",
        "deep/tft",
        "dependency change",
        "dependency expansion",
        "optional dependency",
    ),
    "SHORT_GROUP3_HISTORY": (
        "group 3 begins in 2023",
        "short history",
        "missing 2022 group-3",
        "zero overlap",
    ),
}

# Non-revivable closure classes, checked after the revivable ones.
EVIDENCE_PHRASES = (
    "regressed",
    "bootstrap positivity",
    "fell below",
    "negative",
    "worsened",
    "collapsed",
    "did not improve",
    "failed",
)
DUPLICATE_PHRASES = ("already", "duplicate", "duplicates", "equivalent variants", "covered")
ORACLE_PHRASES = ("same-fold", "oracle")


# Manually verified lane dispositions. The keyword classifier operates on prose and is only
# an first-pass filter; every entry below was confirmed by reading the originating row.
# Frozen here so the report stays reproducible rather than hand-edited.
MANUAL_REVIEW: tuple[dict[str, str], ...] = (
    {
        "lane": "ramp / weather-pattern classification",
        "origin": "task_plan decision row 'Close the ramp/weather-pattern lane without M251'",
        "recorded_reason": (
            "the published method selects among multiple NWP configurations, but the competition"
            " exposes only the supplied GFS/LDAPS forecasts"
        ),
        "disposition": "REVIVED",
        "why": (
            "the blocker was the absence of alternative NWP sources, not negative evidence."
            " Public forecast archives are now permitted, so the premise no longer holds."
        ),
    },
    {
        "lane": "TFT / NGBoost / deep probabilistic models",
        "origin": "IP@v2 exclusion plus progress entries 2026-08-01 and 2026-08-03",
        "recorded_reason": "execution forbidden; would require unauthorized dependency expansion",
        "disposition": "UNOPENED",
        "why": (
            "never executed even once, so there is no negative evidence to overturn. New"
            " dependencies are now permitted. This is an unexplored lane, not a revived one."
        ),
    },
    {
        "lane": "external public weather / multi-source NWP features",
        "origin": "progress entry 2026-08-01 excluding external data, forecasts, power curves",
        "recorded_reason": "external data excluded by project rule",
        "disposition": "UNOPENED",
        "why": (
            "the project rule was stricter than the competition rule. Public data is admissible"
            " subject to the availability-time and reproducibility capability gate (P2)."
        ),
    },
    {
        "lane": "pretrained time-series foundation models",
        "origin": "progress entry 2026-08-01 excluding pretrained weights",
        "recorded_reason": "pretrained weights excluded by project rule",
        "disposition": "UNOPENED",
        "why": (
            "admissible only for weights publicly released on or before 2026-07-05 under a"
            " commercial-use-permitting OSS license, loaded locally. Gated at P2."
        ),
    },
    {
        "lane": "group-3 missing 2022 labels",
        "origin": "progress entry 2026-08-03 on UNISON SCADA starting 2023-01-01",
        "recorded_reason": "no supplied source overlaps the missing 2022 group-3 labels",
        "disposition": "STAYS_CLOSED",
        "why": (
            "the gap is missing GENERATION LABELS, which no public source can supply and which"
            " private operational data may not supply either. External data can improve group-3"
            " FEATURES, but that is a different lane and does not reopen this one."
        ),
    },
    {
        "lane": "contemporaneous SCADA as an inference feature",
        "origin": "progress entries 2026-08-03",
        "recorded_reason": "SCADA is unavailable at inference time",
        "disposition": "STAYS_CLOSED",
        "why": (
            "a structural property of the task, unchanged by the widened scope. Private"
            " operational data remains forbidden and post-reference-time observation is forbidden."
        ),
    },
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decision_rows() -> list[tuple[str, str]]:
    text = TASK_PLAN.read_text(encoding="utf-8")
    start = text.index("## Decisions Made")
    end = text.index("## Active success criterion")
    rows: list[tuple[str, str]] = []
    for line in text[start:end].split("\n"):
        if not line.startswith("|") or line.count("|") < 3:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2 or cells[0] in {"Decision", ""} or set(cells[0]) <= set("-: "):
            continue
        rows.append((cells[0], cells[1]))
    return rows


def progress_lines() -> list[str]:
    return [
        line.strip()
        for line in PROGRESS.read_text(encoding="utf-8").split("\n")
        if line.strip().startswith("-") and NODE_RE.search(line)
    ]


def classify_verdict(decision: str) -> str | None:
    lowered = decision.lower()
    for status, keywords in VERDICTS:
        if any(keyword in lowered for keyword in keywords):
            return status
    return None


def classify_premise(text: str) -> tuple[str, str | None, bool]:
    """Return (premise_code, matched_phrase, revivable)."""
    lowered = text.lower()
    for code, phrases in REVIVABLE_PREMISES.items():
        for phrase in phrases:
            if phrase in lowered:
                return code, phrase, True
    for phrase in ORACLE_PHRASES:
        if phrase in lowered:
            return "SAME_FOLD_ORACLE_ONLY", phrase, False
    for phrase in DUPLICATE_PHRASES:
        if phrase in lowered:
            return "DUPLICATE_COVERAGE", phrase, False
    for phrase in EVIDENCE_PHRASES:
        if phrase in lowered:
            return "EVIDENCE_NEGATIVE", phrase, False
    return "UNCLASSIFIED", None, False


def main() -> None:
    rows = decision_rows()
    lines = progress_lines()

    nodes: dict[str, dict[str, object]] = {}

    def touch(node_id: str) -> dict[str, object]:
        return nodes.setdefault(
            node_id,
            {
                "id": node_id,
                "scope": None,
                "status": "documented",
                "closure_reason": [],
                "closure_premise": None,
                "premise_match": None,
                "revivable": False,
                "decision_rows": 0,
                "progress_rows": 0,
                "closure_text": [],
                "mention_text": [],
            },
        )

    for decision, rationale in rows:
        subject = NODE_RE.search(decision)
        mentioned = {f"M{value}" for value in NODE_RE.findall(decision + " " + rationale)}
        primary = f"M{subject.group(1)}" if subject else None
        verdict = classify_verdict(decision)
        for node_id in mentioned:
            node = touch(node_id)
            node["decision_rows"] = int(node["decision_rows"]) + 1
            node["mention_text"].append(rationale)  # type: ignore[union-attr]
        if primary:
            node = touch(primary)
            node["closure_reason"].append(decision)  # type: ignore[union-attr]
            # Only a CLOSING subject row may establish this node's closure premise.
            # Predeclaration rows list prohibitions ("no external input is allowed"),
            # which are scope constraints and must never be read as blockers.
            if verdict == "rejected":
                node["closure_text"].append(  # type: ignore[union-attr]
                    decision + " " + rationale
                )
            if verdict:
                node["status"] = verdict

    for line in lines:
        for value in set(NODE_RE.findall(line)):
            node = touch(f"M{value}")
            node["progress_rows"] = int(node["progress_rows"]) + 1
            node["mention_text"].append(line)  # type: ignore[union-attr]

    for node in nodes.values():
        closure_blob = " ".join(node["closure_text"])  # type: ignore[arg-type]
        code, phrase, revivable = classify_premise(closure_blob)
        node["closure_premise"] = code if closure_blob else "NO_SUBJECT_ROW"
        node["premise_match"] = phrase if closure_blob else None
        # Revive only an explicitly rejected lane whose own subject row names a blocker.
        node["revivable"] = bool(revivable and node["status"] == "rejected")
        node["closure_chars"] = len(closure_blob)
        node["mention_chars"] = len(" ".join(node["mention_text"]))  # type: ignore[arg-type]
        del node["closure_text"]
        del node["mention_text"]

    ordered = sorted(nodes.values(), key=lambda n: int(str(n["id"])[1:]))
    revived = [n for n in ordered if n["revivable"]]

    graph = {
        "schema_version": 1,
        "stage": "M270_P0_EXPERIMENT_GRAPH_BOOTSTRAP",
        "sources_sha256": {
            "task_plan.md": sha256_file(TASK_PLAN),
            "progress.md": sha256_file(PROGRESS),
        },
        "widened_capabilities": [
            "NEW_DEPENDENCY_ALLOWED",
            "EXTERNAL_PUBLIC_DATA_ALLOWED_CONDITIONAL",
            "PRETRAINED_OSS_WEIGHTS_ALLOWED_CONDITIONAL",
        ],
        "still_forbidden": [
            "REMOTE_API_INFERENCE",
            "PRIVATE_OR_INTERNAL_DATA",
            "POST_REFERENCE_TIME_OBSERVATION",
            "REANALYSIS_OR_POST_HOC_CORRECTION",
            "TEST_PERIOD_ANSWERS",
        ],
        "node_count": len(ordered),
        "revived_count": len(revived),
        "manual_review": list(MANUAL_REVIEW),
        "nodes": ordered,
        "model_fits": 0,
        "new_2024_evaluation": False,
        "lockbox_reopened": False,
        "dacon_upload": False,
        "external_actions": [],
    }
    GRAPH_OUT.write_text(
        json.dumps(graph, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )

    status_counts: dict[str, int] = {}
    premise_counts: dict[str, int] = {}
    for node in ordered:
        status_counts[str(node["status"])] = status_counts.get(str(node["status"]), 0) + 1
        premise_counts[str(node["closure_premise"])] = (
            premise_counts.get(str(node["closure_premise"]), 0) + 1
        )

    out: list[str] = []
    out.append("# M270 P0 — 실험 그래프 부트스트랩 및 소급 재개방 판정\n")
    out.append(f"- 적재 노드: **{len(ordered)}개** (결정표 + 진행로그에 문서화된 M-노드)")
    out.append(f"- 재개방(`revived`) 후보: **{len(revived)}개**")
    out.append("- 확대 역량: 신규 의존성 / 공개 외부데이터(조건부) / OSS 가중치(조건부)")
    out.append(
        "- 유지 금지: 원격 API 추론, 비공개 데이터, 기준시점 이후 관측,"
        " 재분석·사후보정, 평가구간 정답\n"
    )

    out.append("## 1. 상태 분포\n")
    out.append("| status | 개수 |")
    out.append("|---|---:|")
    for key in sorted(status_counts, key=lambda k: -status_counts[k]):
        out.append(f"| {key} | {status_counts[key]} |")

    out.append("\n## 2. 종결 전제 분포\n")
    out.append("| closure_premise | 개수 | 재개방 가능 |")
    out.append("|---|---:|---|")
    revivable_codes = set(REVIVABLE_PREMISES)
    for key in sorted(premise_counts, key=lambda k: -premise_counts[k]):
        mark = "예" if key in revivable_codes else "아니오"
        out.append(f"| `{key}` | {premise_counts[key]} | {mark} |")

    out.append("\n## 3. 재개방 후보 노드\n")
    if revived:
        out.append("| 노드 | 전제 | 매칭 문구 | 종결 기록(요약) |")
        out.append("|---|---|---|---|")
        for node in revived:
            reason = (node["closure_reason"] or ["(결정표 기록 없음, 진행로그만)"])[0]  # type: ignore[index]
            out.append(
                f"| `{node['id']}` | `{node['closure_premise']}` | `{node['premise_match']}` | "
                f"{str(reason)[:110]} |"
            )
    else:
        out.append("재개방 후보가 없습니다.")

    out.append("\n## 4. 수동 검증 레인 판정\n")
    out.append("키워드 분류기는 1차 필터일 뿐이며, 아래는 원본 행을 직접 읽어 확인한 결과다.\n")
    out.append("| 레인 | 판정 | 기록된 종결 사유 | 근거 |")
    out.append("|---|---|---|---|")
    for entry in MANUAL_REVIEW:
        out.append(
            f"| {entry['lane']} | **{entry['disposition']}** | "
            f"{entry['recorded_reason']} | {entry['why']} |"
        )

    out.append("\n## 5. 판정 규율 및 한계\n")
    out.append(
        "- 분류기는 **보수적**이다. 명시적 역량 차단 문구가 있을 때만 재개방으로 "
        "판정하고 나머지는 증거 기반 종결로 남긴다. 과소 재개방은 검토로 복구되지만 "
        "과대 재개방은 실제로 실패한 레인을 되살린다."
    )
    out.append(
        "- 이 표는 **후보 목록이지 승격 목록이 아니다.** 각 노드는 P2 역량 게이트를 "
        "통과한 역량에 한해, 그리고 원래 종결 사유가 정말 그 역량에만 의존했는지 "
        "개별 확인한 뒤에만 실행된다."
    )
    out.append(
        "- `UNCLASSIFIED`와 `documented` 상태 노드는 기록이 진행로그에만 있거나 판정 문구가 없는 "
        "경우입니다. 이후 라운드에서 개별 확인 대상입니다."
    )
    out.append(
        f"- 원본 무결성: `task_plan.md` `{sha256_file(TASK_PLAN)[:16]}...`, "
        f"`progress.md` `{sha256_file(PROGRESS)[:16]}...`"
    )

    (REPORTS / "m270_revived_lanes.md").write_text("\n".join(out) + "\n", encoding="utf-8")

    print(f"nodes={len(ordered)} revived={len(revived)}")
    print("status:", dict(sorted(status_counts.items(), key=lambda kv: -kv[1])))
    print("premise:", dict(sorted(premise_counts.items(), key=lambda kv: -kv[1])))
    for node in revived:
        print(
            f"  REVIVED {node['id']:>5}  {node['closure_premise']:<24}"
            f" <- '{node['premise_match']}'"
        )


if __name__ == "__main__":
    main()
