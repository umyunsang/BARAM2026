"""M271 N0 자식 A2 — 공급 NWP 컬럼 전수 인벤토리와 라벨 관련성 스크린.

동결 사양(`m271_n0_method.SPECS['A2_columns']`)의 3개 산출만 만든다.

방법 리서치(①) 결과: KSG kNN 상호정보량(`sklearn.feature_selection.mutual_info_regression`)
을 채택하되 태그는 `near_match_only` 다. 두 한계가 명시적으로 기록되어 있다.

  1. 주변 MI 는 **기존에 쓰이는 컬럼 대비 추가 정보**를 재지 못한다. A2 가 알고 싶은 것은
     조건부 관련성인데 이 도구는 그것을 재지 않는다.
  2. kNN 추정량은 편향되고, 우리 라벨은 lag-1 자기상관이 0.95 를 넘어 **유효표본수가 행수
     보다 40배 이상 작다**(A1 측정).

따라서 MI 는 **순위 스크린으로만** 쓴다. 절대값과 유의성을 주장하지 않는다. 채택 판정은
시간순 안전 홀드아웃에서의 추가이득으로 별도 측정하며 이 노드는 그것을 하지 않는다.

읽기 전용. 모델을 적합하지 않고 2024 행을 읽지 않는다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import mutual_info_regression

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m271_n0_common import SEED, fmt, load_tables, write_node_artifacts

from baram.constants import CAPACITIES_KWH

NODE = "A2_columns"
ROOT = Path(__file__).resolve().parents[2]
SPATIAL_CONFIG = ROOT / "configs" / "features" / "spatial_v2.yaml"
SRC_DIR = ROOT / "src" / "baram"

ELIGIBLE_FRACTION = 0.10
# 예보 식별·좌표 컬럼. 기상 변수가 아니므로 인벤토리에서 제외한다.
KEY_COLUMNS = {
    "forecast_kst_dtm", "data_available_kst_dtm", "grid_id", "latitude", "longitude",
    "operating_day", "operating_year", "operating_quarter", "lead_hour", "issuance_batch",
}


def declared_usage() -> dict[str, Any]:
    """v2 공간 피처 설정이 선언한 사용 컬럼과, src 전체에서 참조되는 컬럼."""
    config = yaml.safe_load(SPATIAL_CONFIG.read_text(encoding="utf-8"))
    declared = {src: set(meta["variables"]) for src, meta in config["sources"].items()}
    src_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(SRC_DIR.rglob("*.py"))
    )
    return {"declared": declared, "src_text": src_text}


def grid_mean(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """격자 평균으로 시각당 1행을 만든다. 공간 구조는 A3 가 다루고 여기서는 스크린만 한다."""
    return (
        frame.loc[:, ["forecast_kst_dtm", *columns]]
        .groupby("forecast_kst_dtm", as_index=True)
        .mean()
        .sort_index()
    )


def screen_source(
    weather: pd.DataFrame,
    label_wide: pd.DataFrame,
    source: str,
    declared: set[str],
    src_text: str,
) -> list[dict[str, Any]]:
    columns = [c for c in weather.columns if c not in KEY_COLUMNS]
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(weather[c])]
    features = grid_mean(weather, numeric)

    rows: list[dict[str, Any]] = []
    scores: dict[int, np.ndarray] = {}
    for group in (1, 2, 3):
        y = label_wide[f"g{group}"] / CAPACITIES_KWH[group]
        joined = features.join(y.rename("y"), how="inner").dropna()
        joined = joined.loc[joined["y"] >= ELIGIBLE_FRACTION]
        matrix = joined.loc[:, numeric].to_numpy(dtype=float)
        target = joined["y"].to_numpy(dtype=float)
        scores[group] = mutual_info_regression(matrix, target, random_state=SEED)

    for index, column in enumerate(numeric):
        per_group = {g: float(scores[g][index]) for g in (1, 2, 3)}
        rows.append(
            {
                "source": source,
                "column": column,
                "declared_in_spatial_v2": column in declared,
                "referenced_in_src": column in src_text,
                "mi_g1": per_group[1],
                "mi_g2": per_group[2],
                "mi_g3": per_group[3],
                "mi_mean": float(np.mean(list(per_group.values()))),
            }
        )
    rows.sort(key=lambda r: -r["mi_mean"])
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def run(tables: Any, input_hashes: dict[str, str]) -> dict[str, Any]:
    usage = declared_usage()
    label_wide = tables.labels_long.pivot_table(
        index="forecast_kst_dtm", columns="group_id", values="actual_kwh", dropna=False
    ).sort_index()
    label_wide.columns = [f"g{int(c)}" for c in label_wide.columns]

    gfs = screen_source(
        tables.gfs_train, label_wide, "gfs", usage["declared"]["gfs"], usage["src_text"]
    )
    ldaps = screen_source(
        tables.ldaps_train, label_wide, "ldaps", usage["declared"]["ldaps"], usage["src_text"]
    )

    # A1 이 측정한 자기상관을 그대로 물려받아 유효표본을 계산한다.
    rho = {}
    for group in (1, 2, 3):
        y = (label_wide[f"g{group}"] / CAPACITIES_KWH[group]).dropna()
        rho[group] = float(y.autocorr(lag=1))
    n_rows = {
        g: int(
            ((label_wide[f"g{g}"] / CAPACITIES_KWH[g]).dropna() >= ELIGIBLE_FRACTION).sum()
        )
        for g in (1, 2, 3)
    }
    effective = {
        g: {
            "rows_screened": n_rows[g],
            "lag1_autocorr": rho[g],
            "effective_ratio": float((1 - rho[g]) / (1 + rho[g])),
            "effective_rows": int(n_rows[g] * (1 - rho[g]) / (1 + rho[g])),
        }
        for g in (1, 2, 3)
    }

    def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
        used = [r for r in rows if r["declared_in_spatial_v2"]]
        unused = [r for r in rows if not r["declared_in_spatial_v2"]]
        # yaml 미선언과 '실제로 안 쓰임'은 다르다. weather.py 등 비공간 경로가 쓰는 컬럼이
        # 있으므로 둘을 섞으면 미사용 규모를 과장하게 된다.
        untouched = [
            r for r in rows if not r["declared_in_spatial_v2"] and not r["referenced_in_src"]
        ]
        return {
            "total_columns": len(rows),
            "declared_columns": len(used),
            "undeclared_columns": len(unused),
            "untouched_columns": len(untouched),
            "declared_mean_mi": float(np.mean([r["mi_mean"] for r in used])) if used else 0.0,
            "undeclared_mean_mi": float(np.mean([r["mi_mean"] for r in unused]))
            if unused
            else 0.0,
            "untouched_mean_mi": float(np.mean([r["mi_mean"] for r in untouched]))
            if untouched
            else 0.0,
            "best_undeclared": unused[0] if unused else None,
            "best_untouched": untouched[0] if untouched else None,
            "undeclared_in_top10": sum(1 for r in rows[:10] if not r["declared_in_spatial_v2"]),
            "untouched_in_top10": sum(
                1
                for r in rows[:10]
                if not r["declared_in_spatial_v2"] and not r["referenced_in_src"]
            ),
        }

    payload: dict[str, Any] = {
        "gfs": {"columns": gfs, "summary": summarise(gfs)},
        "ldaps": {"columns": ldaps, "summary": summarise(ldaps)},
        "effective_sample": effective,
        "interpretation_limits": [
            "주변 MI 이므로 기존 사용 컬럼 대비 추가 정보를 재지 않는다.",
            "kNN 추정량은 편향되며 자기상관 하에서 유효표본이 크게 줄어든다.",
            "순위 스크린 전용. 절대값과 유의성을 주장하지 않는다.",
        ],
    }

    gs, ls = payload["gfs"]["summary"], payload["ldaps"]["summary"]
    lines = [
        "## 1. 인벤토리 대조",
        "",
        "**`spatial_v2.yaml` 미선언과 '실제로 안 쓰임'은 다르다.** `weather.py` 등 비공간",
        "경로가 참조하는 컬럼이 있으므로 둘을 섞으면 미사용 규모를 과장하게 된다. 세 번째",
        "열이 yaml 과 `src/` 어디에도 없는 진짜 미사용 컬럼이다.",
        "",
        "| 소스 | 전체 기상컬럼 | yaml 선언 | yaml 미선언 | **src 에도 없음** | 상위10중 미사용 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| GFS | {gs['total_columns']} | {gs['declared_columns']} | "
        f"{gs['undeclared_columns']} | **{gs['untouched_columns']}** | "
        f"{gs['untouched_in_top10']} |",
        f"| LDAPS | {ls['total_columns']} | {ls['declared_columns']} | "
        f"{ls['undeclared_columns']} | **{ls['untouched_columns']}** | "
        f"{ls['untouched_in_top10']} |",
        "",
        "## 2. 해석 한계 (먼저 읽을 것)",
        "",
        "| 그룹 | 스크린 행수 | lag-1 autocorr | 유효표본 비율 | **유효 행수** |",
        "|---:|---:|---:|---:|---:|",
    ]
    for group in (1, 2, 3):
        e = effective[group]
        lines.append(
            f"| {group} | {e['rows_screened']:,} | {fmt(e['lag1_autocorr'], 4)} | "
            f"{e['effective_ratio']:.2%} | **{e['effective_rows']:,}** |"
        )

    lines += [
        "",
        f"GFS {gs['total_columns']}개 + LDAPS {ls['total_columns']}개를 유효 수백 행으로 "
        "선별하는 셈이다. 상위 몇 개는 우연으로도 충분히 뜬다.",
        "",
        "MI 는 **순위 스크린 전용**이다. 절대값과 유의성을 주장하지 않으며, 채택 판정은",
        "시간순 안전 홀드아웃에서의 추가이득으로 따로 측정한다. 이 노드는 그것을 하지 않는다.",
        "",
        "## 3. GFS 컬럼 순위",
        "",
        "| 순위 | 컬럼 | 선언 | MI g1 | MI g2 | MI g3 | 평균 |",
        "|---:|---|:---:|---:|---:|---:|---:|",
    ]
    for r in gfs:
        mark = "O" if r["declared_in_spatial_v2"] else "-"
        lines.append(
            f"| {r['rank']} | `{r['column']}` | {mark} | {fmt(r['mi_g1'], 4)} | "
            f"{fmt(r['mi_g2'], 4)} | {fmt(r['mi_g3'], 4)} | **{fmt(r['mi_mean'], 4)}** |"
        )

    lines += [
        "",
        "## 4. LDAPS 컬럼 순위",
        "",
        "| 순위 | 컬럼 | 선언 | MI g1 | MI g2 | MI g3 | 평균 |",
        "|---:|---|:---:|---:|---:|---:|---:|",
    ]
    for r in ldaps:
        mark = "O" if r["declared_in_spatial_v2"] else "-"
        lines.append(
            f"| {r['rank']} | `{r['column']}` | {mark} | {fmt(r['mi_g1'], 4)} | "
            f"{fmt(r['mi_g2'], 4)} | {fmt(r['mi_g3'], 4)} | **{fmt(r['mi_mean'], 4)}** |"
        )

    lines += [
        "",
        "## 5. 선언 대 미선언 대조",
        "",
        "| 소스 | yaml선언 평균 | yaml미선언 평균 | **진짜미사용 평균** | 최고 진짜미사용 |",
        "|---|---:|---:|---:|---|",
    ]
    for name, s in (("GFS", gs), ("LDAPS", ls)):
        best = s["best_untouched"]
        best_str = (
            f"`{best['column']}` (순위 {best['rank']}, MI {fmt(best['mi_mean'], 4)})"
            if best
            else "-"
        )
        lines.append(
            f"| {name} | {fmt(s['declared_mean_mi'], 4)} | {fmt(s['undeclared_mean_mi'], 4)} | "
            f"**{fmt(s['untouched_mean_mi'], 4)}** | {best_str} |"
        )
    lines += [
        "",
        "GFS 상위권의 `isobaricInhPa_850_u`·`planetaryBoundaryLayer_0_u`·`isobaricInhPa_700_u`",
        "는 yaml 에 없을 뿐 `weather.py` 가 비공간 경로에서 쓴다. 이들을 '미사용'으로 세면",
        "안 된다. 진짜 미사용은 대부분 열역학·종관 스칼라(500 hPa 지위고도·기온, 이슬점,",
        "비습, 해면기압, 복사)다.",
    ]

    lines += [
        "",
        "## 6. 사전확약 대조",
        "",
        "동결된 기대: *MI 절대값은 신뢰하지 않는다. 순위만 본다. 상위 컬럼이라도 조건부",
        "추가이득은 별도 측정 전까지 주장하지 않는다.*",
        "",
        "→ 유지. 이 리포트는 순위와 인벤토리만 보고하며 채택을 주장하지 않는다.",
        "",
        "## 7. 라우팅 입력",
        "",
        "동결 사양의 라우팅 규칙: *MI 가 임계 미만이면 C8(PRUNE), 이상이면 C1 로 L2 방향",
        "리서치 발화.* 임계값은 라우터 표가 보유하며 이 노드는 판정하지 않는다.",
    ]

    write_node_artifacts(
        node=NODE,
        title="M271 N0/A2 — NWP 컬럼 인벤토리와 관련성 스크린",
        report_lines=lines,
        payload=payload,
        input_hashes=input_hashes,
        parents=[],
        script_path=Path(__file__),
    )
    return payload


def main() -> int:
    tables, hashes = load_tables()
    payload = run(tables, hashes)
    for name in ("gfs", "ldaps"):
        s = payload[name]["summary"]
        print(f"[A2] {name}: 전체 {s['total_columns']} / 선언 {s['declared_columns']} / "
              f"미선언 {s['undeclared_columns']} / 상위10중 미선언 {s['undeclared_in_top10']}")
        best = s["best_undeclared"]
        if best:
            print(f"      최고 미선언: {best['column']} (순위 {best['rank']}, "
                  f"MI {best['mi_mean']:.4f})")
    for g, e in payload["effective_sample"].items():
        print(f"[A2] g{g} 스크린 {e['rows_screened']:,}행 -> 유효 {e['effective_rows']:,}행")
    return 0


if __name__ == "__main__":
    sys.exit(main())
