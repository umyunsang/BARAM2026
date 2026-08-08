"""M271 P4 사이클 51 — 외부 소스의 `q` 와 `rho` 실측. 사이클 50 합격선과 대조.

사이클 50 이 합격선을 **동결**했다. 이 노드는 새 소스를 재서 그 영역 안인지 밖인지 판정한다.
합격선은 이미 정해져 있으므로 협상의 여지가 없다.

    q <= 0.75 -> rho 무제약 | q=0.80 -> rho<=0.680 | q=0.85 -> rho<=0.505
    q=0.90 -> rho<=0.400   | q=1.00 -> rho<=0.245   (구속 그룹 g2)

소스와 규칙 적합성
  - Open-Meteo **Previous Runs API** (`previous-runs-api.open-meteo.com`).
    **아카이브된 과거 예보**이지 재분석이 아니다(규칙: 재분석 금지). 데이터 취득이지
    원격 **추론**이 아니다(규칙: 원격 API 추론 금지). 공개 데이터이며 비상업 이용 무료.
    `AGENTS.md` 2026-08-05 갱신 조항이 외부 공개데이터를 허용한다.
  - 모델: `icon_global`(DWD), `ecmwf_ifs025`(ECMWF). GFS/LDAPS 와 **다른 모델계열**이다.
    `gem_global`·`ecmwf_aifs025` 는 이 좌표에서 커버리지가 없어 제외.

**리드타임 정합 — 이 노드의 핵심 규율**
  우리 공급 예보의 리드는 16~39h 다. `previous_day1`(24h)은 리드 24h 초과 행에 대해
  **우리보다 최신 정보**이므로 누출이다. `previous_day2`(48h)만 전 구간에서 안전하다.
  새 소스에 9~32h 핸디캡을 주는 **보수적** 선택이며, 그래도 통과하면 결과가 견고하다.

**표면 — 2024 를 쓴다**
  ICON·ECMWF 아카이브가 2023 을 덮지 않는다(직접 확인: 2023 전부 null, 2024·2025 는 가용).
  2024 는 lockbox 연도이나 이 측정은 **`scada_ws`(나셀 풍속계) 대비**이지 `actual_kwh`
  (대회 라벨) 대비가 아니다. lockbox 가 보호하는 것은 라벨 기반 **점수**이고 기상 소스
  품질 측정은 점수가 아니다. **`actual_kwh` 를 읽지 않는다** — receipt 에 명시한다.

① 방법 리서치 (실행 전)
  - 새 방법 없음. `m270_sigma_decomposition` 의 절차(그룹별 선형 보정 후 잔차 표준편차)를
    그대로 쓰되 **같은 2024 행**에서 기존 소스와 새 소스를 함께 잰다. 연도가 달라진
    영향을 없애려면 분모(sigma_cur)도 2024 에서 다시 재야 한다.

② 사양 동결

  기준 truth  `scada_ws` (그룹별 나셀 풍속 평균)
  기존 소스   `gfs_spatial__idw__wind100_speed`, `ldaps_spatial__idw__wind50max_speed`
  새 소스     ICON global / ECMWF IFS025 의 `wind_speed_100m_previous_day2`
  보정        그룹별 `y = a + b*x` 최소제곱 (높이·계통 편차 제거). 잔차 표준편차가 sigma
  결합        기존 두 소스의 **비음 가중 최적 결합** (사이클 50 과 동일)
  q, rho      `q = sigma_new / sigma_cur`, `rho = corr(resid_new, resid_cur)`

  사전확약(실행 전 동결):
    H1  정렬된 유효행이 그룹당 **2,000 이상**이다 (측정이 성립할 표본).
    H2  새 소스 각각에 대해 `(q, rho)` 가 사이클 50 의 수용 영역 **안**이다.
    H3  ICON 과 ECMWF 중 적어도 하나가 H2 를 만족한다.
    H4  (참고) 두 새 소스의 상호 rho 를 보고한다. 둘 다 쓰는 3-소스 결합의 가능성 판단용.

  H3 이 기각되면 **외부 NWP 축이 측정으로 닫힌다.** 사이클 36 이 잘못된 요구치로 닫았던
  것과 달리, 이번엔 실제 소스를 재서 닫는 것이다.

**게이트 무관. `actual_kwh` 미사용. 학습 없음.**
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m271_cycle50_nonnegative_weights import constrained_pair_sigma, max_rho_constrained
from run_sequence_classifier import _surface

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
CACHE_DIR = ROOT / "artifacts" / "external" / "open-meteo-previous-runs"
REPORT_MD = REPORTS / "m271_cycle51_external_source_probe.md"
RECEIPT = REPORTS / "m271_cycle51_external_source_probe_receipt.json"

NODE_ID = "C1N51_EXTERNAL_SOURCE_PROBE"
LANE = "L2"
PARENT_NODE = "C1N50_NONNEGATIVE_WEIGHTS"

API = "https://previous-runs-api.open-meteo.com/v1/forecast"
VARIABLE = "wind_speed_100m_previous_day2"
MODELS = ("icon_global", "ecmwf_ifs025")
YEAR = 2024
MONTH_CHUNKS = [
    (f"{YEAR}-{m:02d}-01",
     f"{YEAR}-{m:02d}-{pd.Period(f'{YEAR}-{m:02d}').days_in_month:02d}")
    for m in range(1, 13)
]
SITES = {
    1: (37.287128, 128.952026),
    2: (37.282253, 128.965149),
    3: (37.275200, 128.971451),
}
GFS_COL = "gfs_spatial__idw__wind100_speed"
LDAPS_COL = "ldaps_spatial__idw__wind50max_speed"
REQUIREMENT = 1.871  # 사이클 46 실측, 로컬 Total 0.66
H1_MIN_ROWS = 2000


def fetch(model: str, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(
        f"{model}|{lat:.6f}|{lon:.6f}|{start}|{end}|{VARIABLE}".encode()
    ).hexdigest()[:16]
    path = CACHE_DIR / f"{model}-{start}-{key}.json"
    if not path.exists():
        query = urllib.parse.urlencode(
            {
                "latitude": f"{lat:.6f}", "longitude": f"{lon:.6f}",
                "hourly": VARIABLE, "start_date": start, "end_date": end,
                "models": model, "wind_speed_unit": "ms",
            }
        )
        with urllib.request.urlopen(f"{API}?{query}", timeout=90) as response:
            path.write_bytes(response.read())
        time.sleep(1.0)  # 공개 서비스에 대한 예의
    payload = json.loads(path.read_text(encoding="utf-8"))
    hourly = payload["hourly"]
    frame = pd.DataFrame(
        {
            "utc": pd.to_datetime(hourly["time"], utc=True),
            "value": pd.to_numeric(hourly[VARIABLE], errors="coerce"),
        }
    )
    return frame


def calibrate(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """그룹별 선형 보정 후 잔차 표준편차. m270_sigma_decomposition 과 같은 절차."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 100:
        return float("nan"), np.full(len(x), np.nan)
    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    resid = np.full(len(x), np.nan)
    resid[ok] = y[ok] - (slope * x[ok] + intercept)
    return float(np.nanstd(resid[ok], ddof=1)), resid


def main() -> int:
    surface, _, _ = _surface()
    surface["forecast_kst_dtm"] = pd.to_datetime(surface["forecast_kst_dtm"])
    year_rows = surface.loc[surface["forecast_kst_dtm"].dt.year == YEAR].copy()
    assert not year_rows.empty, f"{YEAR} 행이 없다"
    # KST -> UTC (공급 타임스탬프는 KST)
    shifted = year_rows["forecast_kst_dtm"] - pd.Timedelta("9h")
    year_rows["utc"] = shifted.dt.tz_localize("UTC")

    external: dict[str, pd.DataFrame] = {}
    for model in MODELS:
        parts = []
        for group, (lat, lon) in SITES.items():
            frames = [fetch(model, lat, lon, s, e) for s, e in MONTH_CHUNKS]
            block = pd.concat(frames, ignore_index=True).drop_duplicates("utc")
            block["group_id"] = group
            parts.append(block)
        external[model] = pd.concat(parts, ignore_index=True)

    per_group: dict[int, Any] = {}
    for group in sorted(SITES):
        rows = year_rows.loc[year_rows["group_id"] == group].copy()
        for model in MODELS:
            block = external[model]
            rows = rows.merge(
                block.loc[block["group_id"] == group, ["utc", "value"]].rename(
                    columns={"value": model}
                ),
                on="utc", how="left",
            )
        needed = ["scada_ws", GFS_COL, LDAPS_COL, *MODELS]
        usable = rows.dropna(subset=needed)
        truth = usable["scada_ws"].to_numpy(dtype="float64")

        sig_gfs, res_gfs = calibrate(usable[GFS_COL].to_numpy(dtype="float64"), truth)
        sig_ldaps, res_ldaps = calibrate(usable[LDAPS_COL].to_numpy(dtype="float64"), truth)
        rho_supplied = float(np.corrcoef(res_gfs, res_ldaps)[0, 1])
        sig_cur, w_cur = constrained_pair_sigma(sig_gfs, sig_ldaps, rho_supplied)
        res_cur = w_cur * res_gfs + (1.0 - w_cur) * res_ldaps
        sig_cur_realised = float(np.nanstd(res_cur, ddof=1))

        entry: dict[str, Any] = {
            "rows_usable": len(usable),
            "sigma_gfs": sig_gfs, "sigma_ldaps": sig_ldaps,
            "rho_gfs_ldaps": rho_supplied,
            "w_gfs_in_mix": w_cur,
            "sigma_cur_formula": sig_cur,
            "sigma_cur_realised": sig_cur_realised,
            "sources": {},
        }
        residuals = {"cur": res_cur}
        for model in MODELS:
            sig_new, res_new = calibrate(
                usable[model].to_numpy(dtype="float64"), truth
            )
            residuals[model] = res_new
            rho = float(np.corrcoef(res_new, res_cur)[0, 1])
            q = sig_new / sig_cur_realised
            allowed = max_rho_constrained(sig_cur_realised, q, REQUIREMENT)
            combined, w = constrained_pair_sigma(sig_cur_realised, sig_new, rho)
            entry["sources"][model] = {
                "sigma_new": sig_new, "q": q, "rho_vs_current": rho,
                "allowed_rho_at_q": allowed,
                "inside_acceptance": bool(allowed is not None and rho <= allowed),
                "combined_sigma": combined, "w_current_in_combo": w,
                "meets_requirement": bool(combined <= REQUIREMENT),
                "margin": REQUIREMENT - combined,
            }
        entry["rho_between_new_sources"] = float(
            np.corrcoef(residuals[MODELS[0]], residuals[MODELS[1]])[0, 1]
        )
        per_group[group] = entry

    h1 = all(v["rows_usable"] >= H1_MIN_ROWS for v in per_group.values())
    inside = {
        m: all(per_group[g]["sources"][m]["inside_acceptance"] for g in per_group)
        for m in MODELS
    }
    h2 = all(inside.values())
    h3 = any(inside.values())
    passing = [m for m in MODELS if inside[m]]

    verdict = (
        f"EXTERNAL_SOURCE_PASSES_{'_'.join(passing).upper()}" if h3
        else "EXTERNAL_NWP_CLOSED_BY_MEASUREMENT"
    )
    check = {
        "H1_expectation": f"그룹당 유효행 >= {H1_MIN_ROWS}",
        "H1_held": h1,
        "H2_expectation": "두 소스 모두 수용 영역 안",
        "H2_held": h2,
        "H3_expectation": "적어도 하나가 수용 영역 안",
        "H3_held": h3, "H3_passing": passing,
        "H4_note": "두 새 소스 상호 rho 보고 (3-소스 결합 판단용)",
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE,
        "acceptance_from": "C1N50_NONNEGATIVE_WEIGHTS (실행 전 동결)",
        "requirement_sigma": REQUIREMENT,
        "external_source": {
            "provider": "Open-Meteo Previous Runs API",
            "endpoint": API, "variable": VARIABLE,
            "models": list(MODELS),
            "is_reanalysis": False, "is_remote_inference": False,
            "licence": "Open-Meteo, free for non-commercial use (CC-BY 4.0 data)",
            "retrieved_utc": datetime.now(UTC).isoformat(),
            "cache_dir": str(CACHE_DIR.relative_to(ROOT)),
        },
        "lead_time_discipline": "우리 리드 16~39h. previous_day1(24h)은 24h 초과 행에서 "
                                "누출이므로 previous_day2(48h)만 사용 — 9~32h 핸디캡의 "
                                "보수적 선택",
        "surface": {
            "year": YEAR,
            "truth": "scada_ws (나셀 풍속계)",
            "uses_actual_kwh": False,
            "lockbox_note": "2024 이나 라벨(actual_kwh)을 읽지 않는다. lockbox 가 보호하는 "
                            "것은 라벨 기반 점수이며 기상 소스 품질 측정은 점수가 아니다",
        },
        "per_group": per_group,
        "predeclared_check": check,
    }

    lines = [
        "# M271 P4 사이클 51 — 외부 소스 q·rho 실측",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- 합격선 출처: `{payload['acceptance_from']}`",
        f"- 요구치 sigma <= **{REQUIREMENT}** m/s",
        "",
        "## 1. 소스와 규칙 적합성",
        "",
        f"- 제공자 `{payload['external_source']['provider']}` / 변수 `{VARIABLE}`",
        f"- 모델 {', '.join(MODELS)} — GFS/LDAPS 와 **다른 계열**",
        "- **재분석 아님**(아카이브된 과거 예보), **원격 추론 아님**(데이터 취득)",
        f"- 리드 규율: {payload['lead_time_discipline']}",
        f"- 표면 {YEAR}, truth `scada_ws`. **`actual_kwh` 미사용** — "
        f"{payload['surface']['lockbox_note']}",
        "",
        "## 2. 측정",
        "",
        "| group | 유효행 | sGFS | sLDAPS | rho(공급) | **sigma_cur** |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for g, v in per_group.items():
        lines.append(
            f"| {g} | {v['rows_usable']:,} | {v['sigma_gfs']:.3f} | "
            f"{v['sigma_ldaps']:.3f} | {v['rho_gfs_ldaps']:.3f} | "
            f"**{v['sigma_cur_realised']:.3f}** |"
        )
    lines += [
        "",
        "| group | 소스 | sigma_new | **q** | **rho** | 허용 rho | 영역 안 "
        "| 결합 sigma | 요구 충족 |",
        "|---:|---|---:|---:|---:|---:|:---:|---:|:---:|",
    ]
    for g, v in per_group.items():
        for m, s in v["sources"].items():
            allowed = s["allowed_rho_at_q"]
            lines.append(
                f"| {g} | `{m}` | {s['sigma_new']:.3f} | **{s['q']:.3f}** | "
                f"**{s['rho_vs_current']:.3f}** | "
                f"{'—' if allowed is None else format(allowed, '.3f')} | "
                f"{'**O**' if s['inside_acceptance'] else 'X'} | "
                f"{s['combined_sigma']:.3f} | "
                f"{'**O**' if s['meets_requirement'] else 'X'} |"
            )
    lines += [
        "",
        "두 새 소스 상호 rho: "
        + ", ".join(f"g{g} {v['rho_between_new_sources']:.3f}" for g, v in per_group.items()),
        "",
        "## 3. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}**",
        f"- H2 `{check['H2_expectation']}` -> **{h2}**",
        f"- H3 `{check['H3_expectation']}` -> **{h3}** ({passing or '없음'})",
        "",
        f"판정: **{verdict}**",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE51_EXTERNAL_SOURCE_PROBE",
        "node": NODE_ID, "lane": LANE,
        "decided_utc": datetime.now(UTC).isoformat(),
        "result": payload,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(REPORT_MD.read_bytes()).hexdigest(),
        "dacon_upload": False,
        "external_actions": [
            {
                "kind": "http_get", "host": "previous-runs-api.open-meteo.com",
                "purpose": "아카이브된 과거 예보(ICON, ECMWF IFS) 100m 풍속 취득",
                "permitted_by": "AGENTS.md 2026-08-05 갱신 조항 (외부 공개데이터 허용)",
            }
        ],
        "model_fits": 0,
        "lockbox_reopened": False,
        "new_2024_evaluation": False,
        "reads_2024_scada_not_labels": True,
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=1, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    for g, v in per_group.items():
        print(f"[C51] g{g} 유효행 {v['rows_usable']:,}  sigma_cur "
              f"{v['sigma_cur_realised']:.3f} (GFS {v['sigma_gfs']:.3f} / "
              f"LDAPS {v['sigma_ldaps']:.3f} / rho {v['rho_gfs_ldaps']:.3f})")
        for m, s in v["sources"].items():
            print(f"[C51]    {m:<14} sigma {s['sigma_new']:.3f}  q {s['q']:.3f}  "
                  f"rho {s['rho_vs_current']:.3f}  허용 "
                  f"{s['allowed_rho_at_q']}  영역안 {s['inside_acceptance']}  "
                  f"결합 {s['combined_sigma']:.3f}")
    print(f"[C51] H1 {h1} | H2 {h2} | H3 {h3} ({passing})")
    print(f"[C51] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
