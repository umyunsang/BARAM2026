"""M271 N0 자식 A1 — 라벨 특성 정밀 파악.

동결 사양(`m271_n0_method.SPECS['A1_labels']`)의 7개 산출만 만든다. 사양에 없는 것은
만들지 않고, 여기서 도메인 결론을 내리지 않는다.

방법 리서치(①) 결과에 따라 catch22/tsfresh 는 채택하지 않았다(pycatch22 는 GPLv3+,
tsfresh 는 목적 불일치). 명시 통계량을 직접 계산하며 신규 의존성은 0이다.

읽기 전용. 모델을 적합하지 않고 2024 행을 읽지 않는다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m271_n0_common import fmt, load_tables, write_node_artifacts

from baram.constants import CAPACITIES_KWH

NODE = "A1_labels"
ELIGIBLE_FRACTION = 0.10  # 공식 산식: actual >= 설비용량의 10% 인 행만 평가
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
# 기존 진단이 써 온 y 대역 경계. A7 의 결손 셀 축 후보다.
Y_BAND_EDGES = (0.10, 0.25, 0.45, 0.70, 1.10)


def to_wide(labels_long: pd.DataFrame) -> pd.DataFrame:
    wide = labels_long.pivot_table(
        index="forecast_kst_dtm", columns="group_id", values="actual_kwh", dropna=False
    ).sort_index()
    wide.columns = [f"g{int(c)}" for c in wide.columns]
    return wide


def coverage(wide: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for group in (1, 2, 3):
        col = f"g{group}"
        series = wide[col]
        cap = CAPACITIES_KWH[group]
        present = series.notna()
        y = series / cap
        rows.append(
            {
                "group": group,
                "capacity_kwh": cap,
                "rows_total": len(series),
                "rows_present": int(present.sum()),
                "missing_fraction": float((~present).mean()),
                "first_present": str(series[present].index.min()),
                "last_present": str(series[present].index.max()),
                "zero_fraction_of_present": float((series[present] == 0).mean()),
                "below_eligible_fraction_of_present": float(
                    (y[present] < ELIGIBLE_FRACTION).mean()
                ),
                "eligible_fraction_of_present": float((y[present] >= ELIGIBLE_FRACTION).mean()),
                "eligible_rows": int((y[present] >= ELIGIBLE_FRACTION).sum()),
            }
        )
    return rows


def eligible_by_month(wide: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    month = wide.index.to_period("M").astype(str)
    for group in (1, 2, 3):
        y = wide[f"g{group}"] / CAPACITIES_KWH[group]
        frame = pd.DataFrame({"month": month, "y": y.to_numpy()}).dropna()
        if frame.empty:
            continue
        agg = frame.groupby("month")["y"].agg(
            rows="size",
            eligible=lambda s: float((s >= ELIGIBLE_FRACTION).mean()),
            mean_y="mean",
        )
        for m, row in agg.iterrows():
            out.append(
                {
                    "group": group,
                    "month": m,
                    "rows": int(row["rows"]),
                    "eligible_fraction": float(row["eligible"]),
                    "mean_y": float(row["mean_y"]),
                }
            )
    return out


def distribution(wide: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for group in (1, 2, 3):
        y = (wide[f"g{group}"] / CAPACITIES_KWH[group]).dropna()
        eligible = y[y >= ELIGIBLE_FRACTION]
        entry: dict[str, Any] = {
            "group": group,
            "n_present": len(y),
            "n_eligible": len(eligible),
            "mean_y_all": float(y.mean()),
            "mean_y_eligible": float(eligible.mean()),
            "std_y_eligible": float(eligible.std()),
        }
        for q in QUANTILES:
            entry[f"q{int(q * 100):02d}"] = float(y.quantile(q))
        bands = pd.cut(eligible, bins=list(Y_BAND_EDGES), right=True)
        share = bands.value_counts(normalize=True).sort_index()
        gen_share = eligible.groupby(bands, observed=False).sum() / eligible.sum()
        entry["band_row_share"] = {str(k): float(v) for k, v in share.items()}
        entry["band_generation_share"] = {str(k): float(v) for k, v in gen_share.items()}
        rows.append(entry)
    return rows


def cycles(wide: pd.DataFrame) -> dict[str, Any]:
    hour = wide.index.hour
    month = wide.index.month
    out: dict[str, Any] = {"by_hour": [], "by_month": []}
    for group in (1, 2, 3):
        y = wide[f"g{group}"] / CAPACITIES_KWH[group]
        frame = pd.DataFrame({"hour": hour, "month": month, "y": y.to_numpy()}).dropna()
        for key, dest in (("hour", "by_hour"), ("month", "by_month")):
            agg = frame.groupby(key)["y"].agg(["mean", "std", "size"])
            for k, row in agg.iterrows():
                out[dest].append(
                    {
                        "group": group,
                        key: int(k),
                        "mean_y": float(row["mean"]),
                        "std_y": float(row["std"]),
                        "rows": int(row["size"]),
                    }
                )
    return out


def ramps(wide: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for group in (1, 2, 3):
        y = wide[f"g{group}"] / CAPACITIES_KWH[group]
        # 시간 간격이 1시간인 연속 쌍만 사용한다. 결측 구간을 건너뛰어 계산하지 않는다.
        gap_ok = wide.index.to_series().diff() == pd.Timedelta("1h")
        dy = y.diff().where(gap_ok.to_numpy()).abs().dropna()
        rho = float(y.autocorr(lag=1))
        entry: dict[str, Any] = {
            "group": group,
            "n_pairs": len(dy),
            "mean_abs_ramp": float(dy.mean()),
            "lag1_autocorr_y": rho,
            # 자기상관 하에서의 유효표본수. AR(1) 근사 n_eff = n * (1-rho)/(1+rho).
            "effective_sample_ratio": float((1.0 - rho) / (1.0 + rho)),
        }
        for q in (0.50, 0.75, 0.90, 0.95, 0.99):
            entry[f"ramp_q{int(q * 100):02d}"] = float(dy.quantile(q))
        rows.append(entry)
    return rows


def group_correlation(wide: pd.DataFrame) -> dict[str, Any]:
    y = pd.DataFrame(
        {f"g{g}": wide[f"g{g}"] / CAPACITIES_KWH[g] for g in (1, 2, 3)}, index=wide.index
    )
    both = y.dropna()
    return {
        "rows_all_three_present": len(both),
        "pearson": {f"{a}~{b}": float(both[a].corr(both[b]))
                    for a, b in (("g1", "g2"), ("g1", "g3"), ("g2", "g3"))},
    }


def group3_gap(wide: pd.DataFrame) -> dict[str, Any]:
    y3 = wide["g3"]
    present = y3.notna()
    by_year = pd.Series(present.to_numpy()).groupby(wide.index.year).mean()
    return {
        "present_fraction_by_year": {int(k): float(v) for k, v in by_year.items()},
        "first_present": str(y3[present].index.min()),
        "rows_lost_vs_g1": int(wide["g1"].notna().sum() - present.sum()),
        "note": "그룹3 라벨이 2022 에 없으므로 그룹 풀링 학습 표면이 그룹1·2 대비 짧다.",
    }


def run(tables: Any, input_hashes: dict[str, str]) -> dict[str, Any]:
    wide = to_wide(tables.labels_long)
    payload: dict[str, Any] = {
        "coverage": coverage(wide),
        "eligible_by_month": eligible_by_month(wide),
        "distribution": distribution(wide),
        "cycles": cycles(wide),
        "ramps": ramps(wide),
        "group_correlation": group_correlation(wide),
        "group3_gap": group3_gap(wide),
        "y_band_edges": list(Y_BAND_EDGES),
    }

    cov = payload["coverage"]
    dist = payload["distribution"]
    rmp = payload["ramps"]
    lines = [
        "## 1. 커버리지와 유효행",
        "",
        "공식 산식은 `actual >= 0.1 * capacity` 인 행만 채점한다. 유효행 비중이 곧 채점 대상의",
        "크기이므로 결손 회계의 분모를 결정한다.",
        "",
        "| 그룹 | 설비 | 전체 | 존재 | 결측 | 0 | 저출력 | **유효행** | 유효행수 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cov:
        lines.append(
            f"| {c['group']} | {c['capacity_kwh']:,.0f} | {c['rows_total']:,} | "
            f"{c['rows_present']:,} | {c['missing_fraction']:.1%} | "
            f"{c['zero_fraction_of_present']:.1%} | "
            f"{c['below_eligible_fraction_of_present']:.1%} | "
            f"**{c['eligible_fraction_of_present']:.1%}** | {c['eligible_rows']:,} |"
        )

    lines += [
        "",
        "## 2. 발전량 분포와 y 대역",
        "",
        "| 그룹 | 유효행 | 평균 y(전체/유효) | 표준편차(유효) | q10 | q50 | q90 |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for d in dist:
        lines.append(
            f"| {d['group']} | {d['n_eligible']:,} | "
            f"{fmt(d['mean_y_all'], 3)} / {fmt(d['mean_y_eligible'], 3)} | "
            f"{fmt(d['std_y_eligible'], 3)} | {fmt(d['q10'], 3)} | {fmt(d['q50'], 3)} | "
            f"{fmt(d['q90'], 3)} |"
        )

    lines += [
        "",
        f"y 대역 경계 `{list(Y_BAND_EDGES)}` 기준, 유효행의 행 비중과 **발전량 질량 비중**:",
        "",
        "| 그룹 | 대역 | 행 비중 | 발전량 질량 비중 |",
        "|---:|---|---:|---:|",
    ]
    for d in dist:
        for band, row_share in d["band_row_share"].items():
            gen = d["band_generation_share"].get(band, 0.0)
            lines.append(f"| {d['group']} | `{band}` | {row_share:.1%} | **{gen:.1%}** |")

    lines += [
        "",
        "FICR 은 발전량 가중이므로 결손 회계에서 중요한 것은 행 비중이 아니라 **질량 비중**이다.",
        "",
        "## 3. 램프와 자기상관",
        "",
        "| 그룹 | 연속쌍 | 평균 \\|Δy\\| | q50 | q90 | q99 | lag-1 autocorr | **유효표본 비율** |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rmp:
        eff = r["effective_sample_ratio"]
        lines.append(
            f"| {r['group']} | {r['n_pairs']:,} | {fmt(r['mean_abs_ramp'], 4)} | "
            f"{fmt(r['ramp_q50'], 4)} | {fmt(r['ramp_q90'], 4)} | {fmt(r['ramp_q99'], 4)} | "
            f"{fmt(r['lag1_autocorr_y'], 4)} | **{eff:.3%}** (≈{int(r['n_pairs'] * eff):,}행) |"
        )

    corr = payload["group_correlation"]
    g3 = payload["group3_gap"]
    lines += [
        "",
        "**lag-1 자기상관이 높다는 것은 유효표본수가 행수보다 훨씬 작다는 뜻이다.** A2 의 상호",
        "정보량 해석과 모든 유의성 주장에 이 사실이 적용된다.",
        "",
        "## 4. 그룹간 상관",
        "",
        f"세 그룹이 모두 존재하는 행: {corr['rows_all_three_present']:,}",
        "",
        "| 쌍 | Pearson |",
        "|---|---:|",
    ]
    for pair, value in corr["pearson"].items():
        lines.append(f"| {pair} | {fmt(value, 4)} |")

    lines += [
        "",
        "## 5. 그룹3 이력 결손",
        "",
        f"- 최초 존재: `{g3['first_present']}`",
        f"- 그룹1 대비 부족한 행: **{g3['rows_lost_vs_g1']:,}**",
        "- 연도별 존재 비중: "
        + ", ".join(f"{y}={v:.0%}" for y, v in g3["present_fraction_by_year"].items()),
        "",
        g3["note"],
        "",
        "## 6. 사전확약 대조",
        "",
        "이 노드는 가설 검정이 아니라 특성 기술이므로 사전확약 기대가 없다. 7개 산출이 모두",
        "생성되었으므로 중단 조건을 충족한다.",
    ]

    write_node_artifacts(
        node=NODE,
        title="M271 N0/A1 — 라벨 특성",
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
    for c in payload["coverage"]:
        print(f"[A1] g{c['group']} 유효행={c['eligible_fraction_of_present']:.1%} "
              f"({c['eligible_rows']:,}행) 결측={c['missing_fraction']:.1%}")
    for r in payload["ramps"]:
        print(f"[A1] g{r['group']} lag1 자기상관={r['lag1_autocorr_y']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
