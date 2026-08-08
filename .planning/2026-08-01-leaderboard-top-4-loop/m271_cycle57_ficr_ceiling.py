"""M271 P4 사이클 57 — 전 구간 FICR 천장. 발전량 가중, 경험적 분포, 시간 해상도 교정.

사이클 55 가 급경사 구간의 집계 산포를 쟀고, 그것을 FICR 로 옮기면 g3 는 이미 천장에
붙어 있고 g1·g2 는 여유가 있어 보였다. 그 계산에 결함이 둘 있다.

  D1  **시간 해상도.** SCADA 는 10 분(`power_kw10m` = 10 분당 kWh)이고 라벨은 **시간**
      단위다. 10 분 잔차 6 개를 합치면 평활되므로 사이클 55 의 산포는 시간 기준으로
      **과대추정**이다.
  D2  **정규 가정.** `2*Phi(band/sigma)-1` 로 적중률을 냈다. 출력 잔차는 정지·포화 때문에
      정규가 아니다. **경험적 분포로 직접 세는 것**이 옳다.

그리고 사이클 55 는 급경사 구간만 봤다. FICR 은 **발전량 가중 전 구간**이므로 y 대역별
질량을 실어야 실제 천장이 나온다.

이 노드가 셋을 다 고쳐 **전 구간 FICR 천장**을 낸다.

    천장 = sum_bin  w_gen(bin) * E[정산단위 | 그 bin 의 잔차분포] / 4

`w_gen` 은 유효행(그룹 출력 >= 10% 용량)의 발전량 몫이다. "완벽한 중앙 예보" 가정이므로
예보 오차는 0 이고 남는 것은 **같은 풍속에서 출력이 흩어지는 폭**뿐이다.

① 방법 리서치 (실행 전)
  - 새 방법 없음. 공식 산식의 정산단위 정의를 잔차분포에 직접 적용한다.
  - **경험적 분포를 쓴다.** 정규 가정을 버리는 것이 이 노드의 방법론적 요점이다.
  - 시간 집계는 라벨 정의에 맞춘다: 10 분 kWh 를 시간별로 **합산**.

② 사양 동결

  자료    `open.zip` 의 SCADA 두 파일 (사이클 55 와 동일 원천·그룹 배정·이상치 규칙)
  집계    터빈별 10 분 kWh -> 그룹 시간 합계. 풍속은 그룹 터빈 평균의 시간 평균
  커브    그룹별 풍속 bin(0.5 m/s, 최소 30 행) 평균 출력 = 그 bin 의 "완벽 예보" 값
  잔차    `(그룹 시간출력 - 그 bin 평균) / 그룹 용량`
  유효행  그룹 출력 >= 10% 용량 (공식 채점 조건)
  단위    `|resid| <= 0.06 -> 4`, `<= 0.08 -> 3`, 그 외 0 (경험적으로 센다)
  천장    발전량 가중 평균 단위 / 4

  사전확약(실행 전 동결):
    H1  시간 집계 산포가 10 분 산포보다 **낮다** (D1 확인).
    H2  전 구간 FICR 천장이 현재 배포 FICR **0.402464 보다 높다**.
    H3  g3 천장이 g1·g2 보다 **낮다** (사이클 55 의 방향이 유지되는가).
    H4  천장이 **로컬 Total 0.66 이 요구하는 FICR 보다 높다**.
        기각되면 출력 산포만으로 로컬 0.66 이 물리적으로 불가능하다는 뜻이다.

**게이트 무관. 학습·수집·외부데이터 없음. `actual_kwh` 미사용(SCADA 만).**
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
C46_RECEIPT = REPORTS / "m271_cycle46_closure_falsification_receipt.json"
OPEN_ZIP = Path("/Users/um-yunsang/Downloads/open.zip")
REPORT_MD = REPORTS / "m271_cycle57_ficr_ceiling.md"
RECEIPT = REPORTS / "m271_cycle57_ficr_ceiling_receipt.json"

NODE_ID = "C1N57_FICR_CEILING"
LANE = "L5"
PARENT_NODE = "C1N55_TURBINE_CORRELATION"
CORRECTS = "C1N55_TURBINE_CORRELATION (10 분 해상도, 정규 가정)"

BAND_HIT, BAND_PARTIAL = 0.06, 0.08
BIN_WIDTH = 0.5
MIN_BIN_ROWS = 30
ELIGIBLE = 0.10
DEPLOYED_FICR = 0.402464
LOCAL_TARGET_TOTAL = 0.66

# 사이클 55 의 10 분 집계 산포 (급경사)
C55_STEEP_10MIN = {1: 0.0685, 2: 0.0614, 3: 0.1469}
SPECS = (
    ("vestas", "train/scada_vestas_train.csv", 600.0, {1: range(1, 7), 2: range(7, 13)}),
    ("unison", "train/scada_unison_train.csv", 700.0, {3: range(1, 6)}),
)


def load_group_hourly() -> dict[int, pd.DataFrame]:
    """그룹별 **시간** 출력(정격비)과 풍속. 라벨 정의에 맞춘 집계."""
    per_group: dict[int, list[pd.DataFrame]] = {}
    with zipfile.ZipFile(OPEN_ZIP) as archive:
        for prefix, member, rated, groups in SPECS:
            with archive.open(member) as stream:
                raw = pd.read_csv(stream, parse_dates=["kst_dtm"])
            for group, numbers in groups.items():
                power_cols, wind_cols = [], []
                block = pd.DataFrame({"kst_dtm": raw["kst_dtm"]})
                for number in numbers:
                    name = f"{prefix}_wtg{number:02d}"
                    power = pd.to_numeric(raw.get(f"{name}_power_kw10m"), errors="coerce")
                    wind = pd.to_numeric(raw.get(f"{name}_ws"), errors="coerce")
                    if power is None or wind is None:
                        continue
                    valid = (
                        wind.between(0.0, 50.0, inclusive="left")
                        & power.between(0.0, rated * 1.1)
                    )
                    block[f"p_{name}"] = (power / rated).where(valid)
                    block[f"w_{name}"] = wind.where(valid)
                    power_cols.append(f"p_{name}")
                    wind_cols.append(f"w_{name}")
                block = block.dropna(subset=power_cols + wind_cols)
                block["norm_power"] = block[power_cols].mean(axis=1)
                block["ws"] = block[wind_cols].mean(axis=1)
                block["hour"] = block["kst_dtm"].dt.floor("h")
                hourly = block.groupby("hour").agg(
                    norm_power=("norm_power", "mean"),
                    ws=("ws", "mean"),
                    samples=("norm_power", "size"),
                )
                # 시간당 6 개 표본이 온전한 시각만. 부분 결측은 집계를 왜곡한다.
                hourly = hourly.loc[hourly["samples"] == 6].reset_index()
                hourly["turbines"] = len(power_cols)
                per_group.setdefault(group, []).append(hourly)
    return {g: pd.concat(v, ignore_index=True) for g, v in per_group.items()}


def settlement_units(resid: np.ndarray) -> np.ndarray:
    a = np.abs(resid)
    return np.where(a <= BAND_HIT, 4.0, np.where(a <= BAND_PARTIAL, 3.0, 0.0))


def main() -> int:
    curve = json.loads(C46_RECEIPT.read_text(encoding="utf-8"))["result"][
        "error_scaling_curve"
    ]
    ks = np.array([c["k"] for c in curve])
    totals = np.array([c["total"] for c in curve])
    ficrs = np.array([c["ficr"] for c in curve])
    k_for_066 = float(np.interp(LOCAL_TARGET_TOTAL, totals[::-1], ks[::-1]))
    ficr_required = float(np.interp(k_for_066, ks, ficrs))

    per_group: dict[int, Any] = {}
    for group, frame in load_group_hourly().items():
        frame = frame.dropna(subset=["ws", "norm_power"]).copy()
        frame["bin"] = (frame["ws"] / BIN_WIDTH).round().astype(int)
        stats = frame.groupby("bin")["norm_power"].agg(["mean", "size"])
        thick = stats.loc[stats["size"] >= MIN_BIN_ROWS, "mean"]
        frame = frame.join(thick.rename("bin_mean"), on="bin").dropna(subset=["bin_mean"])
        frame["resid"] = frame["norm_power"] - frame["bin_mean"]

        eligible = frame.loc[frame["norm_power"] >= ELIGIBLE].copy()
        eligible["unit"] = settlement_units(eligible["resid"].to_numpy(dtype="float64"))
        mass = eligible["norm_power"].to_numpy(dtype="float64")
        ceiling = float((eligible["unit"].to_numpy() * mass).sum() / mass.sum() / 4.0)

        bins = []
        for b, cell in eligible.groupby("bin"):
            if len(cell) < MIN_BIN_ROWS:
                continue
            m = cell["norm_power"].to_numpy(dtype="float64")
            bins.append(
                {
                    "bin_center": float(b * BIN_WIDTH),
                    "mean_power": float(cell["bin_mean"].iloc[0]),
                    "rows": len(cell),
                    "gen_share": float(m.sum()),
                    "sigma_resid": float(cell["resid"].std(ddof=1)),
                    "hit6": float((np.abs(cell["resid"]) <= BAND_HIT).mean()),
                    "unit_mean": float(cell["unit"].mean()),
                }
            )
        total_mass = sum(b["gen_share"] for b in bins)
        for b in bins:
            b["gen_weight"] = b["gen_share"] / total_mass if total_mass else 0.0

        steep = [b for b in bins if 0.20 <= b["mean_power"] <= 0.80]
        per_group[group] = {
            "hours": len(frame),
            "eligible_hours": len(eligible),
            "turbines": int(frame["turbines"].iloc[0]),
            "ficr_ceiling": ceiling,
            "sigma_all_eligible": float(eligible["resid"].std(ddof=1)),
            "sigma_steep_hourly": float(np.average(
                [b["sigma_resid"] for b in steep],
                weights=[b["gen_weight"] for b in steep],
            )) if steep else float("nan"),
            "sigma_steep_10min_cycle55": C55_STEEP_10MIN[group],
            "steep_gen_weight": float(sum(b["gen_weight"] for b in steep)),
            "bins": bins,
        }

    h1 = all(
        v["sigma_steep_hourly"] < v["sigma_steep_10min_cycle55"] for v in per_group.values()
    )
    ceilings = {g: v["ficr_ceiling"] for g, v in per_group.items()}
    mean_ceiling = float(np.mean(list(ceilings.values())))
    h2 = bool(mean_ceiling > DEPLOYED_FICR)
    h3 = bool(ceilings[3] < min(ceilings[1], ceilings[2]))
    h4 = bool(mean_ceiling > ficr_required)

    if not h4:
        verdict = "SCATTER_ALONE_BLOCKS_LOCAL_066"
    elif h2:
        verdict = "CEILING_ABOVE_CURRENT_ROOM_REMAINS"
    else:
        verdict = "CEILING_AT_OR_BELOW_CURRENT"

    check = {
        "H1_expectation": "시간 집계 산포 < 10 분 산포 (사이클 55 과대추정 확인)",
        "H1_held": h1,
        "H2_expectation": f"전 구간 FICR 천장 > 배포 {DEPLOYED_FICR}",
        "H2_held": h2, "H2_measured": mean_ceiling,
        "H3_expectation": "g3 천장 < g1, g2",
        "H3_held": h3,
        "H4_expectation": f"천장 > 로컬 0.66 요구 FICR ({ficr_required:.6f})",
        "H4_held": h4,
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE, "corrects": CORRECTS,
        "defects_fixed": {
            "D1": "10 분 -> 시간 집계 (라벨 정의에 맞춤)",
            "D2": "정규 가정 -> 경험적 잔차분포로 직접 계수",
            "D3": "급경사 한정 -> 발전량 가중 전 구간",
        },
        "assumption": "완벽한 중앙 예보. 예보 오차 0, 남는 것은 같은 풍속에서의 출력 산포뿐",
        "no_training": True, "no_collection": True, "uses_actual_kwh": False,
        "required": {
            "local_target_total": LOCAL_TARGET_TOTAL,
            "k_for_target": k_for_066,
            "ficr_required": ficr_required,
            "deployed_ficr": DEPLOYED_FICR,
        },
        "per_group": per_group,
        "mean_ceiling": mean_ceiling,
        "predeclared_check": check,
    }

    lines = [
        "# M271 P4 사이클 57 — 전 구간 FICR 천장",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        f"- 교정 대상: `{CORRECTS}`",
        "- **학습·수집·외부데이터 없음.** SCADA 만, `actual_kwh` 미사용",
        "",
        "## 1. 교정한 결함",
        "",
    ]
    for key, text in payload["defects_fixed"].items():
        lines.append(f"- **{key}** {text}")
    lines += [
        "",
        f"가정: {payload['assumption']}.",
        "",
        "## 2. 시간 해상도 효과 (H1)",
        "",
        "| group | 급경사 산포 10 분 (C55) | 급경사 산포 시간 | 평활 효과 |",
        "|---:|---:|---:|---:|",
    ]
    for g, v in per_group.items():
        lines.append(
            f"| {g} | {v['sigma_steep_10min_cycle55']:.4f} | "
            f"**{v['sigma_steep_hourly']:.4f}** | "
            f"{v['sigma_steep_hourly'] - v['sigma_steep_10min_cycle55']:+.4f} |"
        )
    lines += [
        "",
        "## 3. 전 구간 FICR 천장 (H2 · H3 · H4)",
        "",
        "| group | 유효 시간 | 터빈 | 전체 잔차 sigma | 급경사 질량 | **FICR 천장** |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for g, v in per_group.items():
        lines.append(
            f"| {g} | {v['eligible_hours']:,} | {v['turbines']} | "
            f"{v['sigma_all_eligible']:.4f} | {v['steep_gen_weight']:.1%} | "
            f"**{v['ficr_ceiling']:.4f}** |"
        )
    lines += [
        "",
        f"평균 천장 **{mean_ceiling:.4f}** / 배포 FICR {DEPLOYED_FICR} / "
        f"로컬 0.66 요구 FICR **{ficr_required:.4f}** (k={k_for_066:.4f})",
        "",
        "## 4. 그룹 1 의 bin 별 분해",
        "",
        "| 풍속 bin | 평균 출력 | 발전량 가중 | 잔차 sigma | 적중률 6% | 평균 단위 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for b in per_group[1]["bins"][::2]:
        lines.append(
            f"| {b['bin_center']:.1f} | {b['mean_power']:.3f} | {b['gen_weight']:.3f} | "
            f"{b['sigma_resid']:.4f} | {b['hit6']:.3f} | {b['unit_mean']:.3f} |"
        )
    lines += [
        "",
        "## 5. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}**",
        f"- H2 `{check['H2_expectation']}` -> **{h2}** ({mean_ceiling:.4f})",
        f"- H3 `{check['H3_expectation']}` -> **{h3}**",
        f"- H4 `{check['H4_expectation']}` -> **{h4}**",
        "",
        f"판정: **{verdict}**",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE57_FICR_CEILING",
        "node": NODE_ID, "lane": LANE,
        "decided_utc": datetime.now(UTC).isoformat(),
        "result": payload,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "report_sha256": hashlib.sha256(REPORT_MD.read_bytes()).hexdigest(),
        "dacon_upload": False, "external_actions": [], "model_fits": 0,
        "lockbox_reopened": False, "new_2024_evaluation": False,
    }
    RECEIPT.write_text(
        json.dumps(receipt, indent=1, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    for g, v in per_group.items():
        print(f"[C57] g{g} 유효시간 {v['eligible_hours']:,}  "
              f"급경사 산포 10분 {v['sigma_steep_10min_cycle55']:.4f} -> "
              f"시간 {v['sigma_steep_hourly']:.4f}  "
              f"**FICR 천장 {v['ficr_ceiling']:.4f}**")
    print(f"[C57] 평균 천장 {mean_ceiling:.4f} | 배포 {DEPLOYED_FICR} | "
          f"로컬 0.66 요구 {ficr_required:.4f}")
    print(f"[C57] H1 {h1} | H2 {h2} | H3 {h3} | H4 {h4}")
    print(f"[C57] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
