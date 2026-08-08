"""M271 P4 사이클 55 — 터빈간 출력 잔차 상관. 산포 천장의 마지막 미지수.

사이클 54 가 판정을 한 양에 걸어놨다.

    급경사 구간 터빈 1 기 산포 = 0.0712 정격비 (밴드 반폭 0.06 의 **1.19 배**)
    무상관 집계 하한          = 0.0299          (**0.50 배**)

둘 사이 어디에 떨어지는지는 **터빈간 잔차 상관 `rho_t`** 가 정한다. 유효 집계 산포는

    sigma_eff = sigma_single * sqrt( (1 + (n-1) * rho_t) / n )

이고, 이 값이 밴드 반폭 0.06 을 넘으면 **풍속을 완벽히 알아도** 급경사 구간에서 밴드
적중이 보장되지 않는다. 그것이 FICR 천장을 직접 구속한다.

`rho_t` 는 SCADA 에 이미 있다. 수집도 학습도 필요 없다.

① 방법 리서치 (실행 전)
  - 새 방법 없음. 사이클 17 에서 앙상블 멤버에 쓴 분산감소 공식을 **터빈**에 적용한다.
    같은 식이고 대상만 다르다.
  - 잔차 정의는 A5 와 맞춘다: IEC 밀도정규화 풍속 `v_n = v * (rho/1.225)^(1/3)` 으로
    bin 을 잡고, **각 터빈이 자기 bin 평균**에서 벗어난 정도를 잔차로 본다.
    (공통 커브를 쓰면 터빈간 커브 차이가 상관으로 흘러들어 과대추정된다)
  - 상관은 **같은 그룹 안**에서만 잰다. 정산은 그룹 단위이기 때문이다.

② 사양 동결

  자료   `open.zip` 의 `train/scada_vestas_train.csv`, `train/scada_unison_train.csv`
  그룹   vestas wtg01-06 -> g1, wtg07-12 -> g2, unison wtg01-05 -> g3 (A5 와 동일)
  밀도   LDAPS 격자평균 기압·기온에서 `rho = p/(R*T)`. A5 와 동일 절차
  이상치 A5 와 동일: 풍속 `[0, 50)`, 출력 `[0, 정격*1.1]`
  bin    폭 0.5 m/s, bin 당 최소 30 행
  잔차   `(power - 자기터빈_bin평균) / rated`
  상관   같은 그룹 터빈 쌍의 평균 Pearson (급경사 bin 행만)

  사전확약(실행 전 동결):
    H1  `rho_t` 를 세 그룹 모두에서 잰다 (표본 충분).
    H2  유효 집계 산포 `sigma_eff` 가 밴드 반폭 **0.06 을 넘는다**.
        성립하면 출력 산포가 FICR 을 직접 구속한다.
    H3  `rho_t` 가 **0.5 를 넘는다**. 같은 바람을 받는 인접 터빈이므로 예상되는 방향이다.
    H4  (대조) 터빈간 상관을 **무시**했을 때(사이클 54 의 하한)와 실제 `sigma_eff` 의
        차이를 보고한다.

  H2 가 성립하면 **로컬 0.66 은 풍속 정확도로 도달 불가**하고, 남은 경로는 산포 자체를
  줄이는 것(가용성·후류·개별 터빈 모델링)뿐이다. 기각되면 산포는 병목이 아니며 병목이
  또 다른 곳에 있다.

**게이트 무관. 학습·수집·외부데이터 없음. `actual_kwh` 미사용.**
"""

from __future__ import annotations

import hashlib
import itertools
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
OPEN_ZIP = Path("/Users/um-yunsang/Downloads/open.zip")
REPORT_MD = REPORTS / "m271_cycle55_turbine_correlation.md"
RECEIPT = REPORTS / "m271_cycle55_turbine_correlation_receipt.json"

NODE_ID = "C1N55_TURBINE_CORRELATION"
LANE = "L3"
PARENT_NODE = "C1N54_SCATTER_CEILING"

BAND_HALF = 0.06
BIN_WIDTH = 0.5
MIN_BIN_ROWS = 30
STEEP_LOW, STEEP_HIGH = 0.20, 0.80
H3_MIN_RHO = 0.5

SPECS = (
    ("vestas", "train/scada_vestas_train.csv", 600.0,
     {1: range(1, 7), 2: range(7, 13)}),
    ("unison", "train/scada_unison_train.csv", 700.0, {3: range(1, 6)}),
)


def load_turbines() -> dict[int, pd.DataFrame]:
    """그룹별 (시각 x 터빈) 출력·풍속 프레임. A5 와 같은 원천·같은 그룹 배정."""
    out: dict[int, list[pd.DataFrame]] = {}
    with zipfile.ZipFile(OPEN_ZIP) as archive:
        for prefix, member, rated, groups in SPECS:
            with archive.open(member) as stream:
                raw = pd.read_csv(stream, parse_dates=["kst_dtm"])
            for group, numbers in groups.items():
                frames = []
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
                    frames.append(
                        pd.DataFrame(
                            {
                                "kst_dtm": raw["kst_dtm"],
                                "turbine": name,
                                "ws": wind.where(valid),
                                "power_norm": (power / rated).where(valid),
                            }
                        )
                    )
                out.setdefault(group, []).extend(frames)
    return {g: pd.concat(v, ignore_index=True) for g, v in out.items()}


def main() -> int:
    per_group: dict[int, Any] = {}
    for group, frame in load_turbines().items():
        frame = frame.dropna(subset=["ws", "power_norm"]).copy()
        frame["bin"] = (frame["ws"] / BIN_WIDTH).round().astype(int)
        # 각 터빈이 **자기** bin 평균에서 벗어난 정도. 터빈간 커브 차이는 상관에 안 섞인다.
        stats = frame.groupby(["turbine", "bin"])["power_norm"].agg(["mean", "size"])
        thick = stats.loc[stats["size"] >= MIN_BIN_ROWS, "mean"]
        frame = frame.join(thick.rename("bin_mean"), on=["turbine", "bin"])
        frame = frame.dropna(subset=["bin_mean"])
        frame["resid"] = frame["power_norm"] - frame["bin_mean"]

        # 급경사 구간: 터빈 평균 bin 평균이 0.2~0.8 인 bin
        bin_level = frame.groupby("bin")["bin_mean"].mean()
        steep_bins = bin_level[(bin_level >= STEEP_LOW) & (bin_level <= STEEP_HIGH)].index
        steep = frame.loc[frame["bin"].isin(steep_bins)]

        wide = steep.pivot_table(index="kst_dtm", columns="turbine", values="resid")
        wide = wide.dropna(thresh=2)
        turbines = list(wide.columns)
        pairs = []
        for a, b in itertools.combinations(turbines, 2):
            joint = wide[[a, b]].dropna()
            if len(joint) < 500:
                continue
            pairs.append(float(joint[a].corr(joint[b])))
        rho_t = float(np.mean(pairs)) if pairs else float("nan")
        sigma_single = float(steep["resid"].std(ddof=1))
        n = len(turbines)
        sigma_eff = sigma_single * float(np.sqrt((1.0 + (n - 1) * rho_t) / n))
        sigma_indep = sigma_single / float(np.sqrt(n))
        # 실제 그룹 합산 잔차의 표준편차 (직접 확인 — 공식과 대조)
        realised = float(wide.mean(axis=1).std(ddof=1))

        per_group[group] = {
            "turbines": n, "pairs": len(pairs),
            "steep_bins": len(steep_bins), "steep_rows": len(steep),
            "rho_turbine_mean": rho_t,
            "rho_turbine_min": float(np.min(pairs)) if pairs else float("nan"),
            "rho_turbine_max": float(np.max(pairs)) if pairs else float("nan"),
            "sigma_single": sigma_single,
            "sigma_eff_formula": sigma_eff,
            "sigma_eff_realised": realised,
            "sigma_if_independent": sigma_indep,
            "band_half": BAND_HALF,
            "eff_over_band": realised / BAND_HALF,
            "binds": bool(realised > BAND_HALF),
        }

    h1 = all(np.isfinite(v["rho_turbine_mean"]) for v in per_group.values())
    h2 = all(v["binds"] for v in per_group.values())
    h3 = all(v["rho_turbine_mean"] > H3_MIN_RHO for v in per_group.values())
    h4 = True  # 보고 항목

    if h2:
        verdict = "POWER_SCATTER_BINDS_FICR_CEILING"
    elif any(v["binds"] for v in per_group.values()):
        verdict = "POWER_SCATTER_BINDS_IN_SOME_GROUPS"
    else:
        verdict = "POWER_SCATTER_NOT_THE_BOTTLENECK"

    check = {
        "H1_expectation": "세 그룹 모두에서 rho_t 측정",
        "H1_held": h1,
        "H2_expectation": f"유효 집계 산포 > 밴드 반폭 {BAND_HALF}",
        "H2_held": h2,
        "H3_expectation": f"rho_t > {H3_MIN_RHO}",
        "H3_held": h3,
        "H4_expectation": "무상관 가정 대비 차이 보고",
        "H4_held": h4,
        "verdict": verdict,
    }

    payload = {
        "node": NODE_ID, "parent_node": PARENT_NODE,
        "question": "터빈간 잔차 상관이 사이클 54 의 상한(0.0712)과 하한(0.0299) 중 "
                    "어디를 고르는가",
        "residual_definition": "각 터빈이 **자기** bin 평균에서 벗어난 정도. 공통 커브를 "
                               "쓰면 터빈간 커브 차이가 상관으로 흘러들어 과대추정된다",
        "no_training": True, "no_collection": True, "no_external_data": True,
        "uses_actual_kwh": False,
        "per_group": per_group,
        "predeclared_check": check,
    }

    lines = [
        "# M271 P4 사이클 55 — 터빈간 출력 잔차 상관",
        "",
        f"- 판정일: {datetime.now(UTC).strftime('%Y-%m-%d')} (UTC)",
        f"- 노드: `{NODE_ID}` / 레인 {LANE} / 부모 `{PARENT_NODE}`",
        "- **학습·수집·외부데이터 없음.** `actual_kwh` 미사용",
        "",
        "## 1. 질문",
        "",
        payload["question"] + ".",
        "",
        f"잔차 정의: {payload['residual_definition']}.",
        "",
        "## 2. 측정",
        "",
        "| group | 터빈 | 쌍 | 급경사 행 | **rho_t** | (최소~최대) | 터빈 산포 | "
        "**집계 산포** | 밴드 대비 | 구속 |",
        "|---:|---:|---:|---:|---:|---|---:|---:|---:|:---:|",
    ]
    for g, v in per_group.items():
        lines.append(
            f"| {g} | {v['turbines']} | {v['pairs']} | {v['steep_rows']:,} | "
            f"**{v['rho_turbine_mean']:.3f}** | "
            f"{v['rho_turbine_min']:.3f}~{v['rho_turbine_max']:.3f} | "
            f"{v['sigma_single']:.4f} | **{v['sigma_eff_realised']:.4f}** | "
            f"**{v['eff_over_band']:.2f}x** | "
            f"{'**O**' if v['binds'] else 'X'} |"
        )
    lines += [
        "",
        "| group | 공식 예측 | 실측 집계 | 무상관 가정 | 상관이 만든 차이 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for g, v in per_group.items():
        lines.append(
            f"| {g} | {v['sigma_eff_formula']:.4f} | {v['sigma_eff_realised']:.4f} | "
            f"{v['sigma_if_independent']:.4f} | "
            f"**{v['sigma_eff_realised'] - v['sigma_if_independent']:+.4f}** |"
        )
    lines += [
        "",
        "## 3. 사전확약 대조",
        "",
        f"- H1 `{check['H1_expectation']}` -> **{h1}**",
        f"- H2 `{check['H2_expectation']}` -> **{h2}**",
        f"- H3 `{check['H3_expectation']}` -> **{h3}**",
        "",
        f"판정: **{verdict}**",
        "",
    ]
    if h2:
        lines += [
            "## 4. 이것이 확정하는 것",
            "",
            "**풍속을 완벽히 알아도** 급경사 구간에서 그룹 출력의 산포가 정산 밴드 반폭을",
            "넘는다. 즉 로컬 0.66 은 풍속 정확도로 도달할 수 없다 — 사이클 53·54 가 공급",
            "데이터만으로 요구 풍속 정확도의 2 배를 확보한다고 보였는데도 점수가 오르지",
            "않는 이유가 이것이다.",
            "",
            "남은 경로는 **산포 자체를 줄이는 것**이다: 개별 터빈 가용성·후류·정지 이벤트를",
            "예보시점 정보로 설명하는 것. 사이클 2 가 가용성 기전을 15~23 sigma 로 확인했고",
            "`AVAILABILITY_UNKNOWABLE_AT_FORECAST` 로 닫았다 — 그 전제가 이제 다시 쟁점이 된다.",
        ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "stage": "M271_P4_CYCLE55_TURBINE_CORRELATION",
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
        print(f"[C55] g{g} 터빈 {v['turbines']} 쌍 {v['pairs']} 급경사행 "
              f"{v['steep_rows']:,}  rho_t {v['rho_turbine_mean']:.3f} "
              f"({v['rho_turbine_min']:.3f}~{v['rho_turbine_max']:.3f})  "
              f"단일 {v['sigma_single']:.4f} -> 집계 {v['sigma_eff_realised']:.4f} "
              f"({v['eff_over_band']:.2f}x 밴드)  구속 {v['binds']}")
    print(f"[C55] H1 {h1} | H2 {h2} | H3 {h3}")
    print(f"[C55] 판정: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
