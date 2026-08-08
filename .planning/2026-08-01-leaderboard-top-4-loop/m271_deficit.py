"""M271 결손 원장 — 루프의 연료.

A7 이 초기화한 셀별 손실 회계를 상태 채널로 올린다. 하드코딩 후보 목록과 달리 이 원장은
**생성적**이다. 분해축을 추가하거나 잔차를 재귀속할 때마다 새 셀이 생기고, 기전이 없는 셀은
C1 을 발화시켜 새 노드를 낳는다. 미설명 손실질량이 임계 아래로 내려가기 전에는 소진되지
않는다.

A7 이 수치로 확인한 성질을 여기서 그대로 쓴다.

    공식 Total 은 임의의 행 분할에 대해 정확히 가법 분해된다 (잔차 5.551e-17).

따라서 셀 손실의 합은 항상 `1 - Total` 과 같아야 하고, 축을 추가해도 그 항등식이 유지된다.
유지되지 않으면 분해가 깨진 것이므로 오류로 처리한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPORTS = Path(__file__).resolve().parents[2] / "reports"
A7_RECEIPT = REPORTS / "m271_n0_deficit_init_receipt.json"

# 셀 상태
UNEXPLAINED = "UNEXPLAINED"  # 기전 없음 -> C1
EXPLAINED = "EXPLAINED"  # 기전 있음, 회수 시도 전
RECOVERING = "RECOVERING"  # 회수 실험 진행 중
CLOSED = "CLOSED"  # 정보량 미달로 종결 -> C8

IDENTITY_TOLERANCE = 1e-9


class LedgerIdentityBroken(RuntimeError):
    """셀 손실 합이 `1 - Total` 과 어긋났다. 분해가 깨진 것이다."""


@dataclass
class DeficitLedger:
    total: float
    target: float
    cells: dict[str, dict[str, Any]] = field(default_factory=dict)
    axes: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- 생성
    @classmethod
    def from_a7(cls, receipt: Path = A7_RECEIPT) -> DeficitLedger:
        payload = json.loads(receipt.read_text(encoding="utf-8"))["result"]
        axes = list(payload["decomposition"]["axes"])
        ledger = cls(total=float(payload["official"]["total"]), target=float(payload["target"]))
        ledger.axes = axes
        # **전 셀**을 읽는다. 상위 N 만 읽고 나머지를 하나로 접으면 그 접힌 덩어리가 최대
        # 셀이 되어 우선순위를 왜곡한다. P4 부트스트랩이 실제로 그 왜곡을 냈다.
        rows = payload.get("cells")
        if rows is None:
            raise KeyError(
                "A7 receipt has no full cell table. Re-run m271_n0_deficit_init.py; "
                "loading only top_cells distorts the ledger."
            )
        for cell in rows:
            key = cls._key(cell, axes)
            ledger.cells[key] = {
                "key": key,
                "axes": {a: cell[a] for a in axes},
                "rows": int(cell["rows"]),
                "loss_share": float(cell["total_loss"]),
                "ficr_loss": float(cell["ficr_loss"]),
                "nmae_loss": float(cell["nmae_loss"]),
                "gen_weight": float(cell["w_gen"]),
                "row_weight": float(cell["w_rows"]),
                "mean_unit": float(cell["ubar"]),
                "mean_abs_err_rate": float(cell["mean_abs_err_rate"]),
                "mechanism": None,
                "recoverable_estimate": None,
                "status": UNEXPLAINED,
                "owner": None,
            }
        ledger.assert_identity()
        return ledger

    @staticmethod
    def _key(cell: dict[str, Any], axes: list[str]) -> str:
        return "|".join(f"{a}={cell[a]}" for a in axes)

    # ---------------------------------------------------------------- 항등식
    def implied_loss(self) -> float:
        return 1.0 - self.total

    def loss_sum(self) -> float:
        return sum(float(c["loss_share"]) for c in self.cells.values())

    def assert_identity(self) -> None:
        residual = self.loss_sum() - self.implied_loss()
        if abs(residual) > IDENTITY_TOLERANCE:
            raise LedgerIdentityBroken(
                f"cell loss sum {self.loss_sum():.15f} != 1-Total "
                f"{self.implied_loss():.15f} (residual {residual:.3e})"
            )

    # ---------------------------------------------------------------- 생성적 성질
    def refine_axis(self, cell_key: str, axis: str, parts: dict[str, float]) -> list[str]:
        """한 셀을 새 축으로 쪼갠다. 손실 질량은 보존된다.

        이것이 원장을 생성적으로 만드는 연산이다. 쪼갠 조각은 기전이 없으므로 `UNEXPLAINED`
        로 태어나고, 그만큼 C1 발화 대상이 늘어난다. 원장은 고갈되지 않는다.
        """
        if cell_key not in self.cells:
            raise KeyError(f"unknown cell: {cell_key}")
        parent = self.cells[cell_key]
        weight_sum = sum(parts.values())
        if abs(weight_sum - 1.0) > 1e-9:
            raise ValueError(f"refinement weights must sum to 1, got {weight_sum}")

        created: list[str] = []
        for label, share in parts.items():
            key = f"{cell_key}|{axis}={label}"
            self.cells[key] = {
                **{k: v for k, v in parent.items() if k not in {"key", "axes"}},
                "key": key,
                "axes": {**parent["axes"], axis: label},
                "loss_share": parent["loss_share"] * share,
                "ficr_loss": parent["ficr_loss"] * share,
                "nmae_loss": parent["nmae_loss"] * share,
                "rows": int(parent["rows"] * share),
                "mechanism": None,
                "recoverable_estimate": None,
                "status": UNEXPLAINED,
                "owner": None,
            }
            created.append(key)
        del self.cells[cell_key]
        if axis not in self.axes:
            self.axes.append(axis)
        self.assert_identity()
        return sorted(created)

    # ---------------------------------------------------------------- 조회
    def unexplained(self) -> list[str]:
        return sorted(k for k, c in self.cells.items() if c["status"] == UNEXPLAINED)

    def unexplained_mass(self) -> float:
        return sum(
            float(c["loss_share"]) for c in self.cells.values() if c["status"] == UNEXPLAINED
        )

    def residual_mass(self) -> float:
        """아직 회수되지 않은 손실 질량 (CLOSED 를 제외한 전부)."""
        return sum(
            float(c["loss_share"]) for c in self.cells.values() if c["status"] != CLOSED
        )

    def gap_to_target(self) -> float:
        return self.target - self.total

    # ---------------------------------------------------------------- 순위
    def compute_efficiency(self) -> None:
        """셀별 초과비율과 회수가능질량을 채운다.

        사이클 3 이 실측한 결함의 수정이다. 원시 `loss_share` 로 순위를 매기면 **질량이 큰**
        셀이 위로 오는데, 고발전 셀은 발전량 질량이 커서 절대 손실액이 클 뿐 단위 발전량당
        효율은 오히려 좋다. 실제로 사이클 1·2 가 그렇게 잘못된 대역을 팠다.

        초과비율만 봐도 안 된다 — 미세한 셀이 큰 비율을 갖는 쪽으로 끌려간다.

        올바른 기준은 **회수가능질량** 이다.

            excess_i    = loss_i / (그 셀이 평균 효율일 때의 손실)
            recoverable = loss_i * (1 - 1/excess_i)

        즉 그 셀이 평균까지만 올라갔을 때 사라지는 손실이다. 질량과 비효율을 함께 담는다.
        """
        gen_weight = sum(float(c["gen_weight"]) for c in self.cells.values())
        row_weight = sum(float(c.get("row_weight", 0.0)) for c in self.cells.values())
        if gen_weight <= 0:
            raise ValueError("ledger has no generation weight; cannot rank by efficiency")

        # 평균 효율: 전 셀을 합쳐 본 단위 손실.
        gen_loss = sum(float(c["ficr_loss"]) for c in self.cells.values())
        row_loss = sum(float(c["nmae_loss"]) for c in self.cells.values())
        avg_ficr_per_gen = gen_loss / gen_weight
        avg_nmae_per_row = row_loss / row_weight if row_weight > 0 else 0.0

        for cell in self.cells.values():
            expected = (
                float(cell["gen_weight"]) * avg_ficr_per_gen
                + float(cell.get("row_weight", 0.0)) * avg_nmae_per_row
            )
            loss = float(cell["loss_share"])
            excess = loss / expected if expected > 0 else float("nan")
            cell["expected_loss_if_average"] = expected
            cell["excess_ratio"] = excess
            cell["recoverable_if_average"] = (
                loss - expected if excess == excess and excess > 1.0 else 0.0
            )

    def top(self, n: int = 10, by: str = "recoverable") -> list[dict[str, Any]]:
        """기본 순위는 **회수가능질량**이다. 원시 손실액 순위는 명시적으로 요청해야 한다."""
        if by == "loss":
            return sorted(self.cells.values(), key=lambda c: -float(c["loss_share"]))[:n]
        if "recoverable_if_average" not in next(iter(self.cells.values())):
            self.compute_efficiency()
        return sorted(
            self.cells.values(), key=lambda c: -float(c["recoverable_if_average"])
        )[:n]

    def to_channel(self) -> dict[str, dict[str, Any]]:
        """상태 채널에 올릴 형태. 셀 단위 쓰기 소유권이 여기서 강제된다."""
        return {k: dict(v) for k, v in self.cells.items()}
