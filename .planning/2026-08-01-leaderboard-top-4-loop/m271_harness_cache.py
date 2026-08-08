"""M271 하네스 캐시 — C1N60 하네스의 fold 프레임을 1 회만 계산해 재사용한다.

**왜 필요한가.** C1N60 이후 모든 노드가 같은 앞단을 반복한다 —

    surface 적재 -> teacher(legacy) -> teacher(allweather) -> sitewind(generic)

이 앞단이 fold 당 30 회 LGBM 적합(그룹 3 x KFold 3 x 2 teacher, 보류 적합 포함)이고
`num_threads=1` 이라 노드 하나에 13~15 분이 든다. **노드 자체의 계산보다 앞단이 크다.**
루프가 앞으로 수십 번 더 돌아야 하므로 이 고정비를 제거한다.

**정확성 계약.** 캐시는 앞단을 결정하는 모든 것의 digest 로 키를 잡는다. teacher 파라미터,
시드, 기저, fold 정의, 아카이브 해시 중 하나라도 바뀌면 digest 가 바뀌어 캐시가 빗나가고
다시 계산한다. 재현 계약을 우회하는 것이 아니라 **같은 계산을 두 번 하지 않는 것**이다.

캐시 적중 여부와 digest 를 호출자가 receipt 에 기록할 수 있도록 함께 돌려준다.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m271_cycle37_band_loss import PROBE, fold_rows
from m271_cycle42_teacher_restored import (
    TEACHER_PARAMS,
    TEACHER_SEED,
    all_weather_columns,
    teach,
)
from m271_cycle56_measured_powercurve import add_sitewind_with_basis, measured_curves
from run_sequence_classifier import OPEN_SHA, _surface

from baram.constants import CAPACITIES_KWH

ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = ROOT / "artifacts/cache/harness"

HARNESS_VERSION = "M271_HARNESS_C1N60_v1"
BASIS = "generic"


def _digest(base_features: list[str]) -> str:
    spec = {
        "version": HARNESS_VERSION,
        "basis": BASIS,
        "archive_sha256": OPEN_SHA,
        "teacher_params": dict(sorted(TEACHER_PARAMS.items())),
        "teacher_seed": TEACHER_SEED,
        "folds": sorted(fold_rows()),
        "base_features": base_features,
        "capacities": dict(sorted(CAPACITIES_KWH.items())),
    }
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:24]


def _compute(base_features: list[str]) -> dict[str, dict[str, Any]]:
    curves = measured_curves()
    surface, _base, auxiliary = _surface()
    surface["forecast_kst_dtm"] = pd.to_datetime(surface["forecast_kst_dtm"])
    surface["capacity"] = surface["group_id"].map(CAPACITIES_KWH).astype(float)
    surface["rate"] = surface["actual_kwh"] / surface["capacity"]
    surface = surface.loc[surface["rate"].notna()].reset_index(drop=True)

    aux_cols = [c for c in auxiliary if c in surface.columns and c != "scada_ws"]
    aw_cols = all_weather_columns(surface)

    out: dict[str, dict[str, Any]] = {}
    for probe_fold, meta in fold_rows().items():
        train = surface.loc[surface["forecast_kst_dtm"] < meta["start"]].copy()
        test = surface.loc[
            np.array(
                [
                    (fid, gid) in meta["keys"]
                    for fid, gid in zip(
                        surface["forecast_id"], surface["group_id"], strict=True
                    )
                ]
            )
        ].copy()
        legacy_tr, legacy_te = teach(train, test, aux_cols)
        aw_tr, aw_te = teach(train, test, aw_cols)
        names = add_sitewind_with_basis(train, legacy_tr, aw_tr, BASIS, curves)
        add_sitewind_with_basis(test, legacy_te, aw_te, BASIS, curves)
        out[probe_fold] = {
            "train": train,
            "test": test,
            "sitewind_names": list(names),
        }
    return out


def fold_frames(
    base_features: list[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], str, bool]:
    """`(store, digest, cache_hit)`. `store[fold]` 는 train/test/sitewind_names."""
    if base_features is None:
        wanted = json.loads(
            (PROBE / "M115_XGBOOST-dev-2023-Q3.json").read_text(encoding="utf-8")
        )["selected_feature_names"]
        # `_surface()` 를 한 번 열어야 컬럼 교집합을 알 수 있으므로 캐시 키는
        # 원본 wanted 목록으로 잡는다. 교집합은 결정론적 함수이므로 키가 보존된다.
        base_features = [c for c in wanted if c != "scada_ws"]

    digest = _digest(base_features)
    home = CACHE_ROOT / digest
    marker = home / "MANIFEST.json"

    if marker.exists():
        manifest = json.loads(marker.read_text(encoding="utf-8"))
        store: dict[str, dict[str, Any]] = {}
        for fold in manifest["folds"]:
            store[fold] = {
                "train": pd.read_parquet(home / f"{fold}__train.parquet"),
                "test": pd.read_parquet(home / f"{fold}__test.parquet"),
                "sitewind_names": manifest["sitewind_names"][fold],
            }
        return store, digest, True

    store = _compute(base_features)
    home.mkdir(parents=True, exist_ok=True)
    for fold, entry in store.items():
        entry["train"].to_parquet(home / f"{fold}__train.parquet", index=False)
        entry["test"].to_parquet(home / f"{fold}__test.parquet", index=False)
    marker.write_text(
        json.dumps(
            {
                "version": HARNESS_VERSION,
                "digest": digest,
                "folds": sorted(store),
                "sitewind_names": {f: e["sitewind_names"] for f, e in store.items()},
                "rows": {
                    f: {"train": len(e["train"]), "test": len(e["test"])}
                    for f, e in store.items()
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return store, digest, False


def resolved_base_features(
    sample: pd.DataFrame, sitewind_names: list[str]
) -> list[str]:
    """캐시된 프레임에서 실제로 쓸 M115 기저 피처(교집합).

    **`sitewind_names` 를 반드시 빼야 한다.** 캐시된 프레임은 `add_sitewind_with_basis`
    가 **이미 적용된 뒤**라 M115 의 `sitewind__*` 항목이 컬럼으로 존재한다. 빼지 않으면
    호출자가 `[*base, *sitewind]` 로 이을 때 같은 이름이 두 번 들어가 LightGBM 이
    `Feature ... appears more than one time` 으로 거부한다. 캐시를 쓰지 않던 C1N60·N5 는
    sitewind 를 붙이기 **전**의 surface 에서 교집합을 잡았으므로 겹치지 않았다 —
    이 함수는 그 피처 목록을 그대로 재현한다.
    """
    wanted = json.loads(
        (PROBE / "M115_XGBOOST-dev-2023-Q3.json").read_text(encoding="utf-8")
    )["selected_feature_names"]
    exclude = {"scada_ws", *sitewind_names}
    return [c for c in wanted if c in sample.columns and c not in exclude]


if __name__ == "__main__":
    store, digest, hit = fold_frames()
    print(f"digest={digest} cache_hit={hit}")
    for fold in sorted(store):
        entry = store[fold]
        print(
            f"  {fold}: train={len(entry['train'])} test={len(entry['test'])} "
            f"sitewind={len(entry['sitewind_names'])}"
        )
