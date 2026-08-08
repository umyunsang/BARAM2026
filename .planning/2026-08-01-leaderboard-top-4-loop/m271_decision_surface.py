"""M271 — 결정층 실험용 확률면 캐시.

사이클 56·58·59·60 이 모두 **같은 확률행렬**을 만들고 그 뒤 결정층만 바꿔 비교했다.
그 앞단(teacher 적합 72 회 + 분류기 3 회)이 매번 75 분씩 든다. C60 이 결정층에 큰
자유도가 남아 있음을 보였으므로 이 축은 계속 팔 것이고, 그러면 앞단 재계산이
반복 비용의 전부가 된다.

**계산을 바꾸지 않는다.** 앞단은 시드·스레드가 고정된 결정적 함수이므로 한 번 만들어
저장하고 다시 읽는다. 캐시 키는 앞단을 결정하는 모든 것의 digest 다 — 하나라도 바뀌면
키가 바뀌어 캐시가 빗나가고 다시 계산한다. 조용히 낡은 캐시를 쓰는 일이 없다.

동일성은 주장이 아니라 검증한다: `--verify` 는 캐시를 무시하고 다시 계산해
확률행렬이 **바이트 단위로** 같은지 본다.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m271_cycle37_band_loss import KEYS, PROBE, fold_rows
from m271_cycle40_band_classifier import (
    CLASS_WIDTH,
    N_CLASS,
    PARAMS,
    ROUNDS,
    make_objective,
    one_hot_targets,
)
from m271_cycle42_teacher_restored import (
    TEACHER_PARAMS,
    TEACHER_SEED,
    all_weather_columns,
    teach,
)
from m271_cycle56_measured_powercurve import add_sitewind_with_basis, measured_curves
from run_sequence_classifier import _surface

from baram.constants import CAPACITIES_KWH

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "artifacts" / "cache" / "m271_decision_surface"
# v2 — 확률행렬에 더해 teacher 의 풍속 추정과 관측 나셀풍속을 함께 저장한다.
# 사이클 65 가 등분산 가정(sigma_v = 1.40 상수)으로 양끝을 반대 방향으로 틀렸고,
# sigma_v(v) 를 **독립적으로** 재려면 이 둘이 있어야 한다. 관측 적중률에서 역산하면
# 순환논법이 되므로 캐시를 넓히는 쪽이 맞다.
#
# v3 — `sitewind__mean` 은 legacy 와 allweather 의 **단순 50/50 평균**이다. C70 이
# 그 평균의 잔차(1.496/1.600/1.657)가 C54 가 잰 allweather 단독(1.373/1.502/1.466)보다
# 나쁘다는 것을 드러냈다 — g3 에서 13%. 가중치가 최적이 아니라는 뜻이므로 두 성분을
# 따로 저장해 직접 잰다.
SURFACE_VERSION = "M271_DECISION_SURFACE_v3"
BASIS = "generic"


def _spec_digest(base_features: list[str]) -> str:
    """앞단을 결정하는 모든 것. 하나라도 바뀌면 캐시가 빗나간다."""
    spec = {
        "version": SURFACE_VERSION,
        "basis": BASIS,
        "class_width": CLASS_WIDTH,
        "n_class": N_CLASS,
        "rounds": ROUNDS,
        "params": {k: v for k, v in sorted(PARAMS.items()) if k != "objective"},
        "teacher_params": dict(sorted(TEACHER_PARAMS.items())),
        "teacher_seed": TEACHER_SEED,
        "folds": sorted(fold_rows()),
        "base_features": base_features,
    }
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:24]


def _compute() -> tuple[dict[str, dict[str, Any]], str]:
    curves = measured_curves()

    surface, _base, auxiliary = _surface()
    surface["forecast_kst_dtm"] = pd.to_datetime(surface["forecast_kst_dtm"])
    surface["capacity"] = surface["group_id"].map(CAPACITIES_KWH).astype(float)
    surface["rate"] = surface["actual_kwh"] / surface["capacity"]
    surface = surface.loc[surface["rate"].notna()].reset_index(drop=True)

    wanted = json.loads(
        (PROBE / "M115_XGBOOST-dev-2023-Q3.json").read_text(encoding="utf-8")
    )["selected_feature_names"]
    base_features = [c for c in wanted if c in surface.columns and c != "scada_ws"]
    aux_cols = [c for c in auxiliary if c in surface.columns and c != "scada_ws"]
    aw_cols = all_weather_columns(surface)
    digest = _spec_digest(base_features)

    store: dict[str, dict[str, Any]] = {}
    for probe_fold, meta in fold_rows().items():
        train = surface.loc[surface["forecast_kst_dtm"] < meta["start"]].copy()
        test = surface.loc[
            np.array(
                [
                    (fid, gid) in meta["keys"]
                    for fid, gid in zip(surface["forecast_id"], surface["group_id"],
                                        strict=True)
                ]
            )
        ].copy()
        legacy_tr, legacy_te = teach(train, test, aux_cols)
        aw_tr, aw_te = teach(train, test, aw_cols)

        rate = np.clip(train["rate"].to_numpy(dtype="float64"), 0.0, None)
        label = np.clip((rate / CLASS_WIDTH).astype(int), 0, N_CLASS - 1)
        target = one_hot_targets(rate)

        names = add_sitewind_with_basis(train, legacy_tr, aw_tr, BASIS, curves)
        add_sitewind_with_basis(test, legacy_te, aw_te, BASIS, curves)
        features = [*base_features, *names]
        dataset = lgb.Dataset(
            train.loc[:, features].astype("float32"), label=label, free_raw_data=False
        )
        params = dict(PARAMS)
        params["objective"] = make_objective(target)
        booster = lgb.train(params, dataset, num_boost_round=ROUNDS)
        raw = np.asarray(
            booster.predict(test.loc[:, features].astype("float32"))
        ).reshape(len(test), N_CLASS)
        exp = np.exp(raw - raw.max(axis=1, keepdims=True))
        store[probe_fold] = {
            "meta": test.loc[:, [*KEYS, "actual_kwh"]].copy(),
            "capacity": test["capacity"].to_numpy(dtype="float64"),
            "group": test["group_id"].to_numpy(),
            "probability": exp / exp.sum(axis=1, keepdims=True),
            # sigma_v(v) 를 독립 계측하기 위한 두 열. `sitewind__mean` 은 teacher 가
            # 편성한 풍속 추정이고 `scada_ws` 는 관측 나셀풍속이다. 후자는 평가기간에
            # 없으므로 **진단 전용**이며 어떤 후보의 피처도 될 수 없다(C1N39).
            "sitewind": test["sitewind__mean"].to_numpy(dtype="float64"),
            "sitewind_legacy": test["sitewind__legacy"].to_numpy(dtype="float64"),
            "sitewind_allweather": test["sitewind__allweather"].to_numpy(dtype="float64"),
            "scada_ws": test["scada_ws"].to_numpy(dtype="float64"),
        }
    return store, digest


def probability_digest(store: dict[str, dict[str, Any]]) -> str:
    return hashlib.sha256(
        b"".join(store[f]["probability"].tobytes() for f in sorted(store))
    ).hexdigest()[:16]


def _save(store: dict[str, dict[str, Any]], digest: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = CACHE_DIR / digest
    target.mkdir(exist_ok=True)
    for fold, entry in store.items():
        entry["meta"].to_parquet(target / f"{fold}__meta.parquet")
        np.savez_compressed(
            target / f"{fold}__arrays.npz",
            probability=entry["probability"],
            capacity=entry["capacity"],
            group=entry["group"],
            sitewind=entry["sitewind"],
            sitewind_legacy=entry["sitewind_legacy"],
            sitewind_allweather=entry["sitewind_allweather"],
            scada_ws=entry["scada_ws"],
        )
    (target / "manifest.json").write_text(
        json.dumps(
            {
                "version": SURFACE_VERSION,
                "spec_digest": digest,
                "probability_digest": probability_digest(store),
                "folds": sorted(store),
                "rows": {f: int(len(store[f]["capacity"])) for f in sorted(store)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return target


def _load(digest: str) -> dict[str, dict[str, Any]] | None:
    target = CACHE_DIR / digest
    manifest = target / "manifest.json"
    if not manifest.exists():
        return None
    info = json.loads(manifest.read_text(encoding="utf-8"))
    store: dict[str, dict[str, Any]] = {}
    for fold in info["folds"]:
        arrays = np.load(target / f"{fold}__arrays.npz")
        store[fold] = {
            "meta": pd.read_parquet(target / f"{fold}__meta.parquet"),
            "capacity": arrays["capacity"],
            "group": arrays["group"],
            "probability": arrays["probability"],
            "sitewind": arrays["sitewind"],
            "sitewind_legacy": arrays["sitewind_legacy"],
            "sitewind_allweather": arrays["sitewind_allweather"],
            "scada_ws": arrays["scada_ws"],
        }
    if probability_digest(store) != info["probability_digest"]:
        raise RuntimeError(f"캐시 손상: {target}")
    return store


def load_surface(force: bool = False) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """확률면을 얻는다. 캐시가 맞으면 읽고 아니면 계산 후 저장한다."""
    store: dict[str, dict[str, Any]] | None = None
    digest: str | None = None
    if not force:
        # 캐시 키가 base_features 에 의존하므로 표면을 한 번 만들어 digest 만 구한다.
        # 적합이 없으므로 앞단 전체(75 분)에 비하면 무시할 만하다.
        probe, _base, _aux = _surface()
        wanted = json.loads(
            (PROBE / "M115_XGBOOST-dev-2023-Q3.json").read_text(encoding="utf-8")
        )["selected_feature_names"]
        digest = _spec_digest(
            [c for c in wanted if c in probe.columns and c != "scada_ws"]
        )
        del probe
        store = _load(digest)

    if store is None:
        store, digest = _compute()
        _save(store, digest)
        cached = False
    else:
        cached = True
    return store, {
        "spec_digest": digest,
        "probability_digest": probability_digest(store),
        "from_cache": cached,
    }


def main() -> int:
    verify = "--verify" in sys.argv
    store, info = load_surface(force=False)
    print(f"[surface] spec {info['spec_digest']} / prob {info['probability_digest']} "
          f"/ 캐시 {info['from_cache']}")
    if verify:
        fresh, fresh_digest = _compute()
        same = probability_digest(fresh) == info["probability_digest"]
        print(f"[surface] 재계산 spec {fresh_digest} / prob "
              f"{probability_digest(fresh)} / 바이트 동일 **{same}**")
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
