from dataclasses import replace

import pandas as pd
import pytest

from baram.contracts.types import ModelManifest
from baram.exceptions import ContractError, ModelError
from baram.features.pipeline import FeaturePipelineState
from baram.inference.final import FinalInferenceContract, predict_final_bundle
from baram.models.baselines import ModelBundle


class _Estimator:
    def predict(self, features: pd.DataFrame) -> list[float]:
        return [0.5] * len(features)


def _bundle() -> ModelBundle:
    state = FeaturePipelineState("final", ("a", "b"), {"a": 0.0, "b": 0.0}, "1" * 64)
    manifest = ModelManifest(
        "final-model",
        "fixture",
        "final",
        "2" * 64,
        "1" * 64,
        "3" * 64,
        42,
    )
    return ModelBundle(
        _Estimator(), manifest, ("a", "b"), state, 100.0, 1, True, "nonnegative_only"
    )


def _contract() -> FinalInferenceContract:
    return FinalInferenceContract("a" * 64, ("oof-a",), ("oof-a",), "final")


def test_final_prediction_obeys_feature_order_and_lineage() -> None:
    """Catches train/inference feature skew or a substituted decision parent."""
    features = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert predict_final_bundle(_bundle(), features, _contract(), "a" * 64).tolist() == [50.0, 50.0]
    with pytest.raises(ModelError, match="columns/order"):
        predict_final_bundle(_bundle(), features[["b", "a"]], _contract(), "a" * 64)
    with pytest.raises(ContractError, match="parent"):
        predict_final_bundle(
            _bundle(),
            features,
            replace(_contract(), decision_parent_model_ids=("other",)),
            "a" * 64,
        )


def test_final_prediction_rejects_source_substitution() -> None:
    """Catches inference from data other than the frozen source archive."""
    features = pd.DataFrame({"a": [1.0], "b": [2.0]})
    with pytest.raises(ContractError, match="source"):
        predict_final_bundle(_bundle(), features, _contract(), "b" * 64)
