"""Frozen-source and lineage guards for final-model inference."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from baram.exceptions import ContractError
from baram.models.baselines import ModelBundle, predict_bundle


@dataclass(frozen=True)
class FinalInferenceContract:
    source_sha256: str
    champion_parent_model_ids: tuple[str, ...]
    decision_parent_model_ids: tuple[str, ...]
    fold_id: str


def _canonical_parent_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    if not values or any(not value for value in values):
        raise ContractError("final inference parent model IDs cannot be empty")
    if len(set(values)) != len(values):
        raise ContractError("final inference parent model IDs must be unique")
    return tuple(sorted(values))


def predict_final_bundle(
    bundle: ModelBundle,
    features: pd.DataFrame,
    contract: FinalInferenceContract,
    observed_source_sha256: str,
) -> np.ndarray:
    """Predict only after source, final-fold, and OOF decision lineage agree."""
    if observed_source_sha256 != contract.source_sha256:
        raise ContractError("final inference source hash differs from the frozen source")
    champion_parents = _canonical_parent_ids(contract.champion_parent_model_ids)
    decision_parents = _canonical_parent_ids(contract.decision_parent_model_ids)
    if champion_parents != decision_parents:
        raise ContractError("decision policy parent models differ from frozen champion inputs")
    if (
        bundle.manifest.fold_id != contract.fold_id
        or bundle.feature_state.fold_id != contract.fold_id
    ):
        raise ContractError("final model bundle does not belong to the declared final fold")
    return predict_bundle(bundle, features, contract.fold_id)
