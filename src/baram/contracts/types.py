"""Frozen value objects shared by all pipeline modules."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

GroupId = Literal[1, 2, 3]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SourceSpec:
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("source identity requires a lowercase 64-character SHA-256")


@dataclass(frozen=True)
class FoldSpec:
    fold_id: str
    train_batches: tuple[str, ...]
    validation_batches: tuple[str, ...]
    eligible_groups: tuple[GroupId, ...]
    official_total_eligible: bool
    is_lockbox: bool = False


@dataclass(frozen=True)
class GroupScore:
    nmae: float
    ficr: float
    valid_rows: int
    settlement_tier_counts: Mapping[str, int]


@dataclass(frozen=True)
class OfficialScore:
    total: float
    one_minus_nmae: float
    ficr: float
    group_nmae: Mapping[GroupId, float]
    group_ficr: Mapping[GroupId, float]
    valid_rows: Mapping[GroupId, int]
    settlement_tier_counts: Mapping[GroupId, Mapping[str, int]]


@dataclass(frozen=True)
class DataManifest:
    source_sha256: str
    members: tuple[str, ...]
    member_sizes: Mapping[str, int]
    member_crc32: Mapping[str, int]
    timezone: str


@dataclass(frozen=True)
class QualityReceipt:
    findings_sha256: str
    quarantined: tuple[str, ...]
    can_build_primary_features: bool


@dataclass(frozen=True)
class FeatureManifest:
    fold_id: str
    feature_names: tuple[str, ...]
    training_rows_sha256: str
    artifact_sha256: str


@dataclass(frozen=True)
class ModelManifest:
    model_id: str
    family: str
    fold_id: str
    feature_manifest_sha256: str
    training_rows_sha256: str
    params_sha256: str
    seed: int


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    source_sha256: str
    code_sha256: str
    split_sha256: str
    feature_sha256: str
    model_sha256: str
    prediction_sha256: str
    metric_sha256: str
    environment_sha256: str
    seed: int
    runtime_seconds: float
    hardware_tier: str


@dataclass(frozen=True)
class V2StageManifest:
    """Immutable accounting record for one approved v2 experiment stage."""

    run_id: str
    stage: str
    slot_limit: int
    slots_used: int
    slots_remaining: int
    input_hashes: Mapping[str, str]
    config_sha256: str
    code_sha256: str
    output_hashes: Mapping[str, str]
    runtime_seconds: float
    seed: int
    worker_count: int
    lockbox_sha256: str

    def __post_init__(self) -> None:
        if not self.run_id or not self.stage:
            raise ValueError("v2 stage identity cannot be empty")
        if self.slot_limit < 0 or not 0 <= self.slots_used <= self.slot_limit:
            raise ValueError("v2 stage slots exceed the frozen limit")
        if self.slots_remaining != self.slot_limit - self.slots_used:
            raise ValueError("v2 stage remaining slots accounting is inconsistent")
        if not 1 <= self.worker_count <= 6:
            raise ValueError("v2 stage worker count must be between 1 and 6")
        if self.runtime_seconds < 0:
            raise ValueError("v2 stage runtime cannot be negative")


@dataclass(frozen=True)
class V2PolicyManifest:
    """Frozen lineage shared by v2 distribution, decision, and ensemble policies."""

    policy_id: str
    policy_kind: Literal["distribution", "expected_utility", "ensemble"]
    parent_hashes: Mapping[str, str]
    configuration_sha256: str
    training_rows_sha256: str
    metric_sha256: str


@dataclass(frozen=True)
class PromotionDecision:
    accepted: bool
    gate: str
    reasons: tuple[str, ...]
    deltas: Mapping[str, float]


@dataclass(frozen=True)
class CalibrationPolicy:
    group_id: GroupId
    scale: float
    offset_capacity: float
    cap_mode: Literal["capacity", "1.01_capacity", "nonnegative_only"]
    training_rows_sha256: str
    parent_model_ids: tuple[str, ...]
    input_prediction_sha256: str
    metric_sha256: str


@dataclass(frozen=True)
class BlendPolicy:
    weights_by_group: Mapping[GroupId, Mapping[str, float]]
    training_rows_sha256: str
    input_prediction_hashes: Mapping[str, str]
    metric_sha256: str


@dataclass(frozen=True)
class ResidualUtilityPolicy:
    shifts_by_state: Mapping[str, float]
    min_residuals_per_group: int
    training_rows_sha256: str
    input_prediction_hashes: Mapping[str, str]
    metric_sha256: str


@dataclass(frozen=True)
class CandidateFreeze:
    freeze_id: str
    candidate_policy_hashes: tuple[str, ...]
    lineage_hashes: Mapping[str, str]
    configuration_slots_used: int
    finalist_seed_runs: int


@dataclass(frozen=True)
class LockboxReceipt:
    candidate_freeze_sha256: str
    consumed_lock_sha256: str
    candidate_scores: Mapping[str, OfficialScore]
    champion_policy_sha256: str
    metric_sha256: str


@dataclass(frozen=True)
class FinalModelReceipt:
    model_artifact_sha256: str
    champion_policy_sha256: str
    source_sha256: str
    feature_manifest_sha256: str
    training_rows_sha256: str
    environment_sha256: str
    code_sha256: str
    seed: int


@dataclass(frozen=True)
class SubmissionReceipt:
    candidate_id: str
    csv_sha256: str
    row_count: int
    source_sha256: str
    champion_policy_sha256: str
    sample_keys_sha256: str
    encoding: Literal["utf-8-sig"]
