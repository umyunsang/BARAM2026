"""YAML configuration loading with fail-closed validation."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

from baram.contracts.types import GroupId, SourceSpec
from baram.exceptions import ContractError


@dataclass(frozen=True)
class ProjectConfig:
    repo_root: Path
    open_zip: SourceSpec
    baseline_notebook: SourceSpec
    capacities: Mapping[GroupId, float]
    seed: int
    n_jobs: int
    artifact_budget_gib: int
    lockbox_year: int


def _absolute_path(value: object, field: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        raise ContractError(f"{field} must be an absolute path")
    return path


def _source_spec(value: object, field: str) -> SourceSpec:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ContractError(f"{field} must contain only path and sha256")
    try:
        return SourceSpec(
            path=_absolute_path(value["path"], f"{field}.path"),
            sha256=str(value["sha256"]),
        )
    except ValueError as error:
        raise ContractError(str(error)) from error


def load_config(path: Path) -> ProjectConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ContractError(f"cannot read configuration: {error}") from error
    if not isinstance(raw, dict):
        raise ContractError("configuration root must be a mapping")
    try:
        capacities_raw = raw["capacities"]
        if not isinstance(capacities_raw, dict):
            raise ContractError("capacities must be a mapping")
        capacities = {int(key): float(value) for key, value in capacities_raw.items()}
        if set(capacities) != {1, 2, 3} or any(value <= 0 for value in capacities.values()):
            raise ContractError("capacities must be positive for groups 1, 2, and 3")
        repo_root = _absolute_path(raw["repo_root"], "repo_root")
        n_jobs = int(raw["n_jobs"])
        artifact_budget_gib = int(raw["artifact_budget_gib"])
        if not 1 <= n_jobs <= 6:
            raise ContractError("n_jobs must be between 1 and 6")
        if artifact_budget_gib <= 0:
            raise ContractError("artifact_budget_gib must be positive")
        return ProjectConfig(
            repo_root=repo_root,
            open_zip=_source_spec(raw["open_zip"], "open_zip"),
            baseline_notebook=_source_spec(raw["baseline_notebook"], "baseline_notebook"),
            capacities=MappingProxyType(capacities),  # type: ignore[arg-type]
            seed=int(raw["seed"]),
            n_jobs=n_jobs,
            artifact_budget_gib=artifact_budget_gib,
            lockbox_year=int(raw["lockbox_year"]),
        )
    except KeyError as error:
        raise ContractError(f"missing configuration field: {error.args[0]}") from error
    except (TypeError, ValueError) as error:
        raise ContractError(f"invalid configuration value: {error}") from error
