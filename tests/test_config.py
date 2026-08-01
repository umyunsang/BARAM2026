from pathlib import Path

import pytest

from baram.config import load_config
from baram.exceptions import ContractError


def test_default_config_loads_frozen_runtime_contract() -> None:
    """Catches drift in source identity, capacity, seed, or resource limits."""
    config = load_config(Path("configs/default.yaml"))
    assert config.repo_root == Path("/Users/um-yunsang/BARAM2026")
    assert config.open_zip.sha256 == (
        "920be0c458d820e855bf79dd25723146f52ce1736138aedba5e6bc853f1f720b"
    )
    assert config.baseline_notebook.sha256 == (
        "712b26f4d2748860c94cff1e0100c23810468c983173f8e9ef8d009fe82df48c"
    )
    assert dict(config.capacities) == {1: 21600.0, 2: 21600.0, 3: 21000.0}
    assert (config.seed, config.n_jobs, config.artifact_budget_gib, config.lockbox_year) == (
        20260801,
        6,
        10,
        2024,
    )


def test_config_rejects_relative_source_path(tmp_path: Path) -> None:
    """Catches ambiguous source resolution across working directories."""
    config = tmp_path / "bad.yaml"
    config.write_text(
        """
repo_root: /tmp/project
open_zip:
  path: relative.zip
  sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
baseline_notebook:
  path: /tmp/base.ipynb
  sha256: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
capacities: {1: 21600, 2: 21600, 3: 21000}
seed: 1
n_jobs: 1
artifact_budget_gib: 1
lockbox_year: 2024
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="absolute"):
        load_config(config)
