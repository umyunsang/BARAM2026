from pathlib import Path

import pytest

from baram.config import ProjectConfig, load_config
from baram.data.canonical import CanonicalTables, load_canonical_tables


@pytest.fixture(scope="session")
def config() -> ProjectConfig:
    return load_config(Path("configs/default.yaml"))


@pytest.fixture(scope="session")
def canonical_tables(config: ProjectConfig) -> CanonicalTables:
    return load_canonical_tables(config.open_zip.path)
