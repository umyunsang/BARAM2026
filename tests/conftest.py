from pathlib import Path

import pytest

from baram.config import ProjectConfig, load_config


@pytest.fixture(scope="session")
def config() -> ProjectConfig:
    return load_config(Path("configs/default.yaml"))
