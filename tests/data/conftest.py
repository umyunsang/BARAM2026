import pytest

from baram.config import ProjectConfig
from baram.data.canonical import CanonicalTables, load_canonical_tables


@pytest.fixture(scope="session")
def canonical_tables(config: ProjectConfig) -> CanonicalTables:
    return load_canonical_tables(config.open_zip.path)
