from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from baram.contracts.types import SourceSpec


def test_source_spec_is_frozen() -> None:
    """Catches mutation of an input identity after validation."""
    spec = SourceSpec(path=Path("/tmp/a"), sha256="a" * 64)
    with pytest.raises(FrozenInstanceError):
        spec.path = Path("/tmp/b")  # type: ignore[misc]


@pytest.mark.parametrize("value", ["short", "g" * 64])
def test_source_spec_rejects_invalid_sha256(value: str) -> None:
    """Catches malformed hashes entering lineage manifests."""
    with pytest.raises(ValueError, match="SHA-256"):
        SourceSpec(path=Path("/tmp/a"), sha256=value)
