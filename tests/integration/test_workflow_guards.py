from pathlib import Path

import pytest

from baram.exceptions import ContractError
from baram.workflows import _acquire_one_use_lock


def test_one_use_lock_is_created_exclusively_before_scoring(tmp_path: Path) -> None:
    """Catches a second lockbox evaluation overwriting the consumed receipt."""
    path = tmp_path / "lockbox.consumed.json"
    first_hash = _acquire_one_use_lock(path, {"state": "CONSUMED_BEFORE_SCORING"})
    assert len(first_hash) == 64
    original = path.read_bytes()
    with pytest.raises(ContractError, match="already been consumed"):
        _acquire_one_use_lock(path, {"state": "SECOND_ATTEMPT"})
    assert path.read_bytes() == original
