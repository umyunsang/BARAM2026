from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import pytest

from baram.constants import EXPECTED_ARCHIVE_MEMBERS
from baram.contracts.hashing import sha256_file
from baram.data.archive import read_csv_member, read_info_workbook, validate_archive
from baram.exceptions import ContractError, DataQualityError


def _write_fixture_zip(path: Path, extra: str | None = None) -> None:
    with ZipFile(path, "w", compression=ZIP_STORED) as archive:
        for member in sorted(EXPECTED_ARCHIVE_MEMBERS):
            content = b"" if member.endswith("/") else b"fixture"
            archive.writestr(member, content)
        if extra is not None:
            archive.writestr(extra, b"unexpected")


def test_archive_identity_and_members(config) -> None:
    """Catches source substitution, missing members, and member-name drift."""
    manifest = validate_archive(config.open_zip.path, config.open_zip.sha256)
    assert set(manifest.members) == EXPECTED_ARCHIVE_MEMBERS
    assert manifest.source_sha256 == config.open_zip.sha256
    assert manifest.timezone == "Asia/Seoul"
    assert manifest.member_sizes["train/train_labels.csv"] > 0


def test_archive_rejects_wrong_expected_hash(config) -> None:
    """Catches use of an unapproved archive before parsing."""
    with pytest.raises(ContractError, match="SHA-256 mismatch"):
        validate_archive(config.open_zip.path, "0" * 64)


def test_archive_rejects_extra_or_unsafe_member(tmp_path: Path) -> None:
    """Catches hidden payloads and path traversal in a replacement archive."""
    extra = tmp_path / "extra.zip"
    _write_fixture_zip(extra, "surprise.txt")
    with pytest.raises(ContractError, match="member set"):
        validate_archive(extra, sha256_file(extra))

    unsafe = tmp_path / "unsafe.zip"
    _write_fixture_zip(unsafe, "../escape.csv")
    with pytest.raises(ContractError, match="unsafe"):
        validate_archive(unsafe, sha256_file(unsafe))


def test_archive_rejects_crc_failure(tmp_path: Path) -> None:
    """Catches byte corruption even when the caller supplies its new hash."""
    path = tmp_path / "corrupt.zip"
    _write_fixture_zip(path)
    with ZipFile(path) as archive:
        info = archive.getinfo("data_description.md")
        offset = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
    with path.open("r+b") as stream:
        stream.seek(offset)
        first = stream.read(1)
        stream.seek(offset)
        stream.write(bytes([first[0] ^ 0xFF]))
    with pytest.raises(DataQualityError, match="CRC"):
        validate_archive(path, sha256_file(path))


def test_csv_reader_normalizes_bom_header(config) -> None:
    """Catches a BOM leaking into the first schema field."""
    frame = read_csv_member(config.open_zip.path, "sample_submission.csv")
    assert frame.columns[0] == "forecast_id"
    assert len(frame) == 8760


def test_reader_rejects_member_outside_allowlist(config) -> None:
    """Catches arbitrary ZIP member access bypassing the contract."""
    with pytest.raises(ContractError, match="allowed CSV"):
        read_csv_member(config.open_zip.path, "data_description.md")


def test_info_reader_returns_original_workbook_bytes(config) -> None:
    """Catches accidental workbook conversion or mutation during access."""
    payload = read_info_workbook(config.open_zip.path)
    assert payload.startswith(b"PK")
    assert len(payload) > 1000
