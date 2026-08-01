"""Read-only validation and member access for the supplied competition ZIP."""

from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

import pandas as pd

from baram.constants import CSV_MEMBERS, EXPECTED_ARCHIVE_MEMBERS, TIMEZONE
from baram.contracts.hashing import sha256_file
from baram.contracts.types import DataManifest
from baram.exceptions import ContractError, DataQualityError


def _assert_safe_member(member: str) -> None:
    path = PurePosixPath(member)
    if path.is_absolute() or ".." in path.parts or "\\" in member:
        raise ContractError(f"archive contains unsafe member path: {member}")


def validate_archive(path: Path, expected_sha: str) -> DataManifest:
    """Verify identity, paths, exact membership, and CRC without extracting."""
    if not path.is_file():
        raise ContractError(f"archive does not exist: {path}")
    observed_sha = sha256_file(path)
    if observed_sha != expected_sha:
        raise ContractError(
            f"archive SHA-256 mismatch: expected {expected_sha}, observed {observed_sha}"
        )
    try:
        with ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            for name in names:
                _assert_safe_member(name)
            if set(names) != EXPECTED_ARCHIVE_MEMBERS or len(names) != len(
                EXPECTED_ARCHIVE_MEMBERS
            ):
                missing = sorted(EXPECTED_ARCHIVE_MEMBERS - set(names))
                extra = sorted(set(names) - EXPECTED_ARCHIVE_MEMBERS)
                raise ContractError(
                    f"archive member set mismatch: missing={missing}, extra={extra}"
                )
            corrupt = archive.testzip()
            if corrupt is not None:
                raise DataQualityError(f"archive CRC failure: {corrupt}")
            return DataManifest(
                source_sha256=observed_sha,
                members=tuple(sorted(names)),
                member_sizes={info.filename: info.file_size for info in infos},
                member_crc32={info.filename: info.CRC for info in infos},
                timezone=TIMEZONE,
            )
    except BadZipFile as error:
        raise DataQualityError(f"archive is not a valid ZIP or has a CRC error: {error}") from error


def read_csv_member(path: Path, member: str) -> pd.DataFrame:
    """Read one allowlisted CSV using BOM-aware UTF-8 without extraction."""
    if member not in CSV_MEMBERS:
        raise ContractError(f"member is not an allowed CSV: {member}")
    try:
        with ZipFile(path) as archive, archive.open(member) as stream:
            return pd.read_csv(stream, encoding="utf-8-sig")
    except (BadZipFile, KeyError, UnicodeError) as error:
        raise DataQualityError(f"cannot read CSV member {member}: {error}") from error


def read_info_workbook(path: Path) -> bytes:
    """Return the original workbook bytes without parsing or rewriting it."""
    try:
        with ZipFile(path) as archive:
            return archive.read("info.xlsx")
    except (BadZipFile, KeyError) as error:
        raise DataQualityError(f"cannot read info.xlsx: {error}") from error


def read_description(path: Path) -> str:
    """Read the bundled UTF-8 data description for local receipts."""
    try:
        with ZipFile(path) as archive:
            return archive.read("data_description.md").decode("utf-8-sig")
    except (BadZipFile, KeyError, UnicodeError) as error:
        raise DataQualityError(f"cannot read data_description.md: {error}") from error
