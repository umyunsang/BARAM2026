"""Local ledger for user-confirmed external submission receipts only."""

import json
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from baram.exceptions import SubmissionError
from baram.experiments.registry import append_canonical_jsonl

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SubmissionEvent:
    date_kst: str
    candidate_sha256: str
    external_receipt: str
    user_confirmed: bool


def _validate_event(event: SubmissionEvent) -> None:
    try:
        date.fromisoformat(event.date_kst)
    except ValueError as error:
        raise SubmissionError("submission event requires an ISO KST date") from error
    if _SHA256.fullmatch(event.candidate_sha256) is None:
        raise SubmissionError("submission event candidate requires a lowercase SHA-256")
    if not event.external_receipt.strip():
        raise SubmissionError("submission event requires an external receipt")
    if event.user_confirmed is not True:
        raise SubmissionError("submission event requires explicit user confirmation")


def read_submission_events(ledger: Path) -> list[SubmissionEvent]:
    if not ledger.exists():
        return []
    events: list[SubmissionEvent] = []
    for line_number, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            raw = json.loads(line)
            event = SubmissionEvent(**raw)
            _validate_event(event)
        except (json.JSONDecodeError, TypeError, SubmissionError) as error:
            raise SubmissionError(
                f"submission budget ledger line {line_number} is invalid: {error}"
            ) from error
        events.append(event)
    return events


def record_submission_event(ledger: Path, event: SubmissionEvent) -> None:
    """Record evidence after a user-confirmed external action; perform no upload."""
    _validate_event(event)
    events = read_submission_events(ledger)
    same_day = [item for item in events if item.date_kst == event.date_kst]
    if len(same_day) >= 5:
        raise SubmissionError("daily Dacon submission limit would be exceeded")
    append_canonical_jsonl(ledger, asdict(event))
