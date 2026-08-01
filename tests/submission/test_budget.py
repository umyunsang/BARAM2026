from pathlib import Path

import pytest

from baram.exceptions import SubmissionError
from baram.submission.budget import (
    SubmissionEvent,
    read_submission_events,
    record_submission_event,
)


def _event(index: int, *, confirmed: bool = True) -> SubmissionEvent:
    return SubmissionEvent("2026-08-01", f"{index:064x}", f"receipt-{index}", confirmed)


def test_budget_starts_empty_and_records_only_confirmed_external_receipts(tmp_path: Path) -> None:
    """Catches local candidates being mistaken for real Dacon submissions."""
    ledger = tmp_path / "submission-events.jsonl"
    assert read_submission_events(ledger) == []
    with pytest.raises(SubmissionError, match="confirmation"):
        record_submission_event(ledger, _event(1, confirmed=False))
    with pytest.raises(SubmissionError, match="receipt"):
        record_submission_event(
            ledger,
            SubmissionEvent("2026-08-01", "1" * 64, "", True),
        )
    record_submission_event(ledger, _event(1))
    assert read_submission_events(ledger) == [_event(1)]


def test_budget_rejects_sixth_confirmed_event_for_kst_date(tmp_path: Path) -> None:
    """Catches exceeding the official five-per-day budget without any upload action."""
    ledger = tmp_path / "submission-events.jsonl"
    for index in range(5):
        record_submission_event(ledger, _event(index))
    with pytest.raises(SubmissionError, match="limit"):
        record_submission_event(ledger, _event(5))
    assert len(read_submission_events(ledger)) == 5
