"""Compatibility tombstone for the retired raw-row chunk arbiter.

Promotion callers must use :mod:`baram.evaluation.prequential`.  The former
implementation split one 01:00--00:00 issuance at midnight, sampled fixed
non-overlapping chunks, and reset the adaptive comparison family.
"""

from __future__ import annotations

from typing import Any, NoReturn


class RetiredArbiterError(RuntimeError):
    """Raised whenever an obsolete arbitration entry point is invoked."""


def _retired() -> NoReturn:
    raise RetiredArbiterError(
        "legacy arbiter retired; use run_prequential_protocol with the "
        "authoritative EventStore comparison index"
    )


def paired_bootstrap(*args: Any, **kwargs: Any) -> NoReturn:
    """Fail closed instead of producing invalid fixed-chunk evidence."""
    _retired()


def arbitrate(*args: Any, **kwargs: Any) -> NoReturn:
    """Fail closed instead of making a promotion decision."""
    _retired()
