"""Closed project failure taxonomy."""


class BaramError(Exception):
    """Base class for expected, operator-readable pipeline failures."""


class ContractError(BaramError):
    """A declared input, schema, lineage, or state contract was violated."""


class LeakageError(ContractError):
    """Information availability or chronological isolation was violated."""


class DataQualityError(ContractError):
    """Supplied data failed a quality gate."""


class MetricError(ContractError):
    """Official metric inputs or evaluation conditions were invalid."""


class ModelError(ContractError):
    """A model or feature-order contract was violated."""


class SubmissionError(ContractError):
    """A local candidate failed schema, byte, or budget validation."""
