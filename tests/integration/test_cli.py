import subprocess
import sys


def test_cli_help_lists_closed_commands() -> None:
    """Catches an operator entry point that omits or silently renames a pipeline stage."""
    result = subprocess.run(
        [sys.executable, "-m", "baram.cli", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for command in (
        "audit",
        "prepare",
        "split-build",
        "backtest",
        "select",
        "lockbox",
        "fit-final",
        "build-submission",
        "reproduce",
    ):
        assert command in result.stdout
