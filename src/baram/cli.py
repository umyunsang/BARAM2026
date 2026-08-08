"""Thin command-line router for the closed local BARAM workflow."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from baram.exceptions import BaramError
from baram.workflows import (
    run_audit,
    run_backtest,
    run_build_submission,
    run_fit_final,
    run_lockbox,
    run_prepare,
    run_reproduce,
    run_select,
    run_split_build,
    run_v2_preflight,
)

COMMANDS = (
    "audit",
    "v2-preflight",
    "prepare",
    "split-build",
    "backtest",
    "select",
    "lockbox",
    "fit-final",
    "build-submission",
    "reproduce",
)

HANDLERS = {
    "audit": run_audit,
    "v2-preflight": run_v2_preflight,
    "prepare": run_prepare,
    "split-build": run_split_build,
    "backtest": run_backtest,
    "select": run_select,
    "lockbox": run_lockbox,
    "fit-final": run_fit_final,
    "build-submission": run_build_submission,
    "reproduce": run_reproduce,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m baram.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, required=True)
        if command in {
            "v2-preflight",
            "prepare",
            "split-build",
            "backtest",
            "select",
            "lockbox",
            "fit-final",
            "build-submission",
        }:
            subparser.add_argument("--run-id", required=True)
    subparsers.choices["lockbox"].add_argument("--candidate-freeze", type=Path, required=True)
    subparsers.choices["backtest"].add_argument(
        "--stage",
        choices=(
            "controls",
            "lightgbm",
            "ablation",
            "challengers",
            "spatial-v2",
            "point-v2",
            "distribution-v2",
            "decision-v2",
            "ensemble-v2",
        ),
        required=True,
    )
    subparsers.choices["fit-final"].add_argument("--champion-receipt", type=Path, required=True)
    subparsers.choices["build-submission"].add_argument("--model-receipt", type=Path, required=True)
    subparsers.choices["reproduce"].add_argument("--candidate-receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = HANDLERS[args.command](args)
    except BaramError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
