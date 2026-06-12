"""CLI subcommand: `hermes oracle`.

Thin shell around agent/oracle.py — generates and prints the fleet
evaluation report ("State of the Matrix") from the local session DB.

This module intentionally has no side effects at import time — main.py
wires the argparse subparsers on demand.
"""

from __future__ import annotations

import argparse
import sys


def _cmd_report(args) -> int:
    from hermes_state import SessionDB
    from agent.oracle import OracleEngine

    db = SessionDB()
    try:
        engine = OracleEngine(db)
        report = engine.generate(
            days=args.days,
            source=args.source,
            bump_iteration=not args.json,
        )
        if args.json:
            print(engine.format_json(report))
        else:
            color = sys.stdout.isatty() and not args.no_color
            print(engine.format_terminal(report, color=color))
        return 0
    finally:
        db.close()


def register_cli(parent: argparse.ArgumentParser) -> None:
    """Attach `oracle` arguments to *parent*.

    main.py calls this with the ArgumentParser returned by
    ``subparsers.add_parser("oracle", ...)``.
    """
    parent.add_argument(
        "--days", type=int, default=7,
        help="Lookback window in days (default: 7)",
    )
    parent.add_argument(
        "--source",
        help="Filter to one platform (cli, telegram, discord, ...)",
    )
    parent.add_argument(
        "--json", action="store_true",
        help="Emit the raw report as JSON (read-only — does not advance "
             "the iteration counter)",
    )
    parent.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI colors in the terminal report",
    )
    parent.set_defaults(func=_cmd_report)
