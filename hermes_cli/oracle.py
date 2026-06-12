"""CLI subcommand: `hermes oracle`.

Thin shell around agent/oracle.py — generates and prints the fleet
evaluation report ("State of the Matrix") from the local session DB,
and manages anomaly acknowledgements.

This module intentionally has no side effects at import time — main.py
wires the argparse subparsers on demand.
"""

from __future__ import annotations

import argparse
import sys


def _open_engine():
    from hermes_state import SessionDB
    from agent.oracle import OracleEngine

    db = SessionDB()
    return db, OracleEngine(db)


def _cmd_report(args) -> int:
    db, engine = _open_engine()
    try:
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


def _cmd_ack(args) -> int:
    db, engine = _open_engine()
    try:
        try:
            ack = engine.acknowledge(args.key, note=args.note)
        except ValueError as exc:
            print(f"✗ {exc}")
            return 1
        print(f"✔ Acknowledged '{args.key}' — demoted to info until its "
              f"numbers materially change (snapshot: {ack['metrics']}).")
        print(f"  Undo with: hermes oracle unack {args.key}")
        return 0
    finally:
        db.close()


def _cmd_unack(args) -> int:
    db, engine = _open_engine()
    try:
        if engine.unacknowledge(args.key):
            print(f"✔ Removed acknowledgement for '{args.key}'.")
            return 0
        print(f"✗ No acknowledgement found for '{args.key}'.")
        return 1
    finally:
        db.close()


def _cmd_acks(args) -> int:
    db, engine = _open_engine()
    try:
        acks = engine.get_acknowledgements()
        if not acks:
            print("No acknowledged anomalies.")
            return 0
        for key, ack in sorted(acks.items()):
            acked = (ack.get("acked_at") or "?")[:19]
            note = f" — {ack['note']}" if ack.get("note") else ""
            print(f"  {key}  (acked {acked}){note}")
        return 0
    finally:
        db.close()


def register_cli(parent: argparse.ArgumentParser) -> None:
    """Attach `oracle` arguments and subcommands to *parent*.

    main.py calls this with the ArgumentParser returned by
    ``subparsers.add_parser("oracle", ...)``. Bare ``hermes oracle``
    (with optional flags) prints the report; ``ack``/``unack``/``acks``
    manage anomaly acknowledgements.
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

    subs = parent.add_subparsers(dest="oracle_command")

    p_ack = subs.add_parser(
        "ack",
        help="Acknowledge an active anomaly — demote it to info until "
             "its numbers materially worsen",
    )
    p_ack.add_argument(
        "key",
        help="Anomaly key from the report, e.g. 'cost_spike' or "
             "'tool_failure_rate:memory'",
    )
    p_ack.add_argument("--note", help="Why this is acknowledged")
    p_ack.set_defaults(func=_cmd_ack)

    p_unack = subs.add_parser("unack", help="Remove an acknowledgement")
    p_unack.add_argument("key", help="Anomaly key")
    p_unack.set_defaults(func=_cmd_unack)

    p_acks = subs.add_parser("acks", help="List acknowledged anomalies")
    p_acks.set_defaults(func=_cmd_acks)
