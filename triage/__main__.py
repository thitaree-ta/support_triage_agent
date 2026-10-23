"""Command line entry point: `python -m triage` (step 1 of the flow).

What it does, in order:
  1. Read settings (environment / .env) and check the API key.
  2. Load the tickets (or the failed ones from an earlier run with --retry-failed).
  3. Build the pipeline and run every ticket through it.
  4. Print the results (text, or JSON with --json; to a file with --output).
  5. With --execute, actually perform the planned actions (writes the outbox files).
  6. Print a summary and return an exit code that scripts can act on.

Exit codes: 0 = fine | 2 = usage/config problem | 3 = too many tickets lost to system failures,
or the outbox could not be written.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path

from triage.actions import OutboxExecutor
from triage.config import Settings
from triage.datastore import RejectedInput, load_tickets_lenient, parse_tickets
from triage.deadletter import load_dead_letter
from triage.factory import build_batch_runner
from triage.report import render, result_to_dict, summary_table

EXIT_OK, EXIT_USAGE, EXIT_ALERT = 0, 2, 3


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # Thai/Japanese text must not crash a Windows console
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    # Step 1: settings and API key.
    try:
        settings = Settings.from_env()  # also loads .env
    except ValueError:
        print("Invalid configuration: check numeric environment settings.", file=sys.stderr)
        return EXIT_USAGE
    if args.model:
        settings = replace(settings, model=args.model)
    if args.mode == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Copy .env.example to .env or export it.", file=sys.stderr)
        return EXIT_USAGE

    # Step 2: load tickets. A broken entry is rejected on its own; a broken FILE is a usage error.
    try:
        tickets, rejected = _load(args, settings)
    except (OSError, ValueError) as exc:
        print(f"Cannot load tickets: {exc}", file=sys.stderr)
        return EXIT_USAGE
    if args.ticket_id:
        wanted = set(args.ticket_id)
        tickets = [t for t in tickets if t.id in wanted]
        rejected = [r for r in rejected if isinstance(r.raw, dict) and r.raw.get("id") in wanted]
        if not tickets and not rejected:
            print(f"No tickets matched {args.ticket_id}", file=sys.stderr)
            return EXIT_USAGE

    # Step 3: run everything.
    if args.mode == "mock":
        from triage.mock_llm import MockLLM
        from triage.factory import build_agent
        runner = build_batch_runner(settings, build_agent(settings, llm=MockLLM()))
    else:
        runner = build_batch_runner(settings)
    report = runner.run(tickets, rejected)

    # Step 4: show the results.
    if args.json:
        payload = [result_to_dict(r, include_traces=args.traces) for r in report.results]
        text = json.dumps(payload, indent=2, ensure_ascii=False)
    else:
        text = "\n".join(render(r) for r in report.results) + f"\n\n{summary_table(report.results)}"
    if args.output:
        try:
            args.output.write_text(text + "\n", encoding="utf-8")
        except OSError:
            # Preserve computed results for recovery; never execute actions after output failure.
            print(text)
            print("Cannot write output file; results printed to stdout. No actions executed.", file=sys.stderr)
            return EXIT_ALERT
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        print(text)

    # Step 5: perform the planned actions, only when asked.
    outbox_ok = True
    if args.execute:
        outbox_ok = _execute(report.planned_actions(), settings)

    # Step 6: summary and exit code.
    return _finish(report, settings, outbox_ok)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="triage", description="Support ticket triage agent")
    parser.add_argument("--tickets", type=Path, help="Tickets JSON file (default: data/tickets.json)")
    parser.add_argument("--retry-failed", type=Path, metavar="FILE", help="Re-run tickets from a dead-letter file")
    parser.add_argument("--ticket-id", action="append", help="Only process this ticket id (repeatable)")
    parser.add_argument("--mode", choices=["openai", "mock"], default="openai")
    parser.add_argument("--model", help="Override OPENAI_MODEL")
    parser.add_argument("--json", action="store_true", help="Print results as JSON")
    parser.add_argument("--traces", action="store_true", help="Include raw tool outputs in --json")
    parser.add_argument("--output", type=Path, metavar="FILE", help="Write the results to FILE (UTF-8) instead of stdout")
    parser.add_argument("--execute", action="store_true",
                        help="Perform the planned actions (write replies/handoffs/feature requests to the outbox folder)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return parser.parse_args(argv)


def _load(args: argparse.Namespace, settings: Settings) -> tuple[list, list[RejectedInput]]:
    if args.retry_failed:
        return parse_tickets(load_dead_letter(args.retry_failed))
    return load_tickets_lenient(args.tickets or settings.data_dir / "tickets.json")


def _execute(actions, settings: Settings) -> bool:
    """Write the planned actions. Returns False if the outbox cannot be written."""
    try:
        summary = OutboxExecutor(settings.outbox_dir).execute(actions)
    except OSError as exc:
        print(f"ERROR: could not write the outbox ({settings.outbox_dir}): {exc}. "
              "The results remain available; partial local writes may exist. No external delivery is implemented.", file=sys.stderr)
        return False
    kinds = ", ".join(f"{k}={v}" for k, v in sorted(summary.by_kind.items())) or "none"
    print(f"Actions executed: {summary.written} written ({kinds}), "
          f"{summary.duplicates_skipped} duplicate(s) skipped -> {settings.outbox_dir}", file=sys.stderr)
    return True


def _finish(report, settings: Settings, outbox_ok: bool) -> int:
    counts = ", ".join(f"{k}={v}" for k, v in sorted(report.counts().items()))
    print(f"\nSummary: {counts}", file=sys.stderr)
    if report.persistence_failures:
        print("ERROR: failed to persist some retry records; retain the output for recovery.", file=sys.stderr)
    if report.dead_lettered:
        print(f"{report.dead_lettered} ticket(s) saved to {settings.dead_letter_path} "
              f"(re-run with --retry-failed {settings.dead_letter_path})", file=sys.stderr)
    if report.breaker_opened:
        print("WARNING: model circuit breaker opened; remaining tickets were sent to humans unanalysed.", file=sys.stderr)
    if report.failure_ratio > settings.degraded_alert_ratio:
        print(f"ALERT: {report.failure_ratio:.0%} of tickets failed (threshold {settings.degraded_alert_ratio:.0%}). "
              "Check OPENAI_API_KEY, OPENAI_MODEL and network.", file=sys.stderr)
        return EXIT_ALERT
    return EXIT_OK if outbox_ok and not report.persistence_failures else EXIT_ALERT


if __name__ == "__main__":
    sys.exit(main())
