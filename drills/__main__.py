"""CLI: python -m drills [--list] [--scenario api_down]

Offline (no API key). Exit code 0 if every check holds, 1 otherwise.
"""
from __future__ import annotations

import argparse
import logging
import sys
import traceback

from drills.scenarios import SCENARIOS, Check


def run_scenario(scenario) -> tuple[str, list[Check]]:
    try:
        return scenario.run()
    except Exception:  # noqa: BLE001 - a crashing drill is itself a failed check
        return "scenario crashed", [Check("scenario ran without crashing: " + traceback.format_exc().strip().splitlines()[-1], False)]


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="drills", description="Offline failure drills for the triage pipeline")
    parser.add_argument("--list", action="store_true", help="List scenarios and exit")
    parser.add_argument("--scenario", action="append", help="Run only this scenario (repeatable)")
    args = parser.parse_args(argv)
    logging.disable(logging.CRITICAL)  # the faults are injected on purpose; the report shows the outcome

    if args.list:
        for s in SCENARIOS:
            print(f"{s.name:<24}{s.fault}")
        return 0
    chosen = [s for s in SCENARIOS if not args.scenario or s.name in args.scenario]
    if not chosen:
        print(f"No scenario matched {args.scenario}. Use --list.", file=sys.stderr)
        return 2

    failed = 0
    for s in chosen:
        observed, checks = run_scenario(s)
        ok = all(c.ok for c in checks)
        failed += not ok
        print(f"\n[{'PASS' if ok else 'FAIL'}] {s.name}")
        print(f"  injected : {s.fault}")
        print(f"  expected : {s.expectation}")
        print(f"  observed : {observed}")
        for c in checks:
            print(f"    {'ok  ' if c.ok else 'FAIL'} {c.label}")
    print(f"\n{len(chosen) - failed}/{len(chosen)} drills passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
