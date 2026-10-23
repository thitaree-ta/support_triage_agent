"""CLI: python -m evals [--runs 3] [--case EV-05] [--write-report evals/RESULTS.md]

Runs the eval cases against the REAL model (needs OPENAI_API_KEY) through the same pipeline
as production, so preflight, policy, SLA and fallbacks are all exercised.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from evals.runner import EvalRunner, load_cases
from triage.breaker import CircuitBreaker
from triage.config import Settings
from triage.factory import build_pipeline

CASES_PATH = Path(__file__).with_name("cases.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals", description="Run triage evals against the live model")
    parser.add_argument("--runs", type=int, default=1, help="Runs per case (use 3+ to measure consistency)")
    parser.add_argument("--case", action="append", help="Only run this case id (repeatable)")
    parser.add_argument("--tag", help="Only run cases with this tag (e.g. safety)")
    parser.add_argument("--model", help="Override OPENAI_MODEL")
    parser.add_argument("--write-report", type=Path, metavar="FILE", help="Write a Markdown report")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    if args.model:
        from dataclasses import replace
        settings = replace(settings, model=args.model)
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2

    cases = load_cases(CASES_PATH)
    if args.case:
        cases = [c for c in cases if c.id in set(args.case)]
    if args.tag:
        cases = [c for c in cases if args.tag in c.tags]
    if not cases:
        print("No eval cases selected.", file=sys.stderr)
        return 2

    # Circuit breaker disabled: an eval should measure every case, not stop after failures.
    pipeline, _ = build_pipeline(settings, breaker=CircuitBreaker(None))
    report = EvalRunner(lambda t: pipeline.process(t).result, runs=args.runs, model=settings.model).run(cases)

    print(report.render_text())
    if args.write_report:
        args.write_report.write_text(report.render_markdown(), encoding="utf-8")
        print(f"\nReport written to {args.write_report}")
    return 0 if report.passed == len(report.outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
