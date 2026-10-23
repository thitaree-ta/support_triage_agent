"""Eval runner: load cases, process each (optionally several times), score, aggregate."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from evals.scoring import KNOWN_KEYS, check
from triage.schemas import Ticket, TriageResult

SAFETY_TAG = "safety"


@dataclass(frozen=True)
class EvalCase:
    id: str
    description: str
    tags: tuple[str, ...]
    ticket: Ticket
    expect: dict[str, Any]


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for raw in json.loads(path.read_text(encoding="utf-8")):
        if raw["id"] in seen:
            raise ValueError(f"Duplicate eval case id {raw['id']}")
        seen.add(raw["id"])
        unknown = set(raw["expect"]) - KNOWN_KEYS
        if unknown:
            raise ValueError(f"{raw['id']}: unknown expectation keys {sorted(unknown)}")
        ticket_data = dict(raw["ticket"])
        ticket_data["messages"] = [  # "repeat": N expands long text without bloating the file
            {"minutes_ago": m["minutes_ago"], "text": m["text"] * m.get("repeat", 1)} for m in ticket_data["messages"]
        ]
        cases.append(EvalCase(raw["id"], raw["description"], tuple(raw.get("tags", [])),
                              Ticket.model_validate(ticket_data), raw["expect"]))
    return cases


@dataclass
class RunOutcome:
    failures: list[str]
    action: str
    urgency: str
    status: str


@dataclass
class CaseOutcome:
    case: EvalCase
    runs: list[RunOutcome] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(not r.failures for r in self.runs)

    @property
    def consistency(self) -> float:
        """Share of runs that chose the most common action (1.0 = fully stable)."""
        counts = Counter(r.action for r in self.runs)
        return max(counts.values()) / len(self.runs) if self.runs else 1.0


@dataclass
class EvalReport:
    outcomes: list[CaseOutcome]
    runs: int
    model: str = ""

    @property
    def passed(self) -> int:
        return sum(o.passed for o in self.outcomes)

    @property
    def safety(self) -> tuple[int, int]:
        cases = [o for o in self.outcomes if SAFETY_TAG in o.case.tags]
        return sum(o.passed for o in cases), len(cases)

    @property
    def system_failures(self) -> int:
        return sum(1 for o in self.outcomes for r in o.runs if r.status in ("degraded", "rejected"))

    @property
    def mean_consistency(self) -> float:
        return sum(o.consistency for o in self.outcomes) / len(self.outcomes) if self.outcomes else 1.0

    def render_markdown(self) -> str:
        safe_ok, safe_total = self.safety
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            "# Eval results", "",
            f"- Model: `{self.model or 'n/a'}`  |  Runs per case: {self.runs}  |  Generated: {stamp}",
            f"- Cases passed: **{self.passed}/{len(self.outcomes)}**",
            f"- Safety cases passed: **{safe_ok}/{safe_total}**",
            f"- Action consistency across runs: **{self.mean_consistency:.0%}**",
            f"- System failures (degraded/rejected runs): **{self.system_failures}**", "",
            "| Case | Description | Result | Final decision | Consistency |", "|---|---|---|---|---|",
        ]
        for o in self.outcomes:
            last = o.runs[-1] if o.runs else None
            decision = f"{last.status}/{last.urgency}/{last.action}" if last else "-"
            verdict = "PASS" if o.passed else "FAIL"
            lines.append(f"| {o.case.id} | {o.case.description} | {verdict} | {decision} | {o.consistency:.0%} |")
        failures = [(o, r) for o in self.outcomes for r in o.runs if r.failures]
        if failures:
            lines += ["", "## Failures", ""]
            for o, r in failures:
                lines.append(f"- **{o.case.id}**: " + "; ".join(r.failures))
        return "\n".join(lines) + "\n"

    def render_text(self) -> str:
        safe_ok, safe_total = self.safety
        rows = [f"{o.case.id:<7}{'PASS' if o.passed else 'FAIL':<6}{o.consistency:<6.0%}{o.case.description}" for o in self.outcomes]
        for o in self.outcomes:
            for r in o.runs:
                for failure in r.failures:
                    rows.append(f"        - {o.case.id}: {failure}")
        rows.append(f"\nPassed {self.passed}/{len(self.outcomes)} | safety {safe_ok}/{safe_total} | "
                    f"consistency {self.mean_consistency:.0%} | system failures {self.system_failures}")
        return "\n".join(rows)


class EvalRunner:
    def __init__(self, process: Callable[[Ticket], TriageResult], runs: int = 1, model: str = "") -> None:
        self._process = process
        self._runs = max(1, runs)
        self._model = model

    def run(self, cases: Sequence[EvalCase]) -> EvalReport:
        outcomes: list[CaseOutcome] = []
        for case in cases:
            outcome = CaseOutcome(case)
            for _ in range(self._runs):
                result = self._process(case.ticket)
                outcome.runs.append(RunOutcome(
                    failures=check(case.expect, result), action=result.action.value,
                    urgency=result.urgency.value, status=result.status.value,
                ))
            outcomes.append(outcome)
        return EvalReport(outcomes, self._runs, self._model)

