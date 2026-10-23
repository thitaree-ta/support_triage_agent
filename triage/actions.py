"""Plan support actions and optionally append them to a local JSONL outbox.

Sequential identical replays deduplicate; concurrent delivery needs transactional storage."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from triage.schemas import (
    Action, ActionKind, PlannedAction, ResultStatus, TriageResult,
)

# Which file each kind of action is written to.
_OUTBOX_FILES = {
    ActionKind.SEND_REPLY: "replies.jsonl",
    ActionKind.CREATE_HANDOFF: "handoffs.jsonl",
    ActionKind.LOG_FEATURE_REQUEST: "feature_requests.jsonl",
}


def _key(ticket_id: str, kind: ActionKind, core: Any) -> str:
    """Fingerprint of an action. Same ticket + same content -> same key."""
    raw = json.dumps([ticket_id, kind.value, core], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# =============================================================================== PLAN
def plan_actions(result: TriageResult, customer_id: str | None = None) -> list[PlannedAction]:
    """Decide what should happen for this result. Does not do anything itself."""
    actions: list[PlannedAction] = []
    analysis = result.analysis

    # (a) Anything that needs people gets a handoff: routed, escalated, failed, blank, rejected.
    if result.action in (Action.ROUTE_TO_SPECIALIST, Action.ESCALATE_TO_HUMAN):
        team = result.route_to.value if result.route_to else "triage_queue"
        summary = (
            (analysis.handoff_summary if analysis else None)
            or result.digest
            or result.status_reason
            or (analysis.reasoning if analysis else "")
        )
        payload = {
            "priority": "escalation" if result.action is Action.ESCALATE_TO_HUMAN else "routing",
            "team": team,
            "urgency": result.urgency.value,
            "status": result.status.value,
            "customer_id": customer_id,
            "summary": summary,
            "sla": result.sla.model_dump(mode="json") if result.sla else None,
            "customer_deadline_minutes": analysis.customer_deadline_minutes if analysis else None,
            # The draft is only a suggestion for the person; it is NOT sent automatically.
            "draft_reply_for_review": analysis.draft_reply if analysis else None,
            "reply_issues": [i.model_dump(mode="json") for i in result.reply_issues],
            "policy_adjustments": [a.rule for a in result.adjustments],
        }
        remaining = result.customer_deadline_remaining_minutes
        payload["customer_deadline_remaining_minutes"] = remaining
        payload["deadline_before_sla"] = (
            remaining is not None and result.sla is not None
            and remaining < result.sla.remaining_minutes
        )
        # Highlight business urgency without changing the contractual SLA or promising a fix.
        if remaining is not None and (remaining <= 0 or payload["deadline_before_sla"]):
            payload["attention_reason"] = "Customer-reported deadline requires review before the SLA target."
            if payload["priority"] != "escalation":
                payload["priority"] = "deadline_sensitive"
        actions.append(PlannedAction(
            kind=ActionKind.CREATE_HANDOFF, ticket_id=result.ticket_id,
            idempotency_key=_key(result.ticket_id, ActionKind.CREATE_HANDOFF, payload),
            payload=payload,
        ))

    # (b) An automatic reply: only when nothing at all stops it (auto_send already checks
    #     status, action, and that the reply guard found no problems).
    if result.auto_send and analysis is not None:
        actions.append(PlannedAction(
            kind=ActionKind.SEND_REPLY, ticket_id=result.ticket_id,
            idempotency_key=_key(result.ticket_id, ActionKind.SEND_REPLY, analysis.draft_reply),
            payload={
                "customer_id": customer_id,
                "language": analysis.customer_language,
                "text": analysis.draft_reply,
                "kb_article_ids": analysis.kb_article_ids,
            },
        ))

    # (c) Product suggestions are logged whenever the model really analysed the ticket.
    if analysis is not None and result.status is ResultStatus.OK:
        for request in analysis.feature_requests:
            actions.append(PlannedAction(
                kind=ActionKind.LOG_FEATURE_REQUEST, ticket_id=result.ticket_id,
                idempotency_key=_key(result.ticket_id, ActionKind.LOG_FEATURE_REQUEST, request),
                payload={"customer_id": customer_id, "request": request},
            ))
    return actions


# ============================================================================ EXECUTE
@dataclass
class ExecutionSummary:
    written: int = 0
    duplicates_skipped: int = 0
    by_kind: Counter = field(default_factory=Counter)  # how many of each kind were written


class OutboxExecutor:
    """Writes planned actions to JSONL files, skipping any it has already written."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._seen: set[str] | None = None  # loaded lazily, on the first execute()

    def execute(self, actions: Iterable[PlannedAction]) -> ExecutionSummary:
        """Raises OSError if the outbox cannot be written (the caller decides what to do)."""
        self._dir.mkdir(parents=True, exist_ok=True)
        if self._seen is None:
            self._seen = self._load_existing_keys()
        summary = ExecutionSummary()
        for action in actions:
            if action.idempotency_key in self._seen:
                summary.duplicates_skipped += 1
                continue
            record = {
                "key": action.idempotency_key,
                "kind": action.kind.value,
                "ticket_id": action.ticket_id,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "payload": action.payload,
            }
            with open(self._dir / _OUTBOX_FILES[action.kind], "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._seen.add(action.idempotency_key)
            summary.written += 1
            summary.by_kind[action.kind.value] += 1
        return summary

    def _load_existing_keys(self) -> set[str]:
        """Read the keys already on disk so a second run does not repeat anything."""
        keys: set[str] = set()
        for filename in _OUTBOX_FILES.values():
            path = self._dir / filename
            if not path.exists():
                continue
            with open(path, encoding="utf-8-sig") as fh:
                for line in fh:
                    try:
                        keys.add(json.loads(line)["key"])
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue  # ignore a damaged line rather than refuse to run
        return keys
