"""Prepare readable, bounded input and mask detected Luhn-valid card numbers. This is not complete PII protection."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from triage.schemas import Message, Ticket

_CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
CARD_PLACEHOLDER = "[CARD NUMBER REDACTED]"


def luhn_valid(digits: str) -> bool:
    """Standard checksum used by real card numbers. Lets us ignore long numbers that are not cards
    (order numbers, phone numbers) so we only hide real-looking cards."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n = n * 2 - 9 if n * 2 > 9 else n * 2
        total += n
    return total % 10 == 0


def mask_card_numbers(text: str) -> tuple[str, bool]:
    """Replace card numbers with a placeholder. Returns (new text, whether anything was hidden)."""
    found = False

    def replace(match: re.Match[str]) -> str:
        nonlocal found
        digits = re.sub(r"\D", "", match.group())
        if 13 <= len(digits) <= 19 and luhn_valid(digits):
            found = True
            return CARD_PLACEHOLDER
        return match.group()

    return _CARD_CANDIDATE.sub(replace, text), found


@dataclass(frozen=True)
class PreflightResult:
    ticket: Ticket
    bypass_reason: str | None = None  # set when the model should not be called
    notes: list[str] = field(default_factory=list)


class Preflight:
    def __init__(self, max_message_chars: int = 4000, max_messages: int = 30) -> None:
        self._max_chars = max_message_chars
        self._max_messages = max_messages

    def run(self, ticket: Ticket) -> PreflightResult:
        notes: list[str] = []

        # Step 1: keep only messages that contain something.
        readable = [m for m in ticket.chronological() if m.text.strip()]
        if not readable:
            return PreflightResult(
                ticket, bypass_reason="Ticket has no readable customer message; ask the customer for details."
            )
        if len(readable) < len(ticket.messages):
            notes.append(f"{len(ticket.messages) - len(readable)} blank message(s) dropped")
        # Step 2: a very long thread is trimmed to the newest messages.
        if len(readable) > self._max_messages:
            notes.append(f"only the latest {self._max_messages} of {len(readable)} messages kept")
            readable = readable[-self._max_messages :]

        # Step 3: per message, hide card numbers first, then cut over-long text.
        cleaned: list[Message] = []
        masked = truncated = False
        for message in readable:
            text, was_masked = mask_card_numbers(message.text.strip())
            masked |= was_masked
            if len(text) > self._max_chars:
                text, truncated = text[: self._max_chars] + " [TRUNCATED]", True
            cleaned.append(Message(minutes_ago=message.minutes_ago, text=text))
        if masked:
            notes.append("payment card number masked before analysis")
        if truncated:
            notes.append(f"long message truncated to {self._max_chars} characters")

        # Step 4: build a NEW ticket (the original is left untouched).
        sanitized = Ticket(
            id=ticket.id,
            customer=ticket.customer.model_copy(update={
                "notes": mask_card_numbers(ticket.customer.notes)[0],
                "customer_id": mask_card_numbers(ticket.customer.customer_id)[0],
                "region": mask_card_numbers(ticket.customer.region)[0] if ticket.customer.region else None,
            }),
            messages=cleaned,
            last_agent_reply_minutes_ago=ticket.last_agent_reply_minutes_ago,
        )
        return PreflightResult(sanitized, notes=notes)
