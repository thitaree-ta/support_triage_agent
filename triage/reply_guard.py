"""Reply guard: checks the WORDS of a draft reply before a customer could ever see it.

Why this exists
---------------
The policy rules decide *what to do* with a ticket, but the model also writes the reply text.
Models sometimes write things that sound helpful yet are false or risky, for example:
    "We've logged your request"      -> nothing was logged (claims an action that did not happen)
    "We'll refund you within 3 days" -> a promise only a person may make
    "Our monitoring shows 41% errors" -> leaks internal information
So every draft is scanned with simple, readable pattern rules (English + Thai). A match does
not delete anything: it is recorded, and an automatic reply becomes "a person must review it".

Limits (be honest): patterns catch common phrasings, not every possible one. That is why the
prompt also forbids these phrases, and why the evals measure how often the model still does it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from triage.preflight import mask_card_numbers
from triage.schemas import ReplyIssue

_APOS = "[’']"  # straight and curly apostrophes both appear in model output
_FLAGS = re.IGNORECASE


@dataclass(frozen=True)
class PatternRule:
    name: str
    reason: str
    patterns: tuple[re.Pattern[str], ...]
    # Some phrases are fine when they merely repeat a fact from a KB article the agent really
    # retrieved (e.g. "reset emails arrive within 5 minutes"). Those matches are ignored.
    allow_if_in_kb: bool = False


def _rx(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, _FLAGS) for p in patterns)


_DONE_VERBS = (
    "logged|noted|recorded|escalated|forwarded|passed|sent|submitted|refunded|reversed|credited|"
    "fixed|resolved|created|opened|raised|activated|cancell?ed|processed"
)
_DOING_VERBS = "sending|escalating|forwarding|passing|logging|submitting|refunding|reversing|opening|raising|processing|fixing"

DEFAULT_RULES: tuple[PatternRule, ...] = (
    PatternRule(
        name="unverified_customer_role",
        reason="The available tools do not verify the customer's administrator permissions.",
        patterns=_rx(r"\bas (?:a |the )?workspace admin(?:istrator)?\b",
                     r"\byou are (?:a |the )?(?:workspace )?admin(?:istrator)?\b",
                     r"คุณเป็นผู้ดูแล"),
    ),
    PatternRule(
        name="claims_action",
        reason="Says an action was done or is being done, but the reply itself does nothing. "
        "Only the system's real actions may be described as done.",
        patterns=_rx(
            rf"\b(?:we{_APOS}ve|we have|i{_APOS}ve|i have|has been|have been)\s+(?:\w+\s+){{0,2}}?(?:{_DONE_VERBS})\b",
            rf"\b(?:we{_APOS}re|we are|i{_APOS}m|i am)\s+(?:now\s+|currently\s+|already\s+)?(?:{_DOING_VERBS})\b",
            r"(?:บันทึก|ส่งต่อ|ส่งเรื่อง|ส่งให้|ดำเนินการ|คืนเงิน|แก้ไข|ยกระดับ).{0,20}?(?:แล้ว|เรียบร้อย)",
            r"กำลัง(?:ส่งเรื่อง|ส่งต่อ|ส่งให้|ดำเนินการ|ยกระดับ|แก้ไข|คืนเงิน|บันทึก)",
            r"ได้ทำการ",
        ),
    ),
    PatternRule(
        name="promises_outcome",
        reason="Promises a refund, credit or guarantee. Only a person can commit to money outcomes.",
        patterns=_rx(
            r"\bwill (?:be )?(?:\w+ ){0,3}?refund(?:ed)?\b",
            r"\bfull refund\b",
            r"\brefund (?:has been|is) (?:approved|issued|processed)\b",
            rf"\b(?:we{_APOS}ll|we will)\b[^.]{{0,30}}\b(?:credit|reverse|waive)\b",
            r"\bguarantee[sd]?\b",
            r"(?:จะ|ขอ)?คืนเงิน(?:ให้|แก่)",
            r"จะคืนเงิน",
            r"รับประกัน",
        ),
    ),
    PatternRule(
        name="promises_timeline",
        reason="Gives a time commitment (ETA) that nobody has agreed to.",
        patterns=_rx(
            r"\bwithin (?:the next )?\d+\s*(?:-\s*\d+\s*)?(?:minutes?|mins?|hours?|hrs?|days?|business days?|weeks?)\b",
            r"\bby (?:tomorrow|tonight|end of (?:the )?day|eod|today)\b",
            r"\bETA\b",
            r"ภายใน\s*\d+\s*(?:-\s*\d+\s*)?(?:นาที|ชั่วโมง|ชม\.|วัน|สัปดาห์)",
            r"ภายในวันนี้|ภายในพรุ่งนี้",
        ),
        allow_if_in_kb=True,
    ),
    PatternRule(
        name="internal_disclosure",
        reason="Reveals internal monitoring/metrics or confirms an outage before the incident team has.",
        patterns=_rx(
            r"\binternal (?:monitoring|metrics|alerts?|dashboards?|logs?|tools?)\b",
            r"\b5xx\b",
            r"\berror rate\b",
            r"\bon-?call\b",
            r"\bapi[- ]gateway\b",
            r"\bap-southeast\b",
            r"\b(?:we|the team) (?:have |has )?(?:confirmed|identified|detected) (?:an? |the )?(?:outage|incident)\b",
            r"\b(?:outage|incident) (?:is|has been) (?:confirmed|acknowledged)\b",
            r"มอนิเตอร์|ออนคอล|5xx",
            r"ยืนยัน(?:ว่า)?.{0,12}(?:ล่ม|ขัดข้อง|เหตุ)",
            r"พบ(?:รายงาน)?(?:ความผิดพลาด|เหตุขัดข้อง|ปัญหา)ของระบบ",
        ),
    ),
    PatternRule(
        name="asks_for_secrets",
        reason="Asks the customer to send a card number, password or similar secret.",
        patterns=_rx(
            r"\b(?:send|provide|share|give|tell|email|type|reply with)\b[^.?!]{0,40}\b"
            r"(?:card number|credit card|cvv|cvc|password|passcode|one-time (?:code|password)|otp|api key|secret key)\b",
            r"(?:ส่ง|แจ้ง|ขอ|บอก|ให้).{0,15}(?:เลขบัตร|หมายเลขบัตร|รหัสผ่าน|cvv|otp)",
        ),
    ),
)


class ReplyGuard:
    """Runs every rule over a draft and returns the problems found (empty list = clean)."""

    def __init__(self, rules: Sequence[PatternRule] = DEFAULT_RULES) -> None:
        self._rules = list(rules)

    def check(self, reply: str | None, kb_text: str = "") -> list[ReplyIssue]:
        """`kb_text` is the text of KB articles the agent really retrieved (used by allow_if_in_kb)."""
        if not reply or not reply.strip():
            return []
        issues: list[ReplyIssue] = []
        for rule in self._rules:
            for pattern in rule.patterns:
                for match in pattern.finditer(reply):
                    if rule.allow_if_in_kb and _documented_reset_timing(reply, match, kb_text):
                        continue  # the reply is just repeating a documented fact
                    issues.append(ReplyIssue(rule=rule.name, excerpt=_excerpt(match.group()), reason=rule.reason))
        # A real card number in the reply is always wrong (the model only ever sees a masked one).
        if mask_card_numbers(reply)[1]:
            issues.append(ReplyIssue(rule="card_number_in_reply", excerpt="[card number]",
                                     reason="The reply contains a payment card number."))
        return _dedupe(issues)


def _documented_reset_timing(reply: str, match: re.Match[str], kb_text: str) -> bool:
    """Only exempt a complete known fact, never a matching time fragment.

    A repair promise must remain blocked even when a retrieved article contains
    the same number of minutes. Thai wording is an application-owned translation.
    """
    fact = "reset emails arrive within 5 minutes"
    if fact not in _normalise(kb_text):
        return False
    patterns = (
        r"(?:^|[.!?]\s*)reset emails arrive within 5 minutes(?:[.!?]|$)",
        r"(?:^|[.!?]\s*|โดย)อีเมลรีเซ็ตปกติควรมาถึงภายใน 5 นาที(?: \(KB-501\))?(?:[.!?]|\s|$)",
    )
    return any(
        allowed.start() <= match.start() and match.end() <= allowed.end()
        for pattern in patterns
        for allowed in re.finditer(pattern, reply, re.IGNORECASE)
    )


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def _excerpt(text: str, limit: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _dedupe(issues: list[ReplyIssue]) -> list[ReplyIssue]:
    seen: set[tuple[str, str]] = set()
    unique: list[ReplyIssue] = []
    for issue in issues:
        key = (issue.rule, issue.excerpt.lower())
        if key not in seen:
            seen.add(key)
            unique.append(issue)
    return unique
