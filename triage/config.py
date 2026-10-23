"""Validated runtime settings loaded from environment variables and an optional .env file."""
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is a convenience, not a requirement
        return
    load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    model: str = "gpt-5.6-terra"  # which OpenAI model to use (OPENAI_MODEL)
    reasoning_effort: str = "low"  # how hard the model "thinks"; "" omits it for non-reasoning models
    max_agent_steps: int = 6  # at most this many AI <-> tool rounds per ticket
    low_confidence_threshold: float = 0.6  # below this the model's own confidence -> ask a person
    kb_top_k: int = 3  # how many help articles a search returns
    request_timeout_seconds: float = 60.0  # per OpenAI request
    ticket_timeout_seconds: float = 90.0  # whole-ticket budget, checked between steps
    auto_respond_enabled: bool = True  # kill switch: False turns every auto-respond into a route
    breaker_threshold: int = 3  # consecutive model failures before the circuit opens
    degraded_alert_ratio: float = 0.2  # batch exits non-zero when more results than this are degraded
    max_message_chars: int = 4000
    dead_letter_path: Path = ROOT_DIR / "failed_tickets.jsonl"
    outbox_dir: Path = ROOT_DIR / "outbox"  # where --execute writes replies, handoffs, feature requests
    data_dir: Path = ROOT_DIR / "data"  # where the mock data files live
    prompt_path: Path = ROOT_DIR / "prompts" / "system.md"

    def __post_init__(self) -> None:
        for name in ("max_agent_steps", "kb_top_k", "request_timeout_seconds",
                     "ticket_timeout_seconds", "breaker_threshold", "max_message_chars"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be positive and finite")
        for name in ("low_confidence_threshold", "degraded_alert_ratio"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(name + " must be between zero and one")

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        defaults = cls()
        return cls(
            model=os.getenv("OPENAI_MODEL", defaults.model),
            reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", defaults.reasoning_effort),
            max_agent_steps=int(os.getenv("TRIAGE_MAX_STEPS", defaults.max_agent_steps)),
            low_confidence_threshold=float(
                os.getenv("TRIAGE_CONFIDENCE_THRESHOLD", defaults.low_confidence_threshold)
            ),
            kb_top_k=int(os.getenv("TRIAGE_KB_TOP_K", defaults.kb_top_k)),
            request_timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", defaults.request_timeout_seconds)),
            ticket_timeout_seconds=float(os.getenv("TRIAGE_TIMEOUT_SECONDS", defaults.ticket_timeout_seconds)),
            auto_respond_enabled=_env_bool("TRIAGE_AUTO_RESPOND", defaults.auto_respond_enabled),
            breaker_threshold=int(os.getenv("TRIAGE_BREAKER_THRESHOLD", defaults.breaker_threshold)),
            degraded_alert_ratio=float(os.getenv("TRIAGE_ALERT_RATIO", defaults.degraded_alert_ratio)),
            max_message_chars=int(os.getenv("TRIAGE_MAX_MESSAGE_CHARS", defaults.max_message_chars)),
            dead_letter_path=Path(os.getenv("TRIAGE_DEAD_LETTER", str(defaults.dead_letter_path))),
            outbox_dir=Path(os.getenv("TRIAGE_OUTBOX", str(defaults.outbox_dir))),
        )
