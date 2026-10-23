"""Minimal payment-card redaction for all storage paths, including invalid input."""
from typing import Any
from triage.preflight import mask_card_numbers

def redact(value: Any) -> Any:
    if isinstance(value, str):
        return mask_card_numbers(value)[0]
    if isinstance(value, dict):
        return {redact(k): redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value
