"""Application-owned reply catalog: retrieved documents cannot change approved text."""
from triage.schemas import Ticket, ToolTrace

EXPORT_TEXT = "Free plan: export to CSV only, up to 100 rows. Pro plan: unlimited CSV, PDF and PowerPoint export. Export features unlock as soon as the Pro plan is active."
INTENTS = {"what export formats can i use on the free plan", "what export formats are supported"}

def approved_reply(ticket: Ticket, traces: list[ToolTrace]) -> str | None:
    # Exact single-message intent avoids silently ignoring extra questions or injected instructions.
    if len(ticket.messages) != 1:
        return None
    intent = ticket.messages[0].text.casefold().strip().rstrip("?.!")
    if intent not in INTENTS:
        return None
    for trace in traces:
        if trace.ok and trace.name == "search_knowledge_base" and isinstance(trace.output, dict):
            for article in trace.output.get("results", []):
                if article.get("id") == "KB-401" and article.get("content") == EXPORT_TEXT:
                    return EXPORT_TEXT
    return None
