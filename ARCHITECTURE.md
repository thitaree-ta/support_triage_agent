# Architecture and Risk Assessment

## Architecture Decisions and Why
- **Small, modular CLI:** separates orchestration, model adapters, tools, policies and presentation without external infrastructure. Dependencies are assembled in `factory.py` and injected for testing.
- **Model proposes; code enforces:** the model selects read-only tools and returns structured analysis. Policies enforce urgency floors and team ownership; final evidence review checks resolution and approved replies.
- **Local BM25 retrieval:** inexpensive and inspectable for a small KB. The model forms English queries for Thai tickets; the retriever can be replaced through its interface.
- **Validated, bounded execution:** Pydantic checks contracts, not factual truth. Defaults cap execution at six model rounds and twelve tool calls. Request timeouts and cooperative ticket deadlines limit waiting.
- **Separate planning and execution:** results contain proposed actions. `--execute` writes a local JSONL outbox; it does not send messages, issue refunds or change accounts.

## What Could Go Wrong and How It Is Handled
- **Incorrect classification:** deterministic floors and explicit risk signals can override under-triage. They do not correct over-escalation, and some rules depend on model-extracted facts.
- **Unsupported claims:** citation checks, wording guards and approved reply templates restrict automatic answers. Retrieved IDs alone do not establish relevance; free-text reasoning still needs review.
- **False resolution:** troubleshooting documentation does not prove success. Customer confirmations are labeled as reports, not independent verification.
- **Model or tool failures:** bounded retries, traces, human fallbacks, a batch circuit breaker and retry records preserve visibility. Cooperative deadlines cannot cancel an in-flight operation.
- **Business deadline before SLA:** handoffs include elapsed-time-adjusted deadlines and an attention flag. This is not a repair promise or an operational scheduling system.
- **Injection and privacy:** untrusted-data instructions, escaped input and customer-scoped read-only tools reduce risk. Card masking is not complete PII protection or proof of injection immunity.
- **Production gaps:** authenticate customer context, add fresh monitoring timestamps and incident-relevance checks, protect stored data, and replace sequential JSONL deduplication with transactional delivery.

## How the Agent Would Be Evaluated in Production
- Build deidentified English/Thai datasets with two independent support reviewers and adjudication. Hold out customers and later time periods from tuning.
- Measure urgency precision/recall, especially critical recall; action accuracy and high-risk escalation recall; retrieval recall@k and response groundedness.
- Track p50/p95 latency, tokens/cost, tool/schema failures and human overrides by language and issue type.
- Agree release gates with support/security based on missed-incident costs and review capacity. Start in shadow mode and rerun holdout/adversarial tests after model, prompt or policy changes. No production performance is claimed.
