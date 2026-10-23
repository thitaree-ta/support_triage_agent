# Support Ticket Triage Agent

## Project Overview
Python CLI for support-ticket triage: extract the issue, retrieve evidence, classify urgency and recommend the next action. OpenAI proposes structured analysis; application rules enforce safety. All bundled customer, billing and service data is synthetic.

## Key Features
- Four urgency levels: `low`, `medium`, `high`, `critical`.
- Three actions: `auto_respond`, `route_to_specialist`, `escalate_to_human`.
- Four read-only tools, evidence traces, bounded execution and human fallbacks.
- English/Thai fixtures; offline mock and API-backed modes.
- Automatic replies require approved text and matching retrieved content. No real messages, refunds or account changes are performed.

## Architecture
1. CLI loads settings and validates each ticket.
2. Preflight trims input and masks detected payment-card numbers.
3. The model selects tools and returns a validated analysis.
4. Policies enforce urgency, ownership and response safety; evidence review checks resolution and approved replies.
5. The application calculates SLA/deadline information and returns a result plus an action plan.

| Start here | Responsibility |
|---|---|
| `triage/__main__.py`, `factory.py` | Entry point and dependency assembly |
| `batch.py`, `agent.py` | Batch failure isolation and bounded model/tool loop |
| `schemas.py`, `llm.py`, `openai_llm.py` | Contracts and provider adapter |
| `tools/` | Validated read-only tools and keyword retrieval |
| `policy.py`, `decision_review.py`, `reply_guard.py` | Rules, evidence review and risky-wording checks |
| `approved_replies.py`, `actions.py`, `report.py` | Approved text, local action plans and output |

See [ARCHITECTURE.md](ARCHITECTURE.md) for design decisions, risks and the production evaluation plan.

## Setup
Requires Python 3.11+. Run from the folder containing this README.
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
On macOS/Linux, activate with `source .venv/bin/activate` instead.

## Run the Application (no API required)
```powershell
# Original three tickets, no API required
python -m triage --mode mock
# Three additional reasoning cases
python -m triage --mode mock --tickets data/additional_tickets.json
# Save JSON with tool evidence
python -m triage --mode mock --json --traces --output results.json
```

## Run the Application (OpenAI API Key required)
Live mode: first copy .env.example to .env and set your OPENAI API key/model
```powershell
cp .env.example .env 
```

Running command
```powershell
# Run all three tickets
python -m triage --mode openai
# Run all three tickets with json format output
python -m triage --mode openai --json
# Run specific tickets with json format output
python -m triage --mode openai --ticket-id T-002 --json
# Run additional tickets for testing
python -m triage --mode openai --tickets data/additional_tickets.json --json --traces
```
Default mode is `openai`; always select `mock` for offline use. `--ticket-id T-002` selects one ticket. `--execute` writes local JSONL outbox files only. `--retry-failed FILE` retries saved failures. Exit codes: 0 success, 2 configuration/input error, 3 processing/persistence alert.

## Run Tests
```powershell
python -m pytest -q
python -m drills
```
Tests use doubles, not live API calls. Drills inject failures such as unavailable models and broken tools. Optional live evaluation: `python -m evals --runs 3` (API charges apply).

## Configuration
| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Required only for live mode; keep `.env` private |
| `OPENAI_MODEL`, `OPENAI_REASONING_EFFORT` | Account-accessible model and supported reasoning setting |
| `TRIAGE_AUTO_RESPOND` | `false` holds all replies for human review |
| `TRIAGE_MAX_STEPS`, `TRIAGE_KB_TOP_K` | Model rounds and retrieved article count |
| `OPENAI_TIMEOUT_SECONDS`, `TRIAGE_TIMEOUT_SECONDS` | Request and cooperative ticket time budgets |
| `TRIAGE_BREAKER_THRESHOLD`, `TRIAGE_ALERT_RATIO` | Batch failure controls |
| `TRIAGE_DEAD_LETTER`, `TRIAGE_OUTBOX` | Retry records and local action files |

See `.env.example` and `triage/config.py`. Defaults: six rounds, twelve tool calls per ticket, 4,000 output tokens per model request. Timeout checks cannot interrupt a running call.

## Evaluation and Limits
- Original offline cases: T-001 high/billing escalation; T-002 critical/incident response; T-003 medium/technical support.
- Additional targets are stored separately in `evals/additional_expected.json`. Mock currently returns low for all three; it exercises plumbing, not general reasoning quality.
- `evals/additional_*_results.json` are historical runs, not current-version accuracy claims. One live run over-escalated a single-user password reset after consulting regional monitoring.
- Retrieval is English keyword matching. A matching article ID does not prove relevance or support every sentence.
- Rules and phrase checks can miss negation or context; approved automatic replies cover only narrow export questions. `confidence` is not a calibrated probability.
- No production accuracy/latency claims. Data authentication, fresh monitoring, complete privacy controls and transactional delivery remain future work.
