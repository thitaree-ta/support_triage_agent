"""Assemble tools, model adapter, policies and batch processing from settings."""
from __future__ import annotations

from triage.agent import TriageAgent
from triage.batch import BatchRunner, TicketPipeline
from triage.breaker import CircuitBreaker
from triage.deadletter import DeadLetterWriter
from triage.preflight import Preflight
from triage.config import Settings
from triage.datastore import load_articles, load_json
from triage.llm import LLMClient
from triage.policy import PolicyEngine
from triage.prompts import load_system_prompt
from triage.sla import SlaPolicy
from triage.tools import (
    BillingTool, CustomerHistoryTool, KnowledgeBaseTool, ServiceStatusTool, ToolRegistry,
)
from triage.tools.retriever import KeywordRetriever


def load_sla(settings: Settings) -> SlaPolicy:
    return SlaPolicy.from_records(load_json(settings.data_dir / "sla.json"))


def build_registry(settings: Settings) -> ToolRegistry:
    data = settings.data_dir
    retriever = KeywordRetriever(load_articles(data / "kb.json"))
    return ToolRegistry(
        [
            KnowledgeBaseTool(retriever, settings.kb_top_k),
            CustomerHistoryTool(load_json(data / "customers.json")),
            BillingTool(load_json(data / "billing.json")),
            ServiceStatusTool(load_json(data / "service_status.json")),
        ]
    )


def build_agent(settings: Settings, llm: LLMClient | None = None) -> TriageAgent:
    if llm is None:
        from triage.openai_llm import OpenAIClient

        llm = OpenAIClient(settings.model, settings.reasoning_effort, request_timeout=settings.request_timeout_seconds)
    return TriageAgent(
        llm=llm,
        registry=build_registry(settings),
        policy=PolicyEngine.default(settings.low_confidence_threshold, settings.auto_respond_enabled),
        system_prompt=load_system_prompt(settings.prompt_path),
        max_steps=settings.max_agent_steps,
        sla_policy=load_sla(settings),
        ticket_timeout_seconds=settings.ticket_timeout_seconds,
    )


def build_pipeline(
    settings: Settings, agent: TriageAgent | None = None, breaker: CircuitBreaker | None = None
) -> tuple[TicketPipeline, CircuitBreaker]:
    agent = agent or build_agent(settings)
    breaker = breaker or CircuitBreaker(settings.breaker_threshold)
    pipeline = TicketPipeline(agent, Preflight(settings.max_message_chars), breaker, load_sla(settings))
    return pipeline, breaker


def build_batch_runner(settings: Settings, agent: TriageAgent | None = None) -> BatchRunner:
    pipeline, breaker = build_pipeline(settings, agent)
    return BatchRunner(pipeline, breaker, DeadLetterWriter(settings.dead_letter_path))
