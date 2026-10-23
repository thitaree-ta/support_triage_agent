"""Tool `search_knowledge_base`: finds help articles that match a question.

The AI may only answer customers using articles returned here. The agent's policy layer later
checks that every article the AI cites was REALLY returned by this tool.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ConfigDict

from triage.tools.base import Tool, ToolContext
from triage.tools.retriever import Retriever

TOOL_NAME = "search_knowledge_base"


class SearchKBParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=500, description="Short English keyword query, e.g. 'duplicate pending charge refund'.")


class KnowledgeBaseTool(Tool):
    name = TOOL_NAME
    description = (
        "Search the support FAQ/docs. Write the query in English even if the ticket is in another "
        "language. Returns the most relevant articles with their full text. Only cite article ids "
        "returned here."
    )
    params_model = SearchKBParams

    def __init__(self, retriever: Retriever, top_k: int = 3) -> None:
        self._retriever = retriever
        self._top_k = top_k

    def run(self, params: SearchKBParams, ctx: ToolContext) -> dict[str, Any]:
        hits = self._retriever.search(params.query, self._top_k)
        return {
            "results": [
                {
                    "id": h.article.id,
                    "title": h.article.title,
                    "content": h.article.body,
                    "score": round(h.score, 2),
                }
                for h in hits
            ],
            "note": None if hits else "No relevant articles found. Do not answer from memory.",
        }
