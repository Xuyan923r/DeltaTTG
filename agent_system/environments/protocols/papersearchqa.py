"""Canonical PaperSearchQA meter envelope used at train and inference."""

from __future__ import annotations

from typing import Any


METER_PREFIX_ADAPTER = "papersearchqa_training_envelope_v1"
METER_SYSTEM_PROMPT = "You are a helpful and harmless assistant."

SEARCH_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the PubMed BM25 corpus for supporting evidence.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    "agent_action_format": "<search>QUERY</search>",
}
