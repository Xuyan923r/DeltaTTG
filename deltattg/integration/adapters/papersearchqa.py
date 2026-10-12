"""PaperSearchQA meter prefix adapter matching meter training data."""

from __future__ import annotations

from typing import Any

from deltattg.meter.prompt import build_meter_messages
from agent_system.environments.protocols.papersearchqa import (
    METER_PREFIX_ADAPTER,
    METER_SYSTEM_PROMPT,
    SEARCH_TOOL_SCHEMA,
)


ADAPTER_NAME = METER_PREFIX_ADAPTER
METER_TOOLS = [SEARCH_TOOL_SCHEMA]


def build_papersearchqa_meter_messages(
    *,
    original_task: str,
    initial_observation: str,
    history_messages: list[dict[str, Any]],
    assistant_token_counts: list[int] | None,
    completed_turns: int,
    output_mode: str = "canonical_generation",
) -> list[dict[str, Any]]:
    """Build the PaperSearchQA progress-meter prompt shared with training."""

    messages = build_meter_messages(
        original_agent_system=METER_SYSTEM_PROMPT,
        tools=METER_TOOLS,
        original_task=original_task,
        initial_observation=initial_observation,
        history_messages=history_messages,
        assistant_token_counts=assistant_token_counts,
        completed_turns=completed_turns,
        output_mode=output_mode,
    )
    return messages
