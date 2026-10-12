"""Prompt construction shared by dataset generation and deltattg inference."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


def build_initial_context(
    *,
    original_agent_system: str,
    tools: list[dict[str, Any]],
    original_task: str,
    initial_observation: str,
) -> str:
    canonical_tools = json.dumps(
        tools, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return (
        "<original_agent_system>\n"
        f"{original_agent_system}\n"
        "</original_agent_system>\n\n"
        "<available_tools>\n"
        f"{canonical_tools}\n"
        "</available_tools>\n\n"
        "<original_task>\n"
        f"{original_task}\n"
        "</original_task>\n\n"
        "<initial_observation>\n"
        f"{initial_observation}\n"
        "</initial_observation>"
    )


def build_estimation_query(
    *,
    completed_turns: int,
    usage_summary: str,
    output_mode: str = "canonical_generation",
) -> str:
    if output_mode == "value_generation":
        return f"""Estimate the final outcome score from the current agent state.

Completed agent turns: {completed_turns}

Observed fresh-token usage in completed turns:
{usage_summary}

Use the task's final-reward scale; higher scores mean better outcomes.
Return only <answer>{{"value":NUMBER}}</answer> or a single finite NUMBER."""
    return f"""Estimate task progress from the current agent state.

Completed agent turns: {completed_turns}

Observed fresh-token usage in completed turns:
{usage_summary}

Estimate whether the continuation will complete the task successfully. If
success is estimated, estimate the remaining completion effort as an integer
number of additional assistant-generated action tokens typically needed until
completion. Count assistant/agent outputs only.
Do not count current or future tool/environment observation tokens.

Return only the required <answer> JSON."""


def build_meter_messages(
    *,
    original_agent_system: str,
    tools: list[dict[str, Any]],
    original_task: str,
    initial_observation: str,
    history_messages: list[dict[str, Any]],
    assistant_token_counts: list[int] | None,
    completed_turns: int | None = None,
    output_mode: str = "canonical_generation",
) -> list[dict[str, Any]]:
    if assistant_token_counts is not None and len(assistant_token_counts) > len(
        history_messages
    ):
        raise ValueError("Assistant-token count metadata exceeds history length")
    if assistant_token_counts is None:
        usage_summary = "Not provided."
    elif not assistant_token_counts:
        usage_summary = "None."
    else:
        usage_summary = "\n".join(
            f"turn_{index}: assistant_output_tokens={assistant_count}"
            for index, assistant_count in enumerate(assistant_token_counts)
        )
    if completed_turns is None:
        completed_turns = len(assistant_token_counts or [])
    from deltattg.meter.schema import METER_SYSTEM_PROMPT, VALUE_SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": (
            VALUE_SYSTEM_PROMPT if output_mode == "value_generation" else METER_SYSTEM_PROMPT
        )},
        {
            "role": "user",
            "content": build_initial_context(
                original_agent_system=original_agent_system,
                tools=tools,
                original_task=original_task,
                initial_observation=initial_observation,
            ),
        },
    ]
    messages.extend(deepcopy(history_messages))
    messages.append(
        {
            "role": "user",
            "content": build_estimation_query(
                completed_turns=completed_turns,
                usage_summary=usage_summary,
                output_mode=output_mode,
            ),
        }
    )
    return messages
