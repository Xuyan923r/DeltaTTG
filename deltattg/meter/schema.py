"""Canonical output schema for task-progress estimates."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


SCHEMA_VERSION = "continuation_action_tokens_v1"

METER_SYSTEM_PROMPT = """You are an agent task progress meter.

You receive the original task, available tools, and the complete observable
trajectory prefix at a decision boundary.

First estimate whether the continuation from the current state will complete
the task successfully. If success is estimated, estimate the remaining
completion effort as one integer: the number of additional assistant-generated
action tokens typically needed until successful completion. Base the estimate
on the task and observed interaction history, using experience learned from
diverse agent trajectories.

Count only future assistant-generated action tokens, including reasoning text,
normal assistant text, tool-call names and arguments, and a final answer when
one is produced. Never count current or future tool/environment observations,
system prompts, user messages, or tokens already present in the prefix.

Output exactly one of these formats:

<answer>{"can_complete":true,"remaining_action_tokens":INTEGER}</answer>
<answer>{"can_complete":false,"remaining_action_tokens":null}</answer>

Do not solve the task, call tools, explain the answer, or output extra text."""

_ANSWER_PATTERN = re.compile(
    r'<answer>\{"can_complete":(true|false),"remaining_action_tokens":'
    r'(null|[0-9]+)\}</answer>'
)

VALUE_SYSTEM_PROMPT = """You are an agent task progress meter.

You receive the original task, available tools, and the complete observable
trajectory prefix at a decision boundary.

Predict the final outcome score of a continuation from this state, using the
same score scale as the task's observed final reward. Higher scores mean better
outcomes. Predict the outcome score, not the remaining action-token count.

Output exactly <answer>{"value":NUMBER}</answer> or a single NUMBER.
NUMBER must be finite. Do not solve the task, call tools, explain the answer,
or output extra text."""


@dataclass(frozen=True)
class TokenEstimate:
    can_complete: bool
    remaining_action_tokens: int | None


def build_target_text(
    can_complete: bool,
    remaining_action_tokens: int | None,
) -> str:
    if can_complete:
        if (
            remaining_action_tokens is None
            or isinstance(remaining_action_tokens, bool)
            or remaining_action_tokens < 0
        ):
            raise ValueError(
                "Completable targets require non-negative remaining action tokens"
            )
        payload = {
            "can_complete": True,
            "remaining_action_tokens": int(remaining_action_tokens),
        }
    else:
        if remaining_action_tokens is not None:
            raise ValueError("Non-completable targets require a null token count")
        payload = {
            "can_complete": False,
            "remaining_action_tokens": None,
        }
    return (
        "<answer>"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "</answer>"
    )


def parse_strict_answer(text: str) -> TokenEstimate | None:
    """Accept only the canonical point-estimate answer."""
    match = _ANSWER_PATTERN.fullmatch(text)
    if match is None:
        return None
    can_complete = match.group(1) == "true"
    token_text = match.group(2)
    if not can_complete:
        if token_text != "null":
            return None
        return TokenEstimate(False, None)
    if token_text == "null":
        return None
    return TokenEstimate(True, int(token_text))
