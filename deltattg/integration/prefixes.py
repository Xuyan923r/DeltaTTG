"""Build post-rollout meter requests with the exact SFT prompt builder."""

from __future__ import annotations

from copy import deepcopy

from deltattg.meter.prompt import build_meter_messages

from deltattg.types import MeterRequest, TrajectoryRecord


def _tool_call_from_message(message: dict) -> dict | None:
    calls = message.get("tool_calls") or []
    return deepcopy(calls[0]) if calls else None


def _history_messages(trajectory: TrajectoryRecord, state_idx: int) -> list[dict]:
    """Represent completed steps without duplicating structured tool calls."""
    messages = []
    for step in trajectory.steps[:state_idx]:
        assistant = step.assistant_message
        tool_call = _tool_call_from_message(assistant)
        if tool_call is not None:
            observation = (step.tool_or_environment_messages[0]
                           if step.tool_or_environment_messages else {})
            call_id = tool_call.get('id')
            messages.append({'role': 'assistant', 'content': '', 'tool_calls': [tool_call]})
            messages.append({
                'role': 'tool',
                'content': str(observation.get('content') or ''),
                'tool_call_id': str(call_id) if call_id is not None else None,
                'name': str(tool_call.get('function', {}).get('name')),
            })
        else:
            messages.append({'role': 'assistant', 'content': str(assistant.get('content') or '')})
    return messages


def build_meter_requests_from_completed_trajectory(
    trajectory: TrajectoryRecord,
    *,
    tokenizer=None,
    include_usage_summary: bool = True,
    enable_thinking: bool = False,
) -> list[MeterRequest]:
    if not trajectory.steps:
        raise ValueError("Completed trajectory has no assistant steps")
    if not trajectory.termination_reason:
        raise ValueError("Meter requests require a completed trajectory")
    requests: list[MeterRequest] = []
    for state_idx in range(len(trajectory.steps)):
        completed_counts = [
            len(step.assistant_output_token_ids) for step in trajectory.steps[:state_idx]
        ]
        messages = build_meter_messages(
            original_agent_system=trajectory.original_agent_system,
            tools=trajectory.tools,
            original_task=trajectory.original_task,
            initial_observation=trajectory.initial_observation,
            history_messages=_history_messages(trajectory, state_idx),
            assistant_token_counts=(
                completed_counts if include_usage_summary else None
            ),
            completed_turns=state_idx,
        )
        serialized_prompt = None
        input_ids = None
        if tokenizer is not None:
            serialized_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        requests.append(
            MeterRequest(
                request_id=f"{trajectory.trajectory_id}:step:{state_idx}",
                trajectory_id=trajectory.trajectory_id,
                step_idx=state_idx,
                messages=messages,
                tools=deepcopy(trajectory.tools),
                serialized_prompt=serialized_prompt,
                input_ids=list(input_ids) if input_ids is not None else None,
            )
        )
    return requests


def build_meter_requests_from_completed_rollouts(
    trajectories: list[TrajectoryRecord],
    *,
    tokenizer=None,
    include_usage_summary: bool = True,
    enable_thinking: bool = False,
) -> list[MeterRequest]:
    requests = []
    for trajectory in trajectories:
        requests.extend(
            build_meter_requests_from_completed_trajectory(
                trajectory,
                tokenizer=tokenizer,
                include_usage_summary=include_usage_summary,
                enable_thinking=enable_thinking,
            )
        )
    return requests
