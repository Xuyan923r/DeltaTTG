"""deltattg credit assignment for verl-agent's step-independent rollout rows."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

import numpy as np
import torch

from deltattg.meter.prompt import build_meter_messages

from deltattg.integration.adapters.alfworld import (
    ADAPTER_NAME as ALFWORLD_ADAPTER_NAME,
    METER_TOOLS as ALFWORLD_METER_TOOLS,
    build_alfworld_meter_messages,
)
from deltattg.integration.adapters.papersearchqa import (
    ADAPTER_NAME as PAPERSEARCHQA_ADAPTER_NAME,
    METER_TOOLS as PAPERSEARCHQA_METER_TOOLS,
    build_papersearchqa_meter_messages,
)
from deltattg.meter.chat_template import apply_chat_template_compat
from deltattg.credit.cost import (
    completion_log_odds_to_cost,
    completion_terminal_cost,
    feasibility_to_cost,
    estimate_to_cost,
    terminal_cost,
    value_to_cost,
)
from deltattg.integration.deduplication import (
    deduplicate_meter_requests,
    restore_deduplicated_estimates,
)
from deltattg.credit.progress import compute_step_progress
from deltattg.credit.step import build_credit_mass, normalize_step_credit
from deltattg.credit.token import build_token_credit_multipliers
from deltattg.types import MeterRequest


def _trajectory_advantage(advantages, response_mask, rows, epsilon):
    """Return the shared scalar, or None when per-action rewards broke that assumption."""
    values = torch.cat([advantages[row][response_mask[row]] for row in rows])
    if values.numel() == 0:
        return 0.0
    if not torch.isfinite(values).all():
        raise ValueError('DeltaTTG requires finite input advantages')
    if not torch.allclose(values, values[0].expand_as(values), rtol=0.0, atol=epsilon):
        return None
    return float(values[0].item())


def _optional_non_tensor(data, key: str, row: int):
    values = data.non_tensor_batch.get(key)
    if values is None:
        return None
    value = values[row]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _trajectory_outcome_reward(data, rows: list[int]) -> float:
    """Use the observed score, never a success flag or shaped per-action reward."""
    rewards = data.non_tensor_batch.get("outcome_reward")
    if rewards is None:
        raise ValueError("outcome_value credit requires rollout outcome_reward metadata")
    values = [float(rewards[row]) for row in rows]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Trajectory outcome_reward must be finite")
    if any(value != values[0] for value in values):
        raise ValueError("Trajectory rows disagree on outcome_reward")
    return values[0]


def _count_with_meter_tokenizer(tokenizer, text: str) -> int:
    """Count an actor turn in the meter model's token space.

    Actor and meter tokenizers happen to be compatible in the original
    Qwen/Qwen experiment, but that assumption is false for a Llama actor with
    the frozen Qwen meter.  The usage summary is part of the meter
    input, so it must use the same tokenization convention as the meter's
    SFT targets instead of the actor's response-mask length.
    """

    if hasattr(tokenizer, "encode"):
        return len(tokenizer.encode(text, add_special_tokens=False))
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return len(input_ids)




def _build_requests(data, tokenizer, meter_tokenizer, include_usage_summary: bool, enable_thinking: bool,
                    output_mode: str = "canonical_generation"):
    decoded = tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)
    trajectories: dict[str, list[int]] = defaultdict(list)
    for row_index, trajectory_id in enumerate(data.non_tensor_batch["traj_uid"]):
        trajectories[str(trajectory_id)].append(row_index)
    step_indices = data.non_tensor_batch["environment_step_index"]
    for row_indices in trajectories.values():
        row_indices.sort(key=lambda index: int(step_indices[index]))

    requests: list[MeterRequest] = []
    request_to_row: list[int] = []
    for trajectory_id, row_indices in trajectories.items():
        completed_history: list[dict[str, str]] = []
        completed_counts: list[int] = []
        task = str(data.non_tensor_batch["environment_task"][row_indices[0]])
        adapter_values = data.non_tensor_batch.get("environment_prefix_adapter")
        adapter_name = (
            str(adapter_values[row_indices[0]])
            if adapter_values is not None
            else "default"
        )
        if adapter_values is not None and any(
            str(adapter_values[row]) != adapter_name for row in row_indices
        ):
            raise ValueError(
                f"Trajectory {trajectory_id} changes meter prefix adapter"
            )
        for step_index, row_index in enumerate(row_indices):
            if step_index > 0:
                completed_history.append(
                    {
                        "role": "user",
                        "content": (
                            "<environment_observation>\n"
                            + str(
                                data.non_tensor_batch["environment_observation"][
                                    row_index
                                ]
                            )
                            + "\n</environment_observation>"
                        ),
                    }
                )
            usage_counts = (
                list(completed_counts) if include_usage_summary else None
            )
            initial_observation = str(
                data.non_tensor_batch["environment_observation"][row_indices[0]]
            )
            if adapter_name == ALFWORLD_ADAPTER_NAME:
                messages = build_alfworld_meter_messages(
                    original_task=task,
                    initial_observation=initial_observation,
                    history_messages=list(completed_history),
                    assistant_token_counts=usage_counts,
                    completed_turns=step_index,
                    output_mode=output_mode,
                )
                request_tools = ALFWORLD_METER_TOOLS
            elif adapter_name == PAPERSEARCHQA_ADAPTER_NAME:
                messages = build_papersearchqa_meter_messages(
                    original_task=task,
                    initial_observation=initial_observation,
                    history_messages=list(completed_history),
                    assistant_token_counts=usage_counts,
                    completed_turns=step_index,
                    output_mode=output_mode,
                )
                request_tools = PAPERSEARCHQA_METER_TOOLS
            else:
                messages = build_meter_messages(
                    original_agent_system=(
                        "verl-agent step-independent environment policy."
                    ),
                    tools=[],
                    original_task=task,
                    initial_observation=initial_observation,
                    history_messages=list(completed_history),
                    assistant_token_counts=usage_counts,
                    completed_turns=step_index,
                    output_mode=output_mode,
                )
                request_tools = []
            serialized_prompt = apply_chat_template_compat(
                meter_tokenizer,
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            input_ids = apply_chat_template_compat(
                meter_tokenizer,
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            requests.append(
                MeterRequest(
                    request_id=f"{trajectory_id}:step:{step_index}",
                    trajectory_id=trajectory_id,
                    step_idx=step_index,
                    messages=messages,
                    tools=request_tools,
                    serialized_prompt=serialized_prompt,
                    input_ids=list(input_ids),
                )
            )
            request_to_row.append(row_index)
            assistant_text = decoded[row_index]
            completed_history.append({"role": "assistant", "content": assistant_text})
            # Meter targets and this prompt-side usage metadata must share
            # one tokenizer convention.  This is essential when the actor is
            # Llama while the frozen meter is Qwen.
            completed_counts.append(
                _count_with_meter_tokenizer(
                    meter_tokenizer,
                    assistant_text,
                )
            )
    return trajectories, requests, request_to_row


def apply_verl_agent_credit_assignment(
    data,
    *,
    actor_tokenizer,
    credit_config: Any,
    meter_config: Any,
    meter_client,
    failure_cost: int,
):
    """Redistribute a trajectory's vanilla GRPO advantage over its step rows."""

    if not bool(credit_config.get("enabled", False)):
        return data, {}
    required = {
        "traj_uid",
        "environment_task",
        "environment_observation",
        "environment_step_index",
        "trajectory_success",
    }
    missing = required.difference(data.non_tensor_batch)
    if missing:
        raise ValueError(f"verl-agent rollout is missing deltattg metadata: {sorted(missing)}")
    if failure_cost <= 0:
        raise ValueError("deltattg failure_cost must be positive")
    state_signal = str(
        credit_config.get("state_signal", "remaining_action_tokens")
    )
    if state_signal not in {
        "remaining_action_tokens",
        "binary_feasibility",
        "completion_log_odds",
        "outcome_value",
    }:
        raise ValueError(f"Unsupported deltattg state_signal: {state_signal}")
    meter_output_mode = str(
        meter_config.get("output_mode", "canonical_generation")
    )
    if state_signal == "completion_log_odds" and meter_output_mode != state_signal:
        raise ValueError(
            "credit_assignment.state_signal=completion_log_odds requires "
            "meter.output_mode=completion_log_odds"
        )
    if (state_signal == "outcome_value") != (meter_output_mode == "value_generation"):
        raise ValueError(
            "credit_assignment.state_signal=outcome_value and "
            "meter.output_mode=value_generation must be selected together"
        )
    log_odds_clip = float(credit_config.get("log_odds_clip", 10.0))
    if state_signal == "completion_log_odds" and log_odds_clip <= 0:
        raise ValueError("credit_assignment.log_odds_clip must be positive")

    response_mask = data.batch["response_mask"].bool()
    original_advantages = data.batch["advantages"]
    trajectories, all_requests, all_request_rows = _build_requests(
        data,
        actor_tokenizer,
        meter_client.tokenizer,
        bool(meter_config.get("include_usage_summary", True)),
        bool(meter_config.get("enable_thinking", False)),
        output_mode=meter_output_mode,
    )
    trajectory_outcomes = (
        {trajectory_id: _trajectory_outcome_reward(data, rows)
         for trajectory_id, rows in trajectories.items()}
        if state_signal == "outcome_value" else {}
    )
    skip_zero = bool(credit_config.get("skip_zero_advantage_trajectories", True))
    epsilon = float(credit_config.get("zero_advantage_epsilon", 1e-8))
    trajectory_advantages = {
        trajectory_id: _trajectory_advantage(original_advantages, response_mask, rows, epsilon)
        for trajectory_id, rows in trajectories.items()
    }
    nonuniform_trajectory_ids = {
        trajectory_id for trajectory_id, scalar in trajectory_advantages.items()
        if scalar is None
    }
    zero_advantage_trajectories = {
        trajectory_id
        for trajectory_id, scalar in trajectory_advantages.items()
        if skip_zero and scalar is not None and abs(scalar) <= epsilon
    }
    skipped_trajectories = zero_advantage_trajectories | nonuniform_trajectory_ids
    meter_trajectory_ids = [
        trajectory_id
        for trajectory_id in trajectories
        if trajectory_id not in skipped_trajectories
    ]
    meter_trajectory_id_set = set(meter_trajectory_ids)
    selected = [
        (request, row)
        for request, row in zip(all_requests, all_request_rows, strict=True)
        if request.trajectory_id in meter_trajectory_id_set
    ]
    raw_requests = [item[0] for item in selected]
    request_to_row = [item[1] for item in selected]
    if bool(meter_config.get("enable_prefix_deduplication", True)):
        unique_requests, inverse = deduplicate_meter_requests(raw_requests)
    else:
        unique_requests, inverse = raw_requests, list(range(len(raw_requests)))
    unique_estimates = (
        meter_client.estimate_batch(unique_requests) if unique_requests else []
    )
    estimates = restore_deduplicated_estimates(unique_estimates, inverse)
    if len(estimates) != len(raw_requests):
        raise ValueError("Meter output count does not match verl-agent step count")

    invalid_rate = (
        sum(not estimate.valid for estimate in unique_estimates) / len(unique_estimates)
        if unique_estimates
        else 0.0
    )
    invalid_trajectory_ids = {
        request.trajectory_id
        for request, estimate in zip(raw_requests, estimates, strict=True)
        if not estimate.valid
    }
    failed_group_trajectory_ids: set[str] = set()
    if invalid_trajectory_ids:
        group_ids = data.non_tensor_batch.get("uid")
        if group_ids is None:
            raise ValueError("Prompt-group fallback requires rollout uid metadata")
        trajectory_groups: dict[str, str] = {}
        for trajectory_id, rows in trajectories.items():
            group_id = str(group_ids[rows[0]])
            if any(str(group_ids[row]) != group_id for row in rows):
                raise ValueError(f"Trajectory {trajectory_id} changes prompt-group uid")
            trajectory_groups[trajectory_id] = group_id
        failed_groups = {
            trajectory_groups[trajectory_id]
            for trajectory_id in invalid_trajectory_ids
        }
        failed_group_trajectory_ids = {
            trajectory_id
            for trajectory_id, group_id in trajectory_groups.items()
            if group_id in failed_groups
        }

    estimate_by_row = dict(zip(request_to_row, estimates, strict=True))
    request_by_row = dict(zip(all_request_rows, all_requests, strict=True))
    new_advantages = torch.zeros_like(original_advantages)
    fallback_count = 0
    meter_error_group_fallback_count = 0
    all_failure_fallback_count = 0
    clip_rates: list[float] = []
    for trajectory_id, rows in trajectories.items():
        scalar = trajectory_advantages[trajectory_id]
        if scalar is None:
            # Per-action penalties can create different advantages within one
            # trajectory. Preserve them exactly rather than selecting one row.
            new_advantages[rows] = original_advantages[rows]
            fallback_count += 1
            meter_error_group_fallback_count += int(trajectory_id in failed_group_trajectory_ids)
            clip_rates.append(0.0)
            continue
        is_zero = abs(scalar) <= epsilon
        success = bool(data.non_tensor_batch["trajectory_success"][rows[0]])
        meter_error_group_bypass = trajectory_id in failed_group_trajectory_ids
        meter_selected = all(row in estimate_by_row for row in rows)
        trajectory_estimates = (
            [estimate_by_row[row] for row in rows]
            if not is_zero
            and not meter_error_group_bypass and meter_selected
            else []
        )
        all_estimated_failure = bool(trajectory_estimates) and all(
            not estimate.can_complete for estimate in trajectory_estimates
        )
        guarded_all_failure = (
            state_signal == "remaining_action_tokens"
            and all_estimated_failure
        )
        fallback = (
            meter_error_group_bypass or is_zero
            or not meter_selected or guarded_all_failure
        )
        costs: list[float] | None = None
        progress: torch.Tensor | None = None
        mass: torch.Tensor | None = None
        if state_signal == "completion_log_odds":
            final_cost = completion_terminal_cost(success, log_odds_clip)
        elif state_signal == "outcome_value":
            final_cost = -trajectory_outcomes[trajectory_id]
        else:
            final_cost = terminal_cost(success, float(failure_cost))
        if fallback:
            row_tokens = torch.tensor(
                [int(response_mask[row].sum().item()) for row in rows],
                dtype=torch.float32,
            )
            # Every fallback preserves vanilla GRPO, irrespective of action length.
            shares = row_tokens / row_tokens.sum()
            fallback_count += int(
                abs(scalar) > epsilon
            )
            meter_error_group_fallback_count += int(meter_error_group_bypass)
            all_failure_fallback_count += int(
                abs(scalar) > epsilon and guarded_all_failure
            )
        else:
            if state_signal == "completion_log_odds":
                if any(
                    estimate.completion_log_odds is None
                    for estimate in trajectory_estimates
                ):
                    raise RuntimeError(
                        "A valid completion-log-odds estimate is missing its score"
                    )
                trajectory_log_odds = [
                    float(estimate.completion_log_odds)
                    for estimate in trajectory_estimates
                    if estimate.completion_log_odds is not None
                ]

                costs = [
                    completion_log_odds_to_cost(value, log_odds_clip)
                    for value in trajectory_log_odds
                ]
            elif state_signal == "outcome_value":
                costs = [value_to_cost(estimate) for estimate in trajectory_estimates]
            elif state_signal == "binary_feasibility":
                costs = [
                    feasibility_to_cost(estimate, float(failure_cost))
                    for estimate in trajectory_estimates
                ]
            else:
                costs = [
                    estimate_to_cost(
                        estimate, float(failure_cost),
                        enable_cost_clipping=bool(credit_config.get("enable_cost_clipping", False)),
                    )
                    for estimate in trajectory_estimates
                ]

            _, progress = compute_step_progress(
                costs,
                final_cost,
                progress_clip=(float(failure_cost) if state_signal == "remaining_action_tokens"
                               and bool(credit_config.get("enable_cost_clipping", False)) else None),
            )

            mass = build_credit_mass(
                progress,
                scalar,
                str(credit_config.get("mode", "sign_aware")),
            )
            shares = normalize_step_credit(
                mass,
                float(credit_config.get("uniform_credit_mix", 0.2)),
            )

        row_width = response_mask.shape[1]
        flat_mask = torch.cat([response_mask[row] for row in rows])
        positions: list[list[int]] = []
        for local_index, row in enumerate(rows):
            base = local_index * row_width
            positions.append(
                (torch.nonzero(response_mask[row], as_tuple=False).flatten() + base).tolist()
            )
        if fallback:
            multipliers = flat_mask.to(dtype=shares.dtype)
            clip_rate = 0.0
        else:
            multipliers, clip_rate = build_token_credit_multipliers(
                shares.to(flat_mask.device),
                positions,
                response_length=flat_mask.numel(),
                trainable_mask=flat_mask,
                max_token_multiplier=credit_config.get("max_token_multiplier", 5.0),
                token_allocation_mode=credit_config.get("token_allocation_mode", "step_broadcast"),
            )
        clip_rates.append(clip_rate)
        for local_index, row in enumerate(rows):
            start = local_index * row_width
            if fallback:
                new_advantages[row] = original_advantages[row]
            else:
                new_advantages[row] = scalar * multipliers[
                    start : start + row_width
                ].to(new_advantages.device)


    data.batch["advantages"] = new_advantages
    data.batch["returns"] = new_advantages.clone()
    metrics = {
        "credit/trajectory_count": len(trajectories),
        "credit/meter_selected_trajectory_count": len(
            meter_trajectory_id_set
        ),
        "credit/meter_raw_prefix_count": len(raw_requests),
        "credit/meter_unique_prefix_count": len(unique_requests),
        "credit/meter_invalid_output_rate": invalid_rate,
        "credit/meter_error_group_fallback_count": meter_error_group_fallback_count,
        "credit/uniform_fallback_rate": fallback_count / len(trajectories) if trajectories else 0.0,
        "credit/all_estimated_failure_fallback_rate": (
            all_failure_fallback_count / len(trajectories) if trajectories else 0.0
        ),
        "credit/zero_advantage_trajectory_count": len(
            zero_advantage_trajectories
        ),
        "credit/nonuniform_advantage_fallback_count": len(nonuniform_trajectory_ids),
        "credit/token_multiplier_clip_rate": float(np.mean(clip_rates)) if clip_rates else 0.0,
    }
    metrics.update(
        {f"credit/{key}": value for key, value in meter_client.last_metrics.items()}
    )
    return data, metrics
