"""Post-rollout continuation-cost credit assignment integration."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from deltattg.credit.cost import estimate_to_cost, terminal_cost
from deltattg.integration.deduplication import (
    deduplicate_meter_requests,
    restore_deduplicated_estimates,
)
from deltattg.integration.prefixes import build_meter_requests_from_completed_rollouts
from deltattg.credit.progress import compute_step_progress
from deltattg.credit.step import build_credit_mass, normalize_step_credit
from deltattg.credit.token import build_token_credit_multipliers, uniform_token_multipliers
from deltattg.types import MeterResult, TrajectoryRecord


def _scalar_advantage(advantages: torch.Tensor, mask: torch.Tensor) -> float:
    positions = torch.nonzero(mask.bool(), as_tuple=False).flatten()
    if positions.numel() == 0:
        return 0.0
    return float(advantages[int(positions[0])].item())






def apply_continuation_credit_assignment(
    data,
    *,
    credit_config: Any,
    meter_config: Any,
    meter_client,
    max_assistant_generated_tokens: int,
) -> tuple[Any, dict[str, float | int]]:
    """Redistribute already-computed GRPO advantage over agent steps.

    This function must be called only after the complete rollout batch, outcome
    reward computation, and vanilla GRPO group normalization have finished.
    """
    if max_assistant_generated_tokens <= 0:
        raise ValueError(
            "GRPO max_assistant_generated_tokens must be a positive "
            "assistant-token budget"
        )
    if not bool(credit_config.get("enabled", False)):
        return data, {}
    if "deltattg_trajectory" not in data.non_tensor_batch:
        raise ValueError(
            "Continuation credit is enabled but rollout did not provide deltattg_trajectory metadata"
        )
    response_mask = data.batch["response_mask"].bool()
    original_advantages = data.batch["advantages"]
    if response_mask.shape != original_advantages.shape:
        raise ValueError("response_mask and advantages must have identical shape")

    trajectories = [
        value
        if isinstance(value, TrajectoryRecord)
        else TrajectoryRecord.from_dict(value)
        for value in data.non_tensor_batch["deltattg_trajectory"]
    ]
    if len(trajectories) != len(data):
        raise ValueError("Trajectory metadata batch size mismatch")
    mismatched_budgets = [
        trajectory.trajectory_id
        for trajectory in trajectories
        if int(trajectory.max_assistant_generated_tokens)
        != int(max_assistant_generated_tokens)
    ]
    if mismatched_budgets:
        raise ValueError(
            "Rollout trajectory assistant-token budget does not "
            "match trainer budget for trajectories: "
            f"{mismatched_budgets[:3]}"
        )
    advantages = [
        _scalar_advantage(original_advantages[index], response_mask[index])
        for index in range(len(trajectories))
    ]
    skip_zero = bool(
        credit_config.get("skip_zero_advantage_trajectories", True)
    )
    epsilon = float(credit_config.get("zero_advantage_epsilon", 1e-8))
    scored_indices = [
        index
        for index, advantage in enumerate(advantages)
        if not (skip_zero and abs(advantage) <= epsilon)
    ]
    scored_trajectories = [trajectories[index] for index in scored_indices]

    tokenizer = getattr(meter_client, "tokenizer", None)
    raw_requests = build_meter_requests_from_completed_rollouts(
        scored_trajectories,
        tokenizer=tokenizer,
        include_usage_summary=bool(
            meter_config.get("include_usage_summary", True)
        ),
        enable_thinking=bool(meter_config.get("enable_thinking", False)),
    )
    if bool(meter_config.get("enable_prefix_deduplication", True)):
        unique_requests, inverse_mapping = deduplicate_meter_requests(raw_requests)
    else:
        unique_requests = raw_requests
        inverse_mapping = list(range(len(raw_requests)))

    unique_estimates = (
        meter_client.estimate_batch(unique_requests) if unique_requests else []
    )
    if len(unique_estimates) != len(unique_requests):
        raise ValueError("Meter returned a different number of outputs than inputs")
    restored = restore_deduplicated_estimates(
        unique_estimates,
        inverse_mapping,
    )
    estimates_by_trajectory: dict[str, list[MeterResult]] = defaultdict(list)
    for request, estimate in zip(raw_requests, restored, strict=True):
        estimates_by_trajectory[request.trajectory_id].append(estimate)
    failed_groups = {
        trajectory.group_id
        for trajectory in scored_trajectories
        if len(estimates_by_trajectory[trajectory.trajectory_id]) != len(trajectory.steps)
        or any(
            not estimate.valid
            for estimate in estimates_by_trajectory[trajectory.trajectory_id]
        )
    }

    new_advantages = torch.zeros_like(original_advantages)
    clip_rates: list[float] = []
    estimated_successes = 0
    estimated_failures_at_cost = 0
    valid_estimate_count = 0
    uniform_fallbacks = 0
    meter_error_fallbacks = 0

    for index, (trajectory, group_advantage) in enumerate(
        zip(trajectories, advantages, strict=True)
    ):
        mask = response_mask[index]
        fallback_reason = (
            "meter_failure_in_group" if trajectory.group_id in failed_groups else None
        )
        state_costs = None
        step_progress = None
        credit_mass = None
        step_shares = None
        estimates = estimates_by_trajectory.get(trajectory.trajectory_id, [])
        should_be_uniform = abs(group_advantage) <= epsilon

        if should_be_uniform or fallback_reason is not None:
            multipliers = uniform_token_multipliers(mask)
            uniform_fallbacks += int(fallback_reason is not None)
            meter_error_fallbacks += int(fallback_reason is not None)
            clip_rate = 0.0
        else:
            failure_cost = float(max_assistant_generated_tokens)
            nonterminal_costs = [
                estimate_to_cost(
                    estimate, failure_cost,
                    enable_cost_clipping=bool(credit_config.get("enable_cost_clipping", False)),
                )
                for estimate in estimates
            ]
            valid_estimate_count += len(estimates)
            estimated_successes += sum(estimate.success for estimate in estimates)
            estimated_failures_at_cost += sum(
                not estimate.success for estimate in estimates
            )

            state_costs, step_progress = compute_step_progress(
                nonterminal_costs,
                terminal_cost(trajectory.trajectory_success, failure_cost),
                progress_clip=(failure_cost if bool(credit_config.get("enable_cost_clipping", False)) else None),
            )

            credit_mass = build_credit_mass(
                step_progress,
                group_advantage,
                str(credit_config.get("mode", "sign_aware")),
            )
            step_shares = normalize_step_credit(
                credit_mass,
                float(credit_config.get("uniform_credit_mix", 0.2)),
            )


            step_positions = [step.token_positions() for step in trajectory.steps]
            multipliers, clip_rate = build_token_credit_multipliers(
                step_shares.to(mask.device),
                step_positions,
                response_length=mask.numel(),
                trainable_mask=mask,
                max_token_multiplier=credit_config.get("max_token_multiplier", 5.0),
                token_allocation_mode=credit_config.get("token_allocation_mode", "step_broadcast"),
            )
        clip_rates.append(clip_rate)

        new_advantages[index] = (
            float(group_advantage) * multipliers.to(new_advantages.device)
        ).detach()
        trajectory.credit_fallback_reason = fallback_reason

    data.batch["advantages"] = new_advantages
    data.batch["returns"] = new_advantages.clone()
    metrics: dict[str, float | int] = {
        "credit/meter_raw_prefix_count": len(raw_requests),
        "credit/meter_unique_prefix_count": len(unique_requests),
        "credit/meter_deduplication_ratio": (
            1.0 - len(unique_requests) / len(raw_requests)
            if raw_requests
            else 0.0
        ),
        "credit/estimated_success_rate": (
            estimated_successes / valid_estimate_count
            if valid_estimate_count
            else 0.0
        ),
        "credit/estimated_failure_cost_rate": (
            estimated_failures_at_cost / valid_estimate_count
            if valid_estimate_count
            else 0.0
        ),
        "credit/uniform_fallback_rate": (
            uniform_fallbacks / len(trajectories) if trajectories else 0.0
        ),
        "credit/meter_error_fallback_rate": (
            meter_error_fallbacks / len(trajectories) if trajectories else 0.0
        ),
        "credit/token_multiplier_clip_rate": (
            sum(clip_rates) / len(clip_rates) if clip_rates else 0.0
        ),
    }
    metrics.update(
        {
            f"credit/{key}": value
            for key, value in getattr(meter_client, "last_metrics", {}).items()
        }
    )
    return data, metrics
