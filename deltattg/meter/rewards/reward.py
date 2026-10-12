"""Custom verl reward for point continuation action-token estimate."""

from __future__ import annotations

from typing import Any

from deltattg.meter.schema import parse_strict_answer


def token_accuracy(estimated: int, target: int) -> float:
    """Bounded symmetric relative accuracy in [0, 1]."""
    scale = max(abs(estimated), abs(target), 1)
    return float(max(0.0, 1.0 - abs(estimated - target) / scale))


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float]:
    del data_source, extra_info
    estimate = parse_strict_answer(solution_str.strip())
    target_can_complete = bool(ground_truth["can_complete"])
    target_tokens = int(ground_truth["remaining_action_tokens"]) if target_can_complete else -1

    def result(
        *,
        score: float,
        format_reward: float,
        can_complete_reward: float,
        accuracy: float,
        estimated_can_complete: float,
        estimated_tokens: int = -1,
        absolute_error: int = -1,
    ) -> dict[str, float]:
        return {
            "score": score,
            "format_reward": format_reward,
            "can_complete_reward": can_complete_reward,
            "token_accuracy": accuracy,
            "target_can_complete": float(target_can_complete),
            "estimated_can_complete": estimated_can_complete,
            "estimated_tokens": float(estimated_tokens),
            "target_tokens": float(target_tokens),
            "token_absolute_error": float(absolute_error),
        }

    if estimate is None:
        return result(
            score=-1.0,
            format_reward=0.0,
            can_complete_reward=0.0,
            accuracy=0.0,
            estimated_can_complete=-1.0,
        )
    if estimate.can_complete != target_can_complete:
        return result(
            score=-1.0,
            format_reward=1.0,
            can_complete_reward=0.0,
            accuracy=0.0,
            estimated_can_complete=float(estimate.can_complete),
            estimated_tokens=(
                estimate.remaining_action_tokens
                if estimate.remaining_action_tokens is not None
                else -1
            ),
        )
    if not target_can_complete:
        return result(
            score=1.0,
            format_reward=1.0,
            can_complete_reward=1.0,
            accuracy=1.0,
            estimated_can_complete=0.0,
        )
    assert estimate.remaining_action_tokens is not None
    accuracy = token_accuracy(estimate.remaining_action_tokens, target_tokens)
    return result(
        score=0.2 + 0.8 * accuracy,
        format_reward=1.0,
        can_complete_reward=1.0,
        accuracy=accuracy,
        estimated_can_complete=1.0,
        estimated_tokens=estimate.remaining_action_tokens,
        absolute_error=abs(estimate.remaining_action_tokens - target_tokens),
    )
