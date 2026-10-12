"""Map meter outputs and real terminal outcomes to scalar costs."""

from __future__ import annotations

import math

from deltattg.types import MeterResult


def value_to_cost(estimate: MeterResult) -> float:
    """Use -S(h) so cost reduction is S(next) - S(current)."""
    if not estimate.valid or estimate.value is None or not math.isfinite(estimate.value):
        raise ValueError("Value credit requires a finite meter value")
    return -float(estimate.value)


def estimate_to_cost(
    estimate: MeterResult,
    failure_cost: float,
    *,
    enable_cost_clipping: bool = False,
) -> float:
    if failure_cost <= 0:
        raise ValueError("failure_cost must be positive")
    if not estimate.valid:
        raise ValueError("Invalid meter result.")
    if not estimate.can_complete:
        return float(failure_cost)
    tokens = estimate.remaining_action_tokens
    if tokens is None or tokens < 0:
        raise ValueError("Invalid completable remaining-action-token estimate.")
    return float(min(tokens, failure_cost) if enable_cost_clipping else tokens)


def feasibility_to_cost(
    estimate: MeterResult,
    failure_cost: float,
) -> float:
    """Map only the meter's binary feasibility decision to a cost."""
    if failure_cost <= 0:
        raise ValueError("failure_cost must be positive")
    if not estimate.valid:
        raise ValueError("Invalid meter result.")
    return 0.0 if estimate.can_complete else float(failure_cost)


def terminal_cost(trajectory_success: bool, failure_cost: float) -> float:
    if failure_cost <= 0:
        raise ValueError("failure_cost must be positive")
    return 0.0 if trajectory_success else float(failure_cost)


def completion_log_odds_to_cost(log_odds: float, clip: float) -> float:
    """Convert higher-is-better completion confidence to lower-is-better cost."""
    if clip <= 0:
        raise ValueError("completion log-odds clip must be positive")
    return -max(-float(clip), min(float(clip), float(log_odds)))


def completion_terminal_cost(trajectory_success: bool, clip: float) -> float:
    """Anchor terminal success/failure at the two ends of the confidence scale."""
    if clip <= 0:
        raise ValueError("completion log-odds clip must be positive")
    return -float(clip) if trajectory_success else float(clip)
