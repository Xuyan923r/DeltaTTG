"""Weak-reference-model-equivalent progress computation."""

from __future__ import annotations

import torch


def compute_step_progress(
    nonterminal_costs: list[float],
    terminal_state_cost: float,
    *,
    progress_clip: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not nonterminal_costs:
        raise ValueError("Trajectory has no nonterminal decision states")
    state_costs = torch.tensor(
        [*nonterminal_costs, terminal_state_cost],
        dtype=torch.float32,
    )
    if not torch.isfinite(state_costs).all():
        raise ValueError("State costs must be finite")
    progress = state_costs[:-1] - state_costs[1:]
    if progress_clip is not None:
        if progress_clip <= 0:
            raise ValueError("progress_clip must be positive")
        progress = progress.clamp(-float(progress_clip), float(progress_clip))
    return state_costs, progress
