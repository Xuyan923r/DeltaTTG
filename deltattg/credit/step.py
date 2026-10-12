"""Sign-aware trajectory-internal step credit."""

from __future__ import annotations

import torch


def build_credit_mass(
    step_progress: torch.Tensor,
    group_advantage: float,
    mode: str,
) -> torch.Tensor:
    if step_progress.ndim != 1:
        raise ValueError("step_progress must be one-dimensional")
    if mode == "uniform":
        return torch.ones_like(step_progress)
    if mode == "positive_progress":
        return torch.relu(step_progress)
    if mode == "sign_aware":
        if group_advantage > 0:
            return torch.relu(step_progress)
        if group_advantage < 0:
            return torch.relu(-step_progress)
        return torch.ones_like(step_progress)
    raise ValueError(f"Unsupported credit mode: {mode}")


def normalize_step_credit(
    credit_mass: torch.Tensor,
    uniform_credit_mix: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    if credit_mass.ndim != 1:
        raise ValueError("credit_mass must be one-dimensional.")
    num_steps = credit_mass.numel()
    if num_steps <= 0:
        raise ValueError("Trajectory has no agent steps.")
    if not 0.0 <= uniform_credit_mix <= 1.0:
        raise ValueError("uniform_credit_mix must be in [0, 1].")
    uniform = torch.full_like(credit_mass, 1.0 / num_steps)
    total = credit_mass.sum()
    if not torch.isfinite(total) or total <= eps:
        return uniform
    focused = credit_mass / total
    shares = uniform_credit_mix * uniform + (1.0 - uniform_credit_mix) * focused
    return shares / shares.sum().clamp_min(eps)
