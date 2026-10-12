"""Map normalized step shares to scale-preserving token multipliers."""

from __future__ import annotations

import torch


def build_token_credit_multipliers(
    step_credit_shares: torch.Tensor,
    step_token_positions: list[list[int]],
    response_length: int,
    trainable_mask: torch.Tensor,
    max_token_multiplier: float | None,
    eps: float = 1e-8,
    token_allocation_mode: str = "step_broadcast",
) -> tuple[torch.Tensor, float]:
    if token_allocation_mode not in {"step_broadcast", "step_share"}:
        raise ValueError(f"Unknown token_allocation_mode: {token_allocation_mode}")
    if step_credit_shares.numel() != len(step_token_positions):
        raise ValueError("Step shares and step-token mapping mismatch.")
    if trainable_mask.ndim != 1 or trainable_mask.numel() != response_length:
        raise ValueError("trainable_mask and response_length mismatch")
    if max_token_multiplier is not None and max_token_multiplier < 1:
        raise ValueError("max_token_multiplier must be at least 1 to preserve mean weight 1")

    mask = trainable_mask.bool()
    multipliers = torch.zeros(
        response_length,
        dtype=step_credit_shares.dtype,
        device=step_credit_shares.device,
    )
    total_trainable_tokens = int(mask.sum().item())
    if total_trainable_tokens <= 0:
        raise ValueError("Trajectory contains no trainable tokens.")

    covered: set[int] = set()
    for step_idx, positions in enumerate(step_token_positions):
        valid_positions = []
        for raw_position in positions:
            position = int(raw_position)
            if position < 0 or position >= response_length:
                raise ValueError(f"Step {step_idx} token position {position} is out of range")
            if bool(mask[position]):
                valid_positions.append(position)
        if not valid_positions:
            raise ValueError(f"Step {step_idx} has no trainable tokens.")
        overlap = covered.intersection(valid_positions)
        if overlap:
            raise ValueError(f"Trainable token positions overlap across steps: {sorted(overlap)[:5]}")
        covered.update(valid_positions)
        per_token = step_credit_shares[step_idx]
        if token_allocation_mode == "step_share":
            per_token = total_trainable_tokens * per_token / len(valid_positions)
        multipliers[valid_positions] = per_token

    expected = set(torch.nonzero(mask, as_tuple=False).flatten().tolist())
    if covered != expected:
        missing = sorted(expected - covered)
        raise ValueError(f"Step mapping does not cover all trainable tokens: {missing[:5]}")

    normalization = multipliers[mask].mean().clamp_min(eps)
    multipliers[mask] = multipliers[mask] / normalization
    clip_rate = 0.0
    if max_token_multiplier is not None:
        cap = float(max_token_multiplier)
        raw = multipliers[mask]
        clip_rate = float((raw > cap).float().mean().item())
        bounded = raw.clamp(min=0.0, max=cap)
        # Return mass removed by the cap to tokens with remaining capacity.
        # This keeps both the final mean at one and the final maximum at cap.
        deficit = bounded.numel() - bounded.sum()
        if deficit > 0:
            capacity = (cap - bounded).clamp_min(0.0)
            bounded = bounded + deficit * capacity / capacity.sum().clamp_min(eps)
        multipliers[mask] = bounded.clamp(max=cap)
    return multipliers, clip_rate


def uniform_token_multipliers(trainable_mask: torch.Tensor) -> torch.Tensor:
    return trainable_mask.to(dtype=torch.float32)
