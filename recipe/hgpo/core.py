"""Core HGPO advantage computation from the official verl-agent recipe.

HGPO clusters every environment step by its current observation and by
increasingly long suffixes of its observation history.  It then combines the
relative advantages from all non-degenerate hierarchy levels using the
length-based weighting rule proposed in arXiv:2602.22817.
Text observations can optionally use the same SequenceMatcher metric as GiGPO;
every corresponding observation in a history suffix must meet the threshold.
"""

from collections import defaultdict
from difflib import SequenceMatcher

import numpy as np
import torch

from verl import DataProto


def to_hashable(value):
    """Recursively convert an observation into an exact-match hash key."""
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return tuple(value.flatten())
    if isinstance(value, (list, tuple)):
        return tuple(to_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, to_hashable(item)) for key, item in value.items()))
    raise TypeError(f"Unsupported observation type: {type(value)}")


def compute_step_discounted_returns(batch: DataProto, gamma: float) -> torch.Tensor:
    """Compute the return from every step to the end of its trajectory."""
    rewards = batch.non_tensor_batch["rewards"].astype(np.float32)
    trajectory_ids = batch.non_tensor_batch["traj_uid"]
    active_masks = batch.non_tensor_batch["active_masks"].astype(np.float32)
    all_returns = np.zeros_like(rewards)

    for trajectory_id in np.unique(trajectory_ids):
        indices = np.flatnonzero(trajectory_ids == trajectory_id)
        if not active_masks[indices].all():
            raise ValueError("HGPO expects only active steps in each gathered trajectory")
        running_return = 0.0
        for index in reversed(indices):
            running_return = float(rewards[index]) + gamma * running_return
            all_returns[index] = running_return

    return torch.as_tensor(
        all_returns,
        dtype=torch.float32,
        device=batch.batch["input_ids"].device,
    )


def _trajectory_advantages(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    group_ids: np.ndarray,
    trajectory_ids: np.ndarray,
    epsilon: float,
    normalize_by_std: bool,
) -> torch.Tensor:
    """Official optional episode-level base group."""
    scores = token_level_rewards.sum(dim=-1)
    grouped_scores: dict[object, list[torch.Tensor]] = defaultdict(list)
    seen = set()
    for row, (group_id, trajectory_id) in enumerate(zip(group_ids, trajectory_ids)):
        key = (group_id, trajectory_id)
        if key not in seen:
            grouped_scores[group_id].append(scores[row])
            seen.add(key)

    normalized = torch.zeros_like(scores)
    for group_id, values in grouped_scores.items():
        values_tensor = torch.stack(values)
        mean = values_tensor.mean()
        std = values_tensor.std(unbiased=False)
        rows = np.flatnonzero(group_ids == group_id)
        centered = scores[rows] - mean
        if normalize_by_std:
            centered = centered / (std + epsilon)
        normalized[rows] = centered
    return normalized.unsqueeze(-1).expand_as(response_mask) * response_mask


def compute_hgpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    anchor_obs: np.ndarray,
    index: np.ndarray,
    traj_index: np.ndarray,
    history_length: int,
    epsilon: float = 1e-6,
    mode: str = "mean_std_norm",
    length_weight_alpha: float = 1.0,
    base_group: bool = False,
    enable_similarity: bool = False,
    similarity_thresh: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute HGPO advantages with exact or text-similar history groups."""
    if history_length < 0:
        raise ValueError("history_length must be non-negative")
    if mode not in {"mean_norm", "mean_std_norm"}:
        raise ValueError(f"Unsupported HGPO normalization mode: {mode}")
    if enable_similarity:
        if not 0.0 < similarity_thresh <= 1.0:
            raise ValueError("HGPO similarity_thresh must be in (0, 1]")
        if any(not isinstance(observation, str) for observation in anchor_obs):
            raise ValueError("Similarity-based HGPO requires text observations")

    device = response_mask.device
    response_length = response_mask.shape[-1]
    rewards = step_rewards.detach().to(device=device, dtype=torch.float32)
    scalar_advantages = torch.zeros(len(anchor_obs), device=device, dtype=torch.float32)

    base_scalars = None
    if base_group:
        base = _trajectory_advantages(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            group_ids=index,
            trajectory_ids=traj_index,
            epsilon=epsilon,
            normalize_by_std=(mode == "mean_std_norm"),
        )
        base_scalars = (base * response_mask).sum(dim=1) / response_mask.sum(dim=1).clamp(min=epsilon)

    def aggregate(items: list[tuple[int, torch.Tensor]]) -> torch.Tensor:
        # The official recipe omits hierarchy levels with zero advantage.
        nonzero = [(level, value) for level, value in items if bool(value != 0)]
        if not nonzero:
            return torch.zeros((), device=device, dtype=torch.float32)
        levels = torch.tensor([level for level, _ in nonzero], device=device, dtype=torch.float32)
        values = torch.stack([value.float() for _, value in nonzero])
        weights = (levels + 1.0).pow(length_weight_alpha)
        weights = weights / (weights.sum() + epsilon)
        return (weights * values).sum()

    for group_id in np.unique(index):
        group_indices = np.flatnonzero(index == group_id)
        group_observations = anchor_obs[group_indices]
        group_trajectory_ids = traj_index[group_indices]
        group_rewards = rewards.index_select(
            0, torch.as_tensor(group_indices, device=device, dtype=torch.long)
        )

        unique_trajectories, inverse = np.unique(group_trajectory_ids, return_inverse=True)
        trajectory_positions = [
            np.flatnonzero(inverse == trajectory_number)
            for trajectory_number in range(len(unique_trajectories))
        ]
        trajectory_observations = [group_observations[positions] for positions in trajectory_positions]
        trajectory_rewards = [
            group_rewards.index_select(
                0, torch.as_tensor(positions, device=device, dtype=torch.long)
            )
            for positions in trajectory_positions
        ]
        trajectory_global_indices = [group_indices[positions] for positions in trajectory_positions]

        clusters: dict[tuple[int, tuple], list[tuple[int, int]]] = defaultdict(list)
        representatives: dict[int, list[tuple]] = defaultdict(list)
        matched_suffixes: dict[tuple[int, tuple], tuple[int, tuple]] = {}
        hashed_trajectories = [
            [to_hashable(observation) for observation in observations]
            for observations in trajectory_observations
        ]
        # A level includes the current state. Thus history_length=2 creates
        # current-only, one-history, and two-history groups.
        max_suffix_length = history_length + 1
        for trajectory_number, observations in enumerate(hashed_trajectories):
            for step_number in range(len(observations)):
                upper = min(max_suffix_length, step_number + 1)
                for suffix_length in range(1, upper + 1):
                    suffix = tuple(observations[step_number - suffix_length + 1 : step_number + 1])
                    key = (suffix_length, suffix)
                    if enable_similarity:
                        if key not in matched_suffixes:
                            # Like GiGPO, use the first matching representative.
                            # Compare each history position separately so a long
                            # shared observation cannot hide a different state.
                            representative = next(
                                (candidate for candidate in representatives[suffix_length]
                                 if all(SequenceMatcher(None, current, previous).ratio() >= similarity_thresh
                                        for current, previous in zip(suffix, candidate))),
                                None,
                            )
                            if representative is None:
                                representative = suffix
                                representatives[suffix_length].append(suffix)
                            matched_suffixes[key] = (suffix_length, representative)
                        key = matched_suffixes[key]
                    clusters[key].append((trajectory_number, step_number))

        per_step_items: dict[tuple[int, int], list[tuple[int, torch.Tensor]]] = defaultdict(list)
        for (suffix_length, _), members in clusters.items():
            if len(members) <= 1:
                continue
            values = torch.stack(
                [trajectory_rewards[trajectory][step] for trajectory, step in members]
            )
            centered = values - values.mean()
            if mode == "mean_std_norm":
                centered = centered / (values.std(unbiased=False) + epsilon)
            for member_index, member in enumerate(members):
                per_step_items[member].append((suffix_length, centered[member_index]))

        for trajectory_number, observations in enumerate(trajectory_observations):
            for step_number in range(len(observations)):
                global_index = int(trajectory_global_indices[trajectory_number][step_number])
                items = list(per_step_items.get((trajectory_number, step_number), []))
                if base_scalars is not None:
                    items.insert(0, (0, base_scalars[global_index]))
                scalar_advantages[global_index] = aggregate(items)

    scores = scalar_advantages.unsqueeze(-1).expand(-1, response_length) * response_mask
    return scores, scores
