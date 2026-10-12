"""Utilities for filtering zero-signal groups in offline GRPO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroupFilterResult:
    selected_indices: np.ndarray
    selected_group_ids: tuple[str, ...]
    effective_group_ratio: float
    zero_signal_group_ratio: float
    unique_estimates_per_group: float


def select_effective_grpo_groups(
    *,
    group_ids: np.ndarray,
    sequence_scores: np.ndarray,
    responses: np.ndarray,
    min_reward_std: float,
    excluded_group_ids: set[str] | None = None,
) -> GroupFilterResult:
    """Select complete GRPO groups whose sampled rewards have nonzero spread.

    ``responses`` is used only to report how much discrete-output exploration
    occurred. Selection itself is based on reward variance because distinct
    strings can still have identical rewards and therefore zero GRPO advantage.
    """

    group_ids = np.asarray(group_ids, dtype=object)
    sequence_scores = np.asarray(sequence_scores, dtype=np.float64)
    responses = np.asarray(responses)
    if not (len(group_ids) == len(sequence_scores) == len(responses)):
        raise ValueError("group_ids, sequence_scores, and responses must align")

    excluded = excluded_group_ids or set()
    ordered_ids = tuple(dict.fromkeys(str(value) for value in group_ids))
    selected_ids: list[str] = []
    selected_indices: list[int] = []
    unique_counts: list[int] = []
    eligible_count = 0

    for group_id in ordered_ids:
        if group_id in excluded:
            continue
        eligible_count += 1
        indices = np.flatnonzero(group_ids == group_id)
        group_scores = sequence_scores[indices]
        unique_counts.append(len({tuple(np.asarray(responses[i]).tolist()) for i in indices}))
        if len(indices) > 1 and float(np.std(group_scores)) > min_reward_std:
            selected_ids.append(group_id)
            selected_indices.extend(indices.tolist())

    effective_ratio = len(selected_ids) / eligible_count if eligible_count else 0.0
    return GroupFilterResult(
        selected_indices=np.asarray(selected_indices, dtype=np.int64),
        selected_group_ids=tuple(selected_ids),
        effective_group_ratio=effective_ratio,
        zero_signal_group_ratio=1.0 - effective_ratio if eligible_count else 1.0,
        unique_estimates_per_group=float(np.mean(unique_counts)) if unique_counts else 0.0,
    )
