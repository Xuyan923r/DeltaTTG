# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dynamic sampling used by the multi-turn DAPO baseline."""

from __future__ import annotations

from typing import Any

import numpy as np


def filter_group_data(
    batch_list: list[list[dict[str, Any]]],
    episode_rewards: np.ndarray,
    episode_lengths: np.ndarray,
    success: dict[str, np.ndarray],
    traj_uid: np.ndarray,
    tool_callings: np.ndarray,
    config,
    last_try: bool = False,
):
    """Filter low-variance groups; retain the final unfiltered refill attempt."""

    if last_try:
        return (
            batch_list,
            episode_rewards,
            episode_lengths,
            success,
            traj_uid,
            tool_callings,
        )
    batch_size = config.data.train_batch_size
    group_n = config.env.rollout.n
    min_reward_std = float(config.algorithm.filter_groups.min_reward_std)
    if not np.isfinite(min_reward_std) or min_reward_std < 0:
        raise ValueError("min_reward_std must be finite and nonnegative")
    if group_n <= 1:
        print("Warning: group_n <= 1, no need to adopt dynamic sampling")

    keep_indices = np.array([], dtype=np.int64)
    for group_index in range(batch_size):
        indices = np.arange(group_index * group_n, (group_index + 1) * group_n)
        rewards = episode_rewards[indices]
        for index in indices:
            assert batch_list[index][0]["uid"] == batch_list[indices[0]][0]["uid"]
        if float(np.std(rewards, dtype=np.float64)) > min_reward_std:
            keep_indices = np.concatenate((keep_indices, indices))

    success = {
        key: value[keep_indices]
        for key, value in success.items()
        if len(value) == len(batch_list)
    }
    return (
        [batch_list[index] for index in keep_indices],
        episode_rewards[keep_indices],
        episode_lengths[keep_indices],
        success,
        traj_uid[keep_indices],
        tool_callings[keep_indices],
    )
