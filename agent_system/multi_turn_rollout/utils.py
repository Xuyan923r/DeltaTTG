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

import torch
import numpy as np
import math
from PIL import Image
from verl import DataProto

def to_list_of_dict(batch: DataProto) -> list[dict]:
    tensors = batch.batch
    non_tensor = batch.non_tensor_batch
    batch_size = len(tensors['input_ids'])
    save_list = []
    for bs in range(batch_size):
        save_dict = dict()
        for key, val in tensors.items():
            save_dict[key] = val[bs]
        for key, val in non_tensor.items():
            save_dict[key] = val[bs]
        save_list.append(save_dict)
    return save_list


def torch_to_numpy(tensor, is_object=False):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        pass
    else:
        raise ValueError(f"Unsupported type: {type(tensor)})")

    if is_object:
        tensor = tensor.astype(object)
    return tensor

def numpy_to_torch(array, device):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array).to(device)
    elif isinstance(array, torch.Tensor):
        array = array.to(device)
    else:
        raise ValueError(f"Unsupported type: {type(array)})")
    return array


def process_image(image, max_pixels: int = 2048 * 2048, min_pixels: int = 256 * 256):
    if isinstance(image, torch.Tensor):
        image = torch_to_numpy(image)
    if image.max() < 1:
        image = image * 255.0
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    image = Image.fromarray(image)

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


def adjust_batch(config, data: DataProto, mode="copy") -> DataProto:
    world_size = config.trainer.n_gpus_per_node * config.trainer.nnodes
    size_divisor_rollout = config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu * world_size
    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        size_divisor_ref = config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu * world_size
    else:
        size_divisor_ref = size_divisor_rollout
    if "multi_modal_inputs" in data.non_tensor_batch:
        size_divisor_actor = config.actor_rollout_ref.actor.ppo_mini_batch_size
    else:
        size_divisor_actor = config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu * world_size
    size_divisor = np.lcm.reduce(np.array([size_divisor_ref, size_divisor_rollout, size_divisor_actor])).item()

    # check if the batch size is divisible by the dp size, if not, delete the last few samples to make it divisible
    bs = len(data)
    remainder = bs % size_divisor
    if remainder == 0:
        return data

    # Agent trajectories span multiple independent rows. Never pad/delete an
    # arbitrary row: that corrupts traj_uid grouping for GRPO, GiGPO and
    # deltattg. Drop the smallest set of complete trajectories whose combined
    # row count has the required remainder. With the project 2-GPU settings
    # this normally removes one odd-length trajectory at most.
    if "traj_uid" in data.non_tensor_batch and mode != "copy_with_advantages":
        grouped: dict[str, list[int]] = {}
        for row_index, trajectory_id in enumerate(data.non_tensor_batch["traj_uid"]):
            grouped.setdefault(str(trajectory_id), []).append(row_index)
        best: dict[int, tuple[int, ...]] = {0: ()}
        group_rows = list(grouped.values())
        for group_index, rows in enumerate(group_rows):
            updates = dict(best)
            for residue, selected in best.items():
                candidate = selected + (group_index,)
                candidate_residue = (residue + len(rows)) % size_divisor
                if candidate_residue not in updates or sum(
                    len(group_rows[index]) for index in candidate
                ) < sum(len(group_rows[index]) for index in updates[candidate_residue]):
                    updates[candidate_residue] = candidate
            best = updates
        selected_groups = best.get(remainder)
        if selected_groups:
            remove = {
                row_index
                for group_index in selected_groups
                for row_index in group_rows[group_index]
            }
            if len(remove) < bs:
                keep = np.array(
                    [index for index in range(bs) if index not in remove],
                    dtype=np.int64,
                )
                return data.select_idxs(keep)
        raise ValueError(
            f"Cannot make a trajectory-complete batch of {bs} rows divisible by {size_divisor}"
        )
    
    if mode == "delete":
        # Generate indices to remove, rather than indices to keep
        remove_indices = np.random.choice(bs, remainder, replace=False)
        # Sort remove_indices to maintain stability when deleting
        remove_indices = np.sort(remove_indices)
        
        # Create a boolean mask for elements to keep
        keep_mask = np.ones(bs, dtype=bool)
        keep_mask[remove_indices] = False

        keep_mask_tensor = torch.tensor(keep_mask, dtype=torch.bool, device=data.batch['input_ids'].device)
        # Apply the mask to keep elements in their original order
        tensor_data = data.batch[keep_mask_tensor]
        non_tensor_data = {key: val[keep_mask] for key, val in data.non_tensor_batch.items()}
        adjusted_batch = DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=data.meta_info)
        del data
    elif mode in {"copy", "copy_with_advantages"}:
        to_add = size_divisor - remainder
        dup_indices = np.random.choice(bs, to_add, replace=False)
        dup_proto = data.select_idxs(dup_indices)

        adjusted_batch = DataProto.concat([data, dup_proto])
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return adjusted_batch
