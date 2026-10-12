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

import json
import os
from pathlib import Path
import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy
from recipe.dapo.dynamic_sampling import filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict, Optional
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor



    @staticmethod
    def _jsonable(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): TrajectoryCollector._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [TrajectoryCollector._jsonable(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _parsed_action(response: str) -> tuple[str, bool]:
        lowered = response.lower()
        start = lowered.find('<action>')
        end = lowered.find('</action>')
        tagged = start >= 0 and end > start
        action = lowered[start + len('<action>'):end].strip() if tagged else lowered[-30:]
        valid = tagged and '<think>' in response and '</think>' in response
        return action, valid

    def _trajectory_records(
        self,
        *,
        total_batch_list,
        total_infos,
        episode_rewards,
        episode_lengths,
        traj_uid,
        envs,
        trial: int,
        global_indices: list[int],
        memory_before: list[list[str]],
    ) -> list[dict]:
        records = []
        for local_index, steps in enumerate(total_batch_list):
            active_steps = [step for step in steps if bool(step['active_masks'])]
            step_records = []
            for step_index, step in enumerate(active_steps):
                response_ids = step['responses']
                response_mask = response_ids.ne(self.tokenizer.pad_token_id)
                response = self.tokenizer.decode(
                    response_ids[response_mask], skip_special_tokens=True
                )
                action, format_valid = self._parsed_action(response)
                prompt_ids = step['prompts']
                prompt_mask = prompt_ids.ne(self.tokenizer.pad_token_id)
                rendered_prompt = self.tokenizer.decode(
                    prompt_ids[prompt_mask], skip_special_tokens=False
                )
                info = total_infos[local_index][step_index]
                step_records.append(
                    {
                        'step_index': step_index,
                        'actor_prompt_messages': self._jsonable(step.get('raw_prompt')),
                        'actor_rendered_prompt': rendered_prompt,
                        'actor_prompt_token_count': int(prompt_mask.sum().item()),
                        'actor_response': response,
                        'actor_response_token_count': int(response_mask.sum().item()),
                        'parsed_action': action,
                        'format_valid': bool(format_valid),
                        'action_valid': bool(step.get('is_action_valid', True)),
                        'environment_observation': str(step.get('environment_observation', '')),
                        'admissible_actions': self._jsonable(
                            step.get('environment_admissible_actions', [])
                        ),
                        'environment_reward': float(step.get('rewards', 0.0)),
                        'environment_info_after': self._jsonable(info),
                    }
                )
            game_file = str(envs.gamefile[local_index])
            records.append(
                {
                    'schema_version': 'alfworld_verl_agent_exhaustive_trajectory_v1',
                    'model_label': str(
                        self.config.env.alfworld.get('eval_model_label') or ''
                    ),
                    'method': (
                        'reflexion'
                        if bool(self.config.env.alfworld.reflexion.get('enabled', False))
                        else 'react'
                    ),
                    'split': str(self.config.env.alfworld.get('eval_split') or ''),
                    'seed': int(self.config.env.seed),
                    'global_task_index': int(global_indices[local_index]),
                    'trial': int(trial),
                    'game_file': os.path.realpath(game_file),
                    'task': str(envs.tasks[local_index]),
                    'trajectory_id': str(traj_uid[local_index]),
                    'success': bool(float(episode_rewards[local_index]) > 0.0),
                    'reward': float(episode_rewards[local_index]),
                    'num_steps': int(episode_lengths[local_index]),
                    'memory_before': list(memory_before[local_index]),
                    'reflection_after': None,
                    'steps': step_records,
                }
            )
        return records

    def _write_eval_records(self, records: list[dict]) -> None:
        path_value = self.config.env.alfworld.get('eval_trajectory_path')
        if not path_value:
            return
        path = Path(str(path_value)).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        expected_path = self.config.env.alfworld.get('fixed_game_files_path')
        if expected_path:
            payload = json.loads(Path(str(expected_path)).read_text(encoding='utf-8'))
            expected = payload.get('game_files') if isinstance(payload, dict) else payload
            expected = [os.path.realpath(str(item)) for item in expected]
            trial_zero = [row['game_file'] for row in records if int(row['trial']) == 0]
            if trial_zero != expected:
                raise RuntimeError(
                    'Exhaustive ALFWorld coverage check failed: trial-0 game order '
                    'does not exactly match the fixed manifest'
                )
        temporary = path.with_suffix(path.suffix + '.tmp')
        with temporary.open('w', encoding='utf-8') as destination:
            for record in records:
                destination.write(json.dumps(record, ensure_ascii=False) + '\n')
        temporary.replace(path)
        print(f'ALFWORLD_EXHAUSTIVE_TRAJECTORIES\t{path}\trecords={len(records)}', flush=True)



    def _search_trajectory_records(
        self,
        *,
        total_batch_list,
        total_infos,
        episode_rewards,
        episode_lengths,
        traj_uid,
        envs,
        global_indices: list[int],
        source_prompts,
        source_extra_infos,
        source_env_kwargs,
    ) -> list[dict]:
        """Export native Search transitions without reimplementing the agent loop."""
        records = []
        for local_index, steps in enumerate(total_batch_list):
            active_steps = [step for step in steps if bool(step['active_masks'])]
            step_records = []
            data_source = ''
            for step_index, step in enumerate(active_steps):
                response_ids = step['responses']
                response_mask = response_ids.ne(self.tokenizer.pad_token_id)
                valid_response_ids = response_ids[response_mask]
                raw_response = self.tokenizer.decode(
                    valid_response_ids, skip_special_tokens=True
                )
                response_with_special_tokens = self.tokenizer.decode(
                    valid_response_ids, skip_special_tokens=False
                )
                prompt_ids = step['prompts']
                prompt_mask = prompt_ids.ne(self.tokenizer.pad_token_id)
                rendered_prompt = self.tokenizer.decode(
                    prompt_ids[prompt_mask], skip_special_tokens=False
                )
                info = total_infos[local_index][step_index]
                data_source = str(info.get('data_source', data_source))
                projected_action = str(
                    info.get('projected_action')
                    or info.get('postprocessed_action')
                    or ''
                )
                step_records.append(
                    {
                        'step_index': step_index,
                        'actor_prompt_messages': self._jsonable(
                            step.get('raw_prompt')
                        ),
                        'actor_rendered_prompt': rendered_prompt,
                        'actor_prompt_token_count': int(prompt_mask.sum().item()),
                        'actor_response': raw_response,
                        'actor_response_with_special_tokens': (
                            response_with_special_tokens
                        ),
                        'actor_response_token_ids': self._jsonable(
                            valid_response_ids
                        ),
                        'actor_response_token_count': int(
                            response_mask.sum().item()
                        ),
                        'projected_action': projected_action,
                        'action_valid': bool(step.get('is_action_valid', True)),
                        # This is the full Search prompt visible immediately
                        # before the action, before the tokenizer chat wrapper.
                        'environment_observation_before': str(
                            step.get('environment_observation', '')
                        ),
                        # This is the raw <information> payload returned by the
                        # retriever after the projected action.
                        'environment_observation_after': str(
                            info.get('environment_observation_after', '')
                        ),
                        'environment_reward': float(step.get('rewards', 0.0)),
                        'environment_done': bool(info.get('environment_done', False)),
                        'environment_info_after': self._jsonable(info),
                    }
                )
            records.append(
                {
                    'schema_version': 'papersearchqa_verl_agent_raw_trajectory_v1',
                    'model_label': str(
                        self.config.env.search.get('eval_model_label') or ''
                    ),
                    'model_path': str(self.config.actor_rollout_ref.model.path),
                    'seed': int(self.config.env.seed),
                    'global_task_index': int(global_indices[local_index]),
                    'data_source': data_source,
                    'task': str(envs.tasks[local_index]),
                    # Keep the source row self-contained for later SFT/RL data
                    # construction.  These fields are labels/metadata and are
                    # never inserted into the actor prompt above.
                    'source_prompt_messages': self._jsonable(
                        source_prompts[local_index]
                    ),
                    'source_extra_info': self._jsonable(
                        source_extra_infos[local_index]
                    ),
                    'ground_truth': self._jsonable(
                        source_env_kwargs[local_index].get('ground_truth')
                    ),
                    'trajectory_id': str(traj_uid[local_index]),
                    'success': bool(float(episode_rewards[local_index]) >= 1.0),
                    'reward': float(episode_rewards[local_index]),
                    'num_steps': int(episode_lengths[local_index]),
                    'rollout_settings': {
                        'max_steps': int(self.config.env.max_steps),
                        'history_length': int(self.config.env.history_length),
                        'max_prompt_length': int(self.config.data.max_prompt_length),
                        'max_response_length': int(self.config.data.max_response_length),
                        'max_action_tokens': int(self.config.actor_rollout_ref.rollout.response_length),
                        'temperature': float(
                            self.config.actor_rollout_ref.rollout.val_kwargs.temperature
                        ),
                        'do_sample': bool(
                            self.config.actor_rollout_ref.rollout.val_kwargs.do_sample
                        ),
                        'top_p': float(self.config.actor_rollout_ref.rollout.top_p),
                        'enable_thinking': bool(
                            self.config.data.apply_chat_template_kwargs.get(
                                'enable_thinking', False
                            )
                        ),
                        'prompt_mode': str(
                            self.config.env.search.get('prompt_mode', 'thinking')
                        ),
                        'search_topk': int(self.config.env.search.topk),
                    },
                    'steps': step_records,
                }
            )
        return records

    def _write_search_eval_records(self, records: list[dict]) -> None:
        """Append complete validation batches with stable task-level deduplication."""
        path_value = self.config.env.search.get('eval_trajectory_path')
        if not path_value:
            return
        path = Path(str(path_value)).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not hasattr(self, '_search_eval_seen_keys'):
            self._search_eval_seen_keys = set()
            if path.is_file():
                with path.open(encoding='utf-8') as source:
                    for line_number, line in enumerate(source, 1):
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise RuntimeError(
                                f'Corrupt Search trajectory JSONL at {path}:{line_number}'
                            ) from exc
                        self._search_eval_seen_keys.add(
                            (
                                str(row.get('model_label', '')),
                                int(row['seed']),
                                int(row['global_task_index']),
                            )
                        )
        new_records = []
        for record in records:
            key = (
                str(record.get('model_label', '')),
                int(record['seed']),
                int(record['global_task_index']),
            )
            if key in self._search_eval_seen_keys:
                continue
            self._search_eval_seen_keys.add(key)
            new_records.append(record)
        if new_records:
            with path.open('a', encoding='utf-8') as destination:
                for record in new_records:
                    destination.write(json.dumps(record, ensure_ascii=False) + '\n')
                destination.flush()
                os.fsync(destination.fileno())
        print(
            f'PAPERSEARCHQA_NATIVE_TRAJECTORIES\t{path}\t'
            f'new={len(new_records)}\ttotal={len(self._search_eval_seen_keys)}',
            flush=True,
        )

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        
        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        
        # Apply chat template
        if obs.get('chat') is not None:
            chat = obs['chat'][item]
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        

        if is_multi_modal:

            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist() if isinstance(chat, np.ndarray) else list(chat)
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            sampling_metrics: Optional[Dict[str, float]] = None,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    data['outcome_reward'] = float(episode_rewards[bs])
                    data['trajectory_success'] = bool(episode_rewards[bs] > 0.0)
                    data['termination_reason'] = (
                        'success'
                        if episode_rewards[bs] > 0.0
                        else 'environment_end_or_horizon'
                    )
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value
                    for key, value in (sampling_metrics or {}).items():
                        data[key] = value

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            reset_kwargs=None,
            evaluation_trial: int = 0,
            evaluation_global_indices: Optional[List[int]] = None,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        batch_size = len(gen_batch.batch)

        # Initial observations from the environment
        if reset_kwargs is None:
            reset_kwargs = gen_batch.non_tensor_batch.get('env_kwargs')
        obs, infos = envs.reset(kwargs=reset_kwargs)

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        if evaluation_global_indices is None:
            evaluation_global_indices = list(range(batch_size))
        if len(evaluation_global_indices) != batch_size:
            raise ValueError('Evaluation global index count must match rollout batch size')
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            # pad to be divisible by dp_size
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            # # unpad
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid
            # deltattg consumes these native step-independent rows. Preserve the
            # observation visible before each action so the meter sees the
            # actual state. Observation tokens are context only and are never
            # counted in the remaining-action-token target.
            batch.non_tensor_batch['environment_task'] = np.array(
                list(envs.tasks), dtype=object
            )
            text_observations = obs.get('deltattg_meter_text', obs.get('text'))
            if text_observations is None:
                text_observations = [''] * batch_size
            batch.non_tensor_batch['environment_observation'] = np.array(
                [str(value) for value in text_observations], dtype=object
            )
            batch.non_tensor_batch['environment_step_index'] = np.full(
                batch_size, _step, dtype=np.int64
            )
            prefix_adapters = obs.get('deltattg_prefix_adapter')
            if prefix_adapters is None:
                prefix_adapters = ['default'] * batch_size
            batch.non_tensor_batch['environment_prefix_adapter'] = np.array(
                [str(value) for value in prefix_adapters], dtype=object
            )
            agent_systems = obs.get('deltattg_agent_system')
            if agent_systems is not None:
                batch.non_tensor_batch['environment_agent_system'] = np.array(
                    [str(value) for value in agent_systems], dtype=object
                )
            if hasattr(envs.envs, 'get_admissible_commands'):
                batch.non_tensor_batch['environment_admissible_actions'] = np.array(
                    [list(commands) for commands in envs.envs.get_admissible_commands],
                    dtype=object,
                )
            batch.non_tensor_batch['environment_evaluation_trial'] = np.full(
                batch_size, int(evaluation_trial), dtype=np.int64
            )
            batch.non_tensor_batch['environment_global_task_index'] = np.array(
                evaluation_global_indices, dtype=np.int64
            )

            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            
            next_obs, rewards, dones, infos = envs.step(text_actions)

            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            # episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        self._last_total_infos = total_infos
        self._last_rollout_records = []
        if (
            'alfworld' in str(self.config.env.env_name).lower()
            and self.config.env.alfworld.get('fixed_game_files_path')
        ):
            self._last_rollout_records = self._trajectory_records(
                total_batch_list=total_batch_list,
                total_infos=total_infos,
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                traj_uid=traj_uid,
                envs=envs,
                trial=evaluation_trial,
                global_indices=evaluation_global_indices,
                memory_before=(reset_kwargs or {}).get(
                    'reflexion_memory', [[] for _ in range(batch_size)]
                ),
            )
        if (
            'search' in str(self.config.env.env_name).lower()
            and self.config.env.search.get('eval_trajectory_path')
        ):
            self._last_rollout_records = self._search_trajectory_records(
                total_batch_list=total_batch_list,
                total_infos=total_infos,
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                traj_uid=traj_uid,
                envs=envs,
                global_indices=evaluation_global_indices,
                source_prompts=gen_batch.non_tensor_batch['raw_prompt'],
                source_extra_infos=gen_batch.non_tensor_batch['extra_info'],
                source_env_kwargs=gen_batch.non_tensor_batch['env_kwargs'],
            )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        raw_success = None
        generated_trajectories = 0
        first_attempt_generated = 0
        first_attempt_accepted = 0
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            rollout_fn = self.vanilla_multi_turn_loop
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = rollout_fn(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            generated_trajectories += len(batch_list)
            if raw_success is None:
                # Keep the first, unfiltered rollout statistics comparable with
                # vanilla GRPO/deltattg.  Later attempts are conditional resamples.
                raw_success = {key: np.asarray(value).copy() for key, value in success.items()}
                first_attempt_generated = len(batch_list)
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list, 
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            if try_count == 1:
                first_attempt_accepted = len(batch_list)
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        # Preserve the familiar metric name for the unconditional first rollout,
        # and expose the conditional, post-filter population under a new name.
        metric_success = {
            f"filtered_{key}": value
            for key, value in total_success.items()
        }
        for key, value in (raw_success or {}).items():
            metric_success[key] = np.full(
                len(total_batch_list),
                float(np.mean(value)),
                dtype=np.float32,
            )

        sampling_metrics = {
            "dynamic_sampling_attempts": float(try_count),
            "dynamic_sampling_initial_keep_ratio": (
                float(first_attempt_accepted / first_attempt_generated)
                if first_attempt_generated else 0.0
            ),
            "dynamic_sampling_overall_keep_ratio": (
                float(len(total_batch_list) / generated_trajectories)
                if generated_trajectories else 0.0
            ),
            "dynamic_sampling_generated_trajectories": float(generated_trajectories),
            "dynamic_sampling_accepted_trajectories": float(len(total_batch_list)),
        }

        return (
            total_batch_list,
            total_episode_rewards,
            total_episode_lengths,
            metric_success,
            total_traj_uid,
            total_tool_callings,
            sampling_metrics,
        )

    def _build_reflection_prompt(self, record: dict, memory: list[str]) -> str:
        lines = [f"Task: {record['task']}"]
        for step in record['steps']:
            info = step.get('environment_info_after') or {}
            observation_after = info.get('observation_text', '') if isinstance(info, dict) else ''
            lines.extend(
                [
                    f"Turn {int(step['step_index']) + 1}",
                    f"Observation: {step['environment_observation']}",
                    f"Assistant response: {step['actor_response']}",
                    f"Executed action: {step['parsed_action']}",
                    f"Result: {observation_after}",
                ]
            )
        trace = '\n'.join(lines)
        max_chars = int(
            self.config.env.alfworld.reflexion.get(
                'reflection_max_input_chars', 6000
            )
        )
        if len(trace) > max_chars:
            header = lines[0] + '\n[Earlier turns truncated]\n'
            trace = header + trace[-max(0, max_chars - len(header)):]
        previous = (
            '\n\nPlans from earlier failed attempts:\n'
            + '\n'.join(f'- {plan.strip()}' for plan in memory if plan.strip())
            if memory
            else ''
        )
        return (
            'You are reflecting on a failed ALFWorld attempt. Diagnose the strategy '
            'and path, not merely the final action. Produce a concise new plan that '
            'identifies the mistake and names concrete admissible actions or search '
            'choices that should be tried on the next reset. Do not claim success. '
            'Return only `Plan: ...`.\n\n'
            f'Failed experience:\n{trace}{previous}\n\nNew plan:'
        )

    def _generate_reflections(self, prompts: list[str], actor_rollout_wg) -> list[str]:
        if not prompts:
            return []
        rows = []
        apply_chat_template_kwargs = self.config.data.get(
            'apply_chat_template_kwargs', {}
        )
        for prompt in prompts:
            rendered = self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}],
                add_generation_prompt=True,
                tokenize=False,
                **apply_chat_template_kwargs,
            )
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=rendered,
                tokenizer=self.tokenizer,
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.config.data.truncation,
            )
            rows.append(
                {
                    'input_ids': input_ids[0],
                    'attention_mask': attention_mask[0],
                    'position_ids': compute_position_id_with_mask(attention_mask)[0],
                    'raw_prompt_ids': self.tokenizer.encode(
                        rendered, add_special_tokens=False
                    ),
                }
            )
        request = DataProto.from_single_dict(data=collate_fn(rows))
        request.meta_info = {
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'recompute_log_prob': False,
            'do_sample': True,
            'validate': True,
        }
        padded, pad_size = pad_dataproto_to_divisor(
            request, actor_rollout_wg.world_size
        )
        generated = actor_rollout_wg.generate_sequences(padded)
        generated = unpad_dataproto(generated, pad_size=pad_size)
        reflections = self.tokenizer.batch_decode(
            generated.batch['responses'], skip_special_tokens=True
        )
        if len(reflections) != len(prompts):
            raise RuntimeError('Reflection output count does not match failed tasks')
        return [text.strip() or 'Plan: Try a different admissible search path.' for text in reflections]

    @staticmethod
    def _alfworld_task_type(game_file: str) -> str:
        for name in (
            'pick_and_place',
            'pick_two_obj_and_place',
            'look_at_obj_in_light',
            'pick_heat_then_place_in_recep',
            'pick_cool_then_place_in_recep',
            'pick_clean_then_place_in_recep',
        ):
            if name in game_file:
                return name
        return 'unknown'

    def reflexion_multi_turn_loop(self, gen_batch, actor_rollout_wg, envs):
        """Run Reflexion inside the native trainer/agent-loop/Ray path."""
        task_count = len(gen_batch.batch)
        trial_count = int(self.config.env.alfworld.reflexion.get('num_trials', 10))
        memory_size = int(self.config.env.alfworld.reflexion.get('memory_size', 3))
        memories: list[list[str]] = [[] for _ in range(task_count)]
        solved = np.zeros(task_count, dtype=bool)
        selected = [None] * task_count
        all_records: list[dict] = []

        for trial in range(trial_count):
            active_indices = [index for index in range(task_count) if not solved[index]]
            if not active_indices:
                break
            trial_batch = gen_batch.select_idxs(active_indices)
            trial_memory = [memories[index] for index in active_indices]
            result = self.vanilla_multi_turn_loop(
                gen_batch=trial_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                reset_kwargs={
                    'active_indices': active_indices,
                    'reflexion_memory': trial_memory,
                },
                evaluation_trial=trial,
                evaluation_global_indices=active_indices,
            )
            batch_list, rewards, lengths, _, traj_uids, tool_callings = result
            trial_records = self._last_rollout_records
            failures = []
            for local_index, global_index in enumerate(active_indices):
                snapshot = (
                    batch_list[local_index],
                    float(rewards[local_index]),
                    float(lengths[local_index]),
                    str(traj_uids[local_index]),
                    float(tool_callings[local_index]),
                )
                selected[global_index] = snapshot
                if rewards[local_index] > 0:
                    solved[global_index] = True
                elif trial + 1 < trial_count:
                    failures.append((local_index, global_index))

            if failures:
                reflection_prompts = [
                    self._build_reflection_prompt(
                        trial_records[local_index], memories[global_index]
                    )
                    for local_index, global_index in failures
                ]
                reflections = self._generate_reflections(
                    reflection_prompts, actor_rollout_wg
                )
                for (local_index, global_index), reflection in zip(
                    failures, reflections, strict=True
                ):
                    trial_records[local_index]['reflection_after'] = reflection
                    memories[global_index].append(reflection)
                    memories[global_index] = memories[global_index][-memory_size:]
            all_records.extend(trial_records)
            print(
                f'ALFWORLD_REFLEXION_TRIAL\ttrial={trial}\t'
                f'attempted={len(active_indices)}\tcumulative_success={int(solved.sum())}/'
                f'{task_count}',
                flush=True,
            )

        if any(item is None for item in selected):
            raise RuntimeError('Reflexion did not produce a final trajectory for every task')
        final_batch_list = [item[0] for item in selected]
        final_rewards = np.asarray([item[1] for item in selected], dtype=np.float32)
        final_lengths = np.asarray([item[2] for item in selected], dtype=np.float32)
        final_traj_uids = np.asarray([item[3] for item in selected], dtype=object)
        final_tool_callings = np.asarray([item[4] for item in selected], dtype=np.float32)
        success = {'success_rate': final_rewards.copy()}
        fixed_files = [os.path.realpath(path) for path in envs.envs.fixed_game_files]
        for task_type in sorted({self._alfworld_task_type(path) for path in fixed_files}):
            values = [
                final_rewards[index]
                for index, path in enumerate(fixed_files)
                if self._alfworld_task_type(path) == task_type
            ]
            success[f'{task_type}_success_rate'] = np.asarray(values, dtype=np.float32)
        self._write_eval_records(all_records)
        return self.gather_rollout_data(
            total_batch_list=final_batch_list,
            episode_rewards=final_rewards,
            episode_lengths=final_lengths,
            success=success,
            traj_uid=final_traj_uids,
            tool_callings=final_tool_callings,
            sampling_metrics={
                'reflexion_trial_count': float(trial_count),
                'reflexion_attempt_count': float(len(all_records)),
            },
        )

    def _build_search_reflection_prompt(
        self,
        task: str,
        interaction_history: list[dict],
        memory: list[str],
    ) -> str:
        lines = [f"Question: {task}"]
        for step_index, step in enumerate(interaction_history, 1):
            lines.extend(
                [
                    f"Step {step_index}",
                    f"Action: {step.get('search', '')}",
                    f"Information: {step.get('information', '')}",
                ]
            )
        trace = "\n".join(lines)
        max_chars = int(
            self.config.env.search.reflexion.get(
                "reflection_max_input_chars", 6000
            )
        )
        if len(trace) > max_chars:
            header = lines[0] + "\n[Earlier search steps truncated]\n"
            trace = header + trace[-max(0, max_chars - len(header)) :]
        previous = (
            "\n\nPlans from earlier failed attempts:\n"
            + "\n".join(f"- {plan.strip()}" for plan in memory if plan.strip())
            if memory
            else ""
        )
        return (
            "You are reflecting on a failed information-seeking question-answering "
            "attempt. Diagnose the search queries, evidence use, and final-answer "
            "choice. Produce a concise new plan for the next reset, including better "
            "queries or how to extract the shortest supported answer. Do not answer "
            "the question and do not claim success. Return only `Plan: ...`.\n\n"
            f"Failed experience:\n{trace}{previous}\n\nNew plan:"
        )

    def search_reflexion_multi_turn_loop(self, gen_batch, actor_rollout_wg, envs):
        """Evaluate Search with Reflexion retries through the native agent loop."""
        task_count = len(gen_batch.batch)
        trial_count = int(self.config.env.search.reflexion.get("num_trials", 10))
        memory_size = int(self.config.env.search.reflexion.get("memory_size", 3))
        memories: list[list[str]] = [[] for _ in range(task_count)]
        solved = np.zeros(task_count, dtype=bool)
        selected = [None] * task_count
        attempt_count = 0

        for trial in range(trial_count):
            active_indices = [index for index in range(task_count) if not solved[index]]
            if not active_indices:
                break
            trial_batch = gen_batch.select_idxs(active_indices)
            trial_memory = [memories[index] for index in active_indices]
            result = self.vanilla_multi_turn_loop(
                gen_batch=trial_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                reset_kwargs={
                    "env_kwargs": trial_batch.non_tensor_batch.get("env_kwargs"),
                    "reflexion_memory": trial_memory,
                },
                evaluation_trial=trial,
                evaluation_global_indices=active_indices,
            )
            batch_list, rewards, lengths, _, traj_uids, tool_callings = result
            attempt_count += len(active_indices)
            attempt_histories = [list(envs.memory[index]) for index in range(len(active_indices))]
            failures = []
            for local_index, global_index in enumerate(active_indices):
                selected[global_index] = (
                    batch_list[local_index],
                    float(rewards[local_index]),
                    float(lengths[local_index]),
                    str(traj_uids[local_index]),
                    float(tool_callings[local_index]),
                )
                if rewards[local_index] > 0:
                    solved[global_index] = True
                elif trial + 1 < trial_count:
                    failures.append((local_index, global_index))

            if failures:
                prompts = [
                    self._build_search_reflection_prompt(
                        str(envs.tasks[local_index]),
                        attempt_histories[local_index],
                        memories[global_index],
                    )
                    for local_index, global_index in failures
                ]
                reflections = self._generate_reflections(prompts, actor_rollout_wg)
                max_reflection_chars = int(
                    self.config.env.search.reflexion.get(
                        "reflection_max_output_chars", 1200
                    )
                )
                for (_, global_index), reflection in zip(
                    failures, reflections, strict=True
                ):
                    memories[global_index].append(
                        reflection[:max_reflection_chars]
                    )
                    memories[global_index] = memories[global_index][-memory_size:]

            print(
                f"SEARCH_REFLEXION_TRIAL\ttrial={trial}\t"
                f"attempted={len(active_indices)}\tcumulative_success="
                f"{int(solved.sum())}/{task_count}",
                flush=True,
            )

        if any(item is None for item in selected):
            raise RuntimeError("Search Reflexion produced no trajectory for a task")
        final_batch_list = [item[0] for item in selected]
        final_rewards = np.asarray([item[1] for item in selected], dtype=np.float32)
        final_lengths = np.asarray([item[2] for item in selected], dtype=np.float32)
        final_traj_uids = np.asarray([item[3] for item in selected], dtype=object)
        final_tool_callings = np.asarray([item[4] for item in selected], dtype=np.float32)
        data_sources = np.asarray(gen_batch.non_tensor_batch["data_source"], dtype=object)
        success = {"success_rate": final_rewards.copy()}
        for data_source in np.unique(data_sources):
            success[f"{data_source}_success_rate"] = final_rewards[
                data_sources == data_source
            ]
        return self.gather_rollout_data(
            total_batch_list=final_batch_list,
            episode_rewards=final_rewards,
            episode_lengths=final_lengths,
            success=success,
            traj_uid=final_traj_uids,
            tool_callings=final_tool_callings,
            sampling_metrics={
                "reflexion_trial_count": float(trial_count),
                "reflexion_attempt_count": float(attempt_count),
            },
        )


    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)


        if (
            not is_train
            and 'alfworld' in str(self.config.env.env_name).lower()
            and bool(self.config.env.alfworld.reflexion.get('enabled', False))
        ):
            return self.reflexion_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        if (
            not is_train
            and 'search' in str(self.config.env.env_name).lower()
            and bool(self.config.env.search.reflexion.get('enabled', False))
        ):
            return self.search_reflexion_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            
        # Initial observations from the environment
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings, sampling_metrics = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            # Vanilla Sampling   
            sampling_metrics = None
            reset_kwargs = None
            evaluation_global_indices = None
            if not is_train and (
                'search' in str(self.config.env.env_name).lower()
            ):
                raw_indices = gen_batch.non_tensor_batch.get('index')
                if raw_indices is None:
                    extra_info = gen_batch.non_tensor_batch.get('extra_info')
                    if extra_info is not None:
                        raw_indices = [
                            value.get('index') if isinstance(value, dict) else None
                            for value in extra_info
                        ]
                if raw_indices is None or any(value is None for value in raw_indices):
                    raise RuntimeError(
                        'Indexed trajectory export requires an index either '
                        'at the dataset top level or in extra_info'
                    )
                evaluation_global_indices = [int(value) for value in raw_indices]
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                reset_kwargs=reset_kwargs,
                evaluation_global_indices=evaluation_global_indices,
            )
            if not is_train and self._last_rollout_records:
                if 'search' in str(self.config.env.env_name).lower():
                    self._write_search_eval_records(self._last_rollout_records)
                else:
                    self._write_eval_records(self._last_rollout_records)
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)
        

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
            sampling_metrics=sampling_metrics,
        )
        
        return gen_batch_output
