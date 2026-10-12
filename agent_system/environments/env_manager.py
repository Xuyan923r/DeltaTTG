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

from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.protocols.alfworld import METER_PREFIX_ADAPTER, format_meter_observation
from agent_system.environments.protocols.papersearchqa import METER_PREFIX_ADAPTER as PAPERSEARCHQA_METER_PREFIX_ADAPTER
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory, SearchMemory
from omegaconf import OmegaConf

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        self.prompt_mode = str(config.env.search.get("prompt_mode", "thinking"))
        if self.prompt_mode not in {"thinking", "non_thinking"}:
            raise ValueError(
                "env.search.prompt_mode must be 'thinking' or 'non_thinking', "
                f"got {self.prompt_mode!r}"
            )
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        reflexion_memory = None
        if isinstance(kwargs, dict) and "env_kwargs" in kwargs:
            reflexion_memory = kwargs.get("reflexion_memory")
            kwargs = kwargs["env_kwargs"]
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        if reflexion_memory is None:
            reflexion_memory = [[] for _ in obs]
        if len(reflexion_memory) != len(obs):
            raise ValueError("Search Reflexion memory count must match active tasks")
        self.reflexion_memory = [list(items) for items in reflexion_memory]

        self.memory.reset(batch_size=len(obs))

        full_text_obs = self.build_text_obs(obs, init=True)
        # Actor prompts follow prompt_mode. Meter inputs use the canonical
        # non-thinking initial observation shared with meter training data.
        # Keep the two formats separate when constructing rollout inputs.
        meter_initial_obs = [
            SEARCH_TEMPLATE_NON_THINKING_NO_HIS.format(
                task_description=task
            )
            for task in self.tasks
        ]
        observations = {
            "text": full_text_obs,
            "image": None,
            "anchor": obs.copy(),
            # Meter-only canonical observation; actor still sees
            # ``full_text_obs`` selected by ``prompt_mode`` above.
            "deltattg_meter_text": meter_initial_obs,
            "deltattg_prefix_adapter": [
                PAPERSEARCHQA_METER_PREFIX_ADAPTER
            ] * len(obs),
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy(),
            "deltattg_meter_text": [str(value) for value in next_obs],
            "deltattg_prefix_adapter": [
                PAPERSEARCHQA_METER_PREFIX_ADAPTER
            ] * len(next_obs),
        }

        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])
            # Preserve both sides of the native Search transition for raw
            # trajectory export.  `actions` is the projection consumed by the
            # environment, whereas `text_actions` is the model's raw response.
            info["projected_action"] = str(actions[i])
            info["environment_observation_after"] = str(next_obs[i])
            info["environment_done"] = bool(dones[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            no_history_template = (
                SEARCH_TEMPLATE_NON_THINKING_NO_HIS
                if self.prompt_mode == "non_thinking"
                else SEARCH_TEMPLATE_NO_HIS
            )
            history_template = (
                SEARCH_TEMPLATE_NON_THINKING
                if self.prompt_mode == "non_thinking"
                else SEARCH_TEMPLATE
            )
            if init or self.config.env.history_length <= 0:
                obs_i = no_history_template.format(
                    task_description=self.tasks[i]
                )
            else:
                obs_i = history_template.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            episodic_memory = self.reflexion_memory[i]
            if episodic_memory:
                max_plan_chars = int(
                    self.config.env.search.reflexion.get(
                        "reflection_max_output_chars", 1200
                    )
                )
                plans = "\n".join(
                    f"- {plan.strip()[:max_plan_chars]}"
                    for plan in episodic_memory
                    if plan.strip()
                )
                obs_i = (
                    obs_i.rstrip()
                    + "\n\n<reflexion_memory>\n"
                    + "Plans learned from earlier failed attempts on this same question:\n"
                    + plans
                    + "\n</reflexion_memory>\n"
                    + "Use these lessons to choose a better search strategy or final answer."
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask
            

class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        kwargs = kwargs or {}
        active_indices = kwargs.get('active_indices')
        text_obs, image_obs, infos = self.envs.reset(indices=active_indices)
        memory = kwargs.get('reflexion_memory')
        if memory is None:
            memory = [[] for _ in text_obs]
        if len(memory) != len(text_obs):
            raise ValueError('Reflexion memory count must match active ALFWorld workers')
        self.reflexion_memory = [list(items) for items in memory]
        admissible_commands = self.envs.get_admissible_commands
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(text_obs, admissible_commands, init=True)
        meter_text_obs = [
            format_meter_observation(str(feedback), list(commands))
            for feedback, commands in zip(text_obs, admissible_commands, strict=True)
        ]
        return {
            'text': full_text_obs,
            'image': image_obs,
            'anchor': text_obs,
            'deltattg_meter_text': meter_text_obs,
            'deltattg_prefix_adapter': [
                METER_PREFIX_ADAPTER
            ] * len(text_obs),
        }, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        admissible_commands = self.envs.get_admissible_commands
        full_text_obs = self.build_text_obs(text_obs, admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        meter_text_obs = [
            format_meter_observation(
                str(feedback),
                list(commands),
                action=str(action),
                steps=len(self.memory[index]),
                success=bool(info.get('won', False)),
                finished=bool(dones[index]),
            )
            for index, (feedback, commands, action, info) in enumerate(
                zip(text_obs, admissible_commands, actions, infos, strict=True)
            )
        ]
        next_observations = {
            'text': full_text_obs,
            'image': image_obs,
            'anchor': text_obs,
            'deltattg_meter_text': meter_text_obs,
            'deltattg_prefix_adapter': [
                METER_PREFIX_ADAPTER
            ] * len(text_obs),
        }

        return next_observations, rewards, dones, infos
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            episodic_memory = self.reflexion_memory[i] if hasattr(self, 'reflexion_memory') else []
            if episodic_memory:
                plans = "\n\n".join(
                    f"Plan from failed trial {index + 1}:\n{plan.strip()}"
                    for index, plan in enumerate(episodic_memory)
                )
                block = (
                    "\n\n<reflexion_memory>\n"
                    "Use these plans from previous failed attempts to avoid repeating "
                    "the same mistakes:\n\n"
                    f"{plans}\n"
                    "</reflexion_memory>\n"
                )
                marker = "\nNow it's your turn to take an action."
                obs = obs.replace(marker, block + marker, 1) if marker in obs else obs + block

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break


def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(search_projection)
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset, # 'eval_in_distribution' or 'eval_out_of_distribution'
            'fixed_game_files_path': config.env.alfworld.get('fixed_game_files_path'),
        }
        # Validation-only benchmark jobs should not allocate the otherwise
        # unused training Ray workers. The trainer and validation agent loop
        # remain identical; this only removes idle environments.
        _envs = None
        if not bool(config.trainer.get('val_only', False)):
            _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs, resources_per_worker=resources_per_worker)
        
        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config) if _envs is not None else None
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        raise ValueError(f"Unsupported environment: {config.env.env_name}")
