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
import yaml
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torchvision.transforms as T
import ray

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    # Project policy: every benchmark uses the same strict terminal binary
    # reward. Partial ALFRED goal-condition scores are metrics, never rewards.
    return float(bool(info['won']))

class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, config, seed, base_env, fixed_game_file=None):
        # A final benchmark worker can be pinned to exactly one official game.
        # This keeps the native Ray environment while preventing duplicate or
        # omitted tasks caused by independent random resets.
        self.fixed_game_file = fixed_game_file
        if fixed_game_file is not None:
            base_env.game_files = [fixed_game_file]
            base_env.num_games = 1
        self.env = base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        self.env.seed(seed)
    
    def step(self, action):
        """Execute a step in the environment"""
        actions = [action] 
        
        obs, scores, dones, infos = self.env.step(actions)
        infos['observation_text'] = obs
        return obs, scores, dones, infos
    
    def reset(self):
        """Reset the environment"""
        obs, infos = self.env.reset()
        infos['observation_text'] = obs
        return obs, infos
    
    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()  
        return image

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        env_type = config['env']['type']
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n
        fixed_game_files = None
        fixed_game_files_path = env_kwargs.get('fixed_game_files_path')
        if fixed_game_files_path:
            if is_train:
                raise ValueError('fixed_game_files_path is evaluation-only')
            with open(os.path.expanduser(fixed_game_files_path), encoding='utf-8') as source:
                payload = json.load(source)
            fixed_game_files = payload.get('game_files') if isinstance(payload, dict) else payload
            if not isinstance(fixed_game_files, list):
                raise ValueError('Fixed ALFWorld manifest must be a list or {"game_files": [...]}')
            fixed_game_files = [os.path.realpath(str(path)) for path in fixed_game_files]
            if len(fixed_game_files) != self.num_processes:
                raise ValueError(
                    f'Fixed ALFWorld manifest has {len(fixed_game_files)} games, '
                    f'but validation requested {self.num_processes} workers'
                )
            if len(set(fixed_game_files)) != len(fixed_game_files):
                raise ValueError('Fixed ALFWorld evaluation manifest contains duplicate games')
            available = {os.path.realpath(path) for path in base_env.game_files}
            unexpected = sorted(set(fixed_game_files).difference(available))
            if unexpected:
                raise ValueError(f'Fixed games do not belong to {eval_dataset}: {unexpected[:3]}')
        self.fixed_game_files = fixed_game_files
        self.active_indices = list(range(self.num_processes))

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []
        for i in range(self.num_processes):
            fixed_game_file = fixed_game_files[i] if fixed_game_files is not None else None
            worker = env_worker.remote(
                config, seed + (i // self.group_n), base_env, fixed_game_file
            )
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == len(self.active_indices), \
            "The num of actions must equal the active ALFWorld workers"

        # Send step commands to all workers
        futures = []
        active_workers = [self.workers[index] for index in self.active_indices]
        for i, worker in enumerate(active_workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            worker_index = self.active_indices[i]
            self.prev_admissible_commands[worker_index] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self, indices=None):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        if indices is None:
            self.active_indices = list(range(self.num_processes))
        else:
            self.active_indices = [int(index) for index in indices]
            if len(set(self.active_indices)) != len(self.active_indices):
                raise ValueError('Active ALFWorld worker indices must be unique')
            if any(index < 0 or index >= self.num_processes for index in self.active_indices):
                raise IndexError('Active ALFWorld worker index is out of range')

        # Send reset commands to active workers.
        futures = []
        for index in self.active_indices:
            worker = self.workers[index]
            future = worker.reset.remote()
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            worker_index = self.active_indices[i]
            self.prev_admissible_commands[worker_index] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        futures = []
        for index in self.active_indices:
            worker = self.workers[index]
            future = worker.getobs.remote()
            futures.append(future)

        images = ray.get(futures)
        return images

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return [self.prev_admissible_commands[index] for index in self.active_indices]

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs={}):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)
