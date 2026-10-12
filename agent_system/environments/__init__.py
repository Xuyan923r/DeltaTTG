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

"""Environment entry points with lazy imports for provisioning utilities."""


def make_envs(config):
    from agent_system.environments.env_manager import make_envs as _make_envs

    return _make_envs(config)


def __getattr__(name):
    if name == "EnvironmentManagerBase":
        from agent_system.environments.base import EnvironmentManagerBase

        return EnvironmentManagerBase
    raise AttributeError(name)


__all__ = ["EnvironmentManagerBase", "make_envs"]
