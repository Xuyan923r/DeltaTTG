"""Shared exact-game ALFWorld TextWorld runtime."""

from __future__ import annotations

import os
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import textworld
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

from agent_system.environments.protocols.alfworld import MAX_ENV_STEPS, format_observation


_environment_pool: deque[Any] = deque()
_environment_pool_lock = threading.Lock()
# TextWorld's PDDL loader and text generator use process-global Tatsu parsers
# whose mutable parse stacks are not thread-safe. FastAPI executes reset and
# step requests in different worker threads, so every operation entering
# TextWorld must be serialized across episodes.
_textworld_lock = threading.Lock()


def alfworld_data_root() -> Path:
    return Path(
        os.environ.get("ALFWORLD_DATA", "./data/alfworld")
    ).expanduser().resolve()


def validate_game_file(game_file: str | Path) -> Path:
    root = alfworld_data_root()
    path = Path(game_file).expanduser().resolve()
    if path.name != "game.tw-pddl" or root not in path.parents:
        raise ValueError(f"Game file must be a game.tw-pddl below {root}: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


@dataclass
class AlfWorldEpisode:
    env: Any
    game_file: str
    max_steps: int
    state: Any
    steps: int = 0

    @classmethod
    def create(cls, game_file: str | Path, max_steps: int = MAX_ENV_STEPS) -> "AlfWorldEpisode":
        path = validate_game_file(game_file)
        request_infos = textworld.EnvInfos(
            won=True,
            admissible_commands=True,
            extras=["gamefile"],
        )
        with _environment_pool_lock:
            env = _environment_pool.pop() if _environment_pool else None
        try:
            with _textworld_lock:
                if env is None:
                    env = textworld.start(
                        str(path),
                        request_infos=request_infos,
                        wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
                    )
                else:
                    env.load(str(path))
                state = env.reset()
        except BaseException:
            # A failed reload may leave Fast Downward in an unknown state.
            # Discard this environment; the process will reclaim it at exit.
            env = None
            raise
        return cls(env=env, game_file=str(path), max_steps=int(max_steps), state=state)

    @property
    def admissible_commands(self) -> list[str]:
        return list(self.state.admissible_commands)

    @property
    def initial_observation(self) -> str:
        return format_observation(self.state.feedback, self.admissible_commands)

    def step(self, action: str) -> dict[str, Any]:
        command = action.strip()
        invalid = command not in self.admissible_commands
        # Invalid commands are still sent to TextWorld. This advances the same
        # bounded interaction budget used by standalone rollout and GRPO.
        with _textworld_lock:
            self.state, reward, done = self.env.step(command)
        self.steps += 1
        success = bool(self.state.won)
        truncated = bool(not done and self.steps >= self.max_steps)
        finished = bool(done or truncated)
        return {
            "observation": format_observation(
                self.state.feedback,
                self.admissible_commands,
                action=command,
                steps=self.steps,
                success=success,
                finished=finished,
            ),
            "reward": float(success),
            "environment_reward": float(reward),
            "terminated": bool(done),
            "truncated": truncated,
            "success": success,
            "invalid_action": invalid,
            "steps": self.steps,
            "admissible_commands": self.admissible_commands,
        }

    def close(self) -> None:
        if self.env is not None:
            env, self.env = self.env, None
            # PddlEnv.close() is a no-op and every newly constructed PddlEnv
            # loads another private libdownward.so. Reuse the initialized
            # environment instead, bounding the number of shared-object
            # mappings to the server's peak episode concurrency.
            with _environment_pool_lock:
                _environment_pool.append(env)
