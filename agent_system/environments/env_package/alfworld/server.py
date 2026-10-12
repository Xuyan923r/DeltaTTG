"""FastAPI server hosting exact-game ALFWorld text environments."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent_system.environments.protocols.alfworld import MAX_ENV_STEPS
from .runtime import AlfWorldEpisode


@dataclass
class HostedEnvironment:
    episode: AlfWorldEpisode
    lock: threading.RLock = field(default_factory=threading.RLock)


class ResetItem(BaseModel):
    game_file: str
    max_steps: int = Field(default=MAX_ENV_STEPS, gt=0)


class ResetBatchRequest(BaseModel):
    items: list[ResetItem] = Field(min_length=1)


class StepItem(BaseModel):
    session_id: str
    action: str


class StepBatchRequest(BaseModel):
    items: list[StepItem] = Field(min_length=1)


class CloseBatchRequest(BaseModel):
    session_ids: list[str]


app = FastAPI(title="ALFWorld Environment Server", version="1.0")
_environments: dict[str, HostedEnvironment] = {}
_map_lock = threading.RLock()


@app.get("/health")
def health() -> dict[str, Any]:
    with _map_lock:
        active = len(_environments)
    return {"status": "ok", "active_environments": active}


@app.post("/v1/environments/reset_batch")
def reset_batch(request: ResetBatchRequest) -> dict[str, Any]:
    results = []
    created: list[tuple[str, HostedEnvironment]] = []
    try:
        for item in request.items:
            episode = AlfWorldEpisode.create(item.game_file, max_steps=item.max_steps)
            session_id = uuid4().hex
            hosted = HostedEnvironment(episode)
            created.append((session_id, hosted))
            results.append(
                {
                    "session_id": session_id,
                    "game_file": episode.game_file,
                    "observation": episode.initial_observation,
                    "admissible_commands": episode.admissible_commands,
                }
            )
    except Exception:
        for _, hosted in created:
            hosted.episode.close()
        raise
    with _map_lock:
        _environments.update(created)
    return {"items": results}


@app.post("/v1/environments/step_batch")
def step_batch(request: StepBatchRequest) -> dict[str, Any]:
    results = []
    for item in request.items:
        with _map_lock:
            hosted = _environments.get(item.session_id)
        if hosted is None:
            results.append({"session_id": item.session_id, "error": "unknown_session"})
            continue
        with hosted.lock:
            result = hosted.episode.step(item.action)
        result["session_id"] = item.session_id
        results.append(result)
        if result["terminated"] or result["truncated"]:
            with _map_lock:
                _environments.pop(item.session_id, None)
            hosted.episode.close()
    return {"items": results}


@app.post("/v1/environments/close_batch")
def close_batch(request: CloseBatchRequest) -> dict[str, int]:
    closed = 0
    for session_id in request.session_ids:
        with _map_lock:
            hosted = _environments.pop(session_id, None)
        if hosted is not None:
            hosted.episode.close()
            closed += 1
    return {"closed": closed}
