"""Validate the extra GPU required by a dedicated Ray meter worker."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def validate_meter_gpu_resources(
    config: Mapping[str, Any], available_resources: Mapping[str, float]
) -> None:
    if not config.get("credit_assignment", {}).get("enabled", False):
        return
    backend = str(config.get("meter", {}).get("backend", "transformers")).lower()
    if backend not in {"ray_vllm", "ray_transformers"}:
        return
    trainer = config["trainer"]
    actor_gpus = int(trainer["n_gpus_per_node"]) * int(trainer["nnodes"])
    required_gpus = actor_gpus + 1
    available_gpus = float(available_resources.get("GPU", 0))
    if available_gpus < required_gpus:
        raise ValueError(
            f"DeltaTTG meter.backend={backend} requires {actor_gpus} actor GPUs "
            f"plus 1 dedicated meter GPU ({required_gpus} total), but Ray has "
            f"{available_gpus:g} available GPUs. Allocate the extra GPU to this "
            "Ray job; ACTOR_GPUS counts actor training/rollout GPUs only."
        )
