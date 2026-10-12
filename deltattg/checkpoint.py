#!/usr/bin/env python3
"""Merge a verl FSDP actor checkpoint into a Hugging Face safetensors model."""

from __future__ import annotations

import argparse
import os
import re
import shutil
from collections.abc import Mapping
from pathlib import Path

import torch
from huggingface_hub import save_torch_state_dict

try:
    from torch.distributed.tensor import DTensor
except ImportError:  # pragma: no cover - compatibility with older torch
    from torch.distributed._tensor import DTensor


NON_WEIGHT_FILES = (
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


def discover_world_size(actor_dir: Path) -> int:
    candidates: list[int] = []
    for path in actor_dir.glob("model_world_size_*_rank_0.pt"):
        match = re.fullmatch(r"model_world_size_(\d+)_rank_0\.pt", path.name)
        if match:
            candidates.append(int(match.group(1)))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one checkpoint world size in {actor_dir}, got {candidates}"
        )
    return candidates[0]


def load_shards(actor_dir: Path, world_size: int) -> list[dict[str, object]]:
    shards: list[dict[str, object]] = []
    for rank in range(world_size):
        path = actor_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing FSDP model shard: {path}")
        state = torch.load(
            path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        if not isinstance(state, Mapping) or not state:
            raise TypeError(f"FSDP shard is not a non-empty state dict: {path}")
        shards.append(dict(state))
    expected_keys = set(shards[0])
    for rank, shard in enumerate(shards[1:], start=1):
        if set(shard) != expected_keys:
            raise RuntimeError(f"Parameter keys differ between rank 0 and rank {rank}")
    return shards


def merge_shards(shards: list[dict[str, object]]) -> dict[str, torch.Tensor]:
    merged: dict[str, torch.Tensor] = {}
    for key in sorted(shards[0]):
        values = [shard.pop(key) for shard in shards]
        first = values[0]
        if isinstance(first, DTensor):
            placements = tuple(first.placements)
            if len(placements) != 1:
                raise NotImplementedError(f"Multiple DTensor placements for {key}: {placements}")
            for rank, value in enumerate(values):
                if not isinstance(value, DTensor):
                    raise TypeError(f"Mixed DTensor/plain tensors for {key} at rank {rank}")
                if tuple(value.placements) != placements:
                    raise RuntimeError(f"DTensor placements differ for {key} at rank {rank}")
            placement = placements[0]
            local_tensors = [value._local_tensor for value in values]
            if placement.is_shard():
                tensor = torch.cat(local_tensors, dim=placement.dim).contiguous()
            elif placement.is_replicate():
                tensor = local_tensors[0].contiguous()
            elif placement.is_partial():
                raise NotImplementedError(f"Partial DTensor placement is unsupported: {key}")
            else:
                raise NotImplementedError(f"Unknown DTensor placement for {key}: {placement}")
            if tuple(tensor.shape) != tuple(first.shape):
                raise RuntimeError(
                    f"Merged shape mismatch for {key}: got {tuple(tensor.shape)}, "
                    f"expected {tuple(first.shape)}"
                )
        else:
            if not all(isinstance(value, torch.Tensor) for value in values):
                raise TypeError(f"Unexpected non-tensor checkpoint entry: {key}")
            # FSDP sharded state dicts store plain local shards along dim 0.
            tensor = torch.cat(values, dim=0).contiguous()
        merged[key] = tensor
    return merged


def main() -> int:
    args = parse_args()
    actor_dir = args.actor_dir.resolve()
    target_dir = args.target_dir.resolve()
    if target_dir.exists():
        raise FileExistsError(f"Refusing to overwrite merge target: {target_dir}")
    if not (actor_dir / "config.json").is_file():
        raise FileNotFoundError(f"Missing Hugging Face config: {actor_dir / 'config.json'}")

    world_size = discover_world_size(actor_dir)
    print(f"Loading {world_size} FSDP shards from {actor_dir}", flush=True)
    shards = load_shards(actor_dir, world_size)
    merged = merge_shards(shards)
    del shards
    print(f"Merged {len(merged)} tensors; writing {target_dir}", flush=True)

    temporary = target_dir.with_name(f".{target_dir.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=True)
    try:
        for name in NON_WEIGHT_FILES:
            source = actor_dir / name
            if source.is_file():
                shutil.copy2(source, temporary / name)
        save_torch_state_dict(
            merged,
            temporary,
            max_shard_size=args.max_shard_size,
            safe_serialization=True,
            metadata={"format": "pt"},
        )
        if not any(temporary.glob("*.safetensors")):
            raise RuntimeError(f"No safetensors weights were written under {temporary}")
        os.replace(temporary, target_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(f"MERGED_HF\t{actor_dir}\t{target_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

