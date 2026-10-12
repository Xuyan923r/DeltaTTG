"""Equal-length evaluation shards for collective-based model inference."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar


T = TypeVar("T")


def shard_eval_records(
    records: Sequence[T], rank: int, world_size: int
) -> list[tuple[int | None, T]]:
    """Keep each real global index once; mark padding records with ``None``.

    Equal shard lengths keep every rank in the same number of generation
    batches, including ranks with no real samples. Padding must participate in
    model forwards but must not contribute to metrics or saved estimates.
    """
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("Expected world_size > 0 and 0 <= rank < world_size")
    if not records:
        raise ValueError("Evaluation requires at least one record")
    shard = [(index, records[index]) for index in range(rank, len(records), world_size)]
    shard_size = (len(records) + world_size - 1) // world_size
    shard.extend((None, records[0]) for _ in range(shard_size - len(shard)))
    return shard
