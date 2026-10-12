"""Exact meter-prefix deduplication and result restoration."""

from __future__ import annotations

import hashlib
import json
import struct

from deltattg.types import MeterRequest, MeterResult


def meter_request_hash(request: MeterRequest) -> str:
    digest = hashlib.sha256()
    if request.input_ids is not None:
        for token_id in request.input_ids:
            digest.update(struct.pack("<q", int(token_id)))
        return digest.hexdigest()
    payload = {
        "messages": request.messages,
        "tools": request.tools,
        "serialized_prompt": request.serialized_prompt,
    }
    digest.update(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def deduplicate_meter_requests(
    requests: list[MeterRequest],
) -> tuple[list[MeterRequest], list[int]]:
    unique: list[MeterRequest] = []
    inverse: list[int] = []
    seen: dict[str, int] = {}
    for request in requests:
        key = meter_request_hash(request)
        unique_index = seen.get(key)
        if unique_index is None:
            unique_index = len(unique)
            seen[key] = unique_index
            unique.append(request)
        inverse.append(unique_index)
    return unique, inverse


def restore_deduplicated_estimates(
    unique_estimates: list[MeterResult],
    inverse_mapping: list[int],
) -> list[MeterResult]:
    if inverse_mapping and max(inverse_mapping) >= len(unique_estimates):
        raise ValueError("Estimate restoration mapping is out of range")
    return [unique_estimates[index] for index in inverse_mapping]
