#!/usr/bin/env python3
"""Build leakage-safe PaperSearchQA progress-meter SFT/RL data.

The input is the native ``papersearchqa_verl_agent_raw_trajectory_v1`` format.
Every pre-action state becomes one point-estimate example.  The label is the
terminal outcome of that sampled continuation and, for successful
continuations only, the suffix sum of assistant-generated tokens.  Search
observations are visible in the prefix but never counted in the token target.

Tasks are split before prefix construction. SFT and RL use the same train
partition; validation and test tasks remain held out. No class, source, or
length resampling is applied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from deltattg.meter.prompt import build_meter_messages  # noqa: E402
from deltattg.meter.schema import SCHEMA_VERSION, build_target_text  # noqa: E402


RAW_SCHEMA_VERSION = "papersearchqa_verl_agent_raw_trajectory_v1"
DATA_SCHEMA_VERSION = "papersearchqa_continuation_action_tokens_v1"
TARGET_DEFINITION = (
    "future assistant-generated action tokens only; current/future search "
    "observations excluded"
)
SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the PubMed BM25 corpus for supporting evidence.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    "agent_action_format": "<search>QUERY</search>",
}

MESSAGE_TYPE = pa.list_(
    pa.struct([pa.field("role", pa.string()), pa.field("content", pa.string())])
)
SFT_SCHEMA = pa.schema(
    [
        pa.field("messages", MESSAGE_TYPE),
        pa.field("tools", pa.list_(pa.string())),
        pa.field("enable_thinking", pa.bool_()),
        pa.field("metadata", pa.string()),
    ]
)
RL_SCHEMA = pa.schema(
    [
        pa.field("data_source", pa.string()),
        pa.field("prompt", MESSAGE_TYPE),
        pa.field("ability", pa.string()),
        pa.field(
            "reward_model",
            pa.struct(
                [
                    pa.field("style", pa.string()),
                    pa.field(
                        "ground_truth",
                        pa.struct(
                            [
                                pa.field("can_complete", pa.bool_()),
                                pa.field("remaining_action_tokens", pa.int64()),
                            ]
                        ),
                    ),
                ]
            ),
        ),
        pa.field(
            "extra_info",
            pa.struct(
                [
                    pa.field("sample_id", pa.string()),
                    pa.field("task_id", pa.string()),
                    pa.field("trajectory_id", pa.string()),
                    pa.field("source_kind", pa.string()),
                    pa.field("state_idx", pa.int64()),
                    pa.field("split", pa.string()),
                    pa.field("reference_model", pa.string()),
                    pa.field("source_success", pa.bool_()),
                    pa.field("searches_before_state", pa.int64()),
                    pa.field("target_token_definition", pa.string()),
                ]
            ),
        ),
    ]
)

# Approximately equal-mass bins under the inspected 109,814-rollout mixture.


def parse_args() -> argparse.Namespace:
    data_root = Path(
        "./data/PaperSearchQA"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qwen3-4b-input",
        type=Path,
        default=data_root
        / "meter_raw/trajectories/qwen3-4b-instruct-2507/seed_0.jsonl",
    )
    parser.add_argument(
        "--qwen35-27b-input",
        type=Path,
        default=data_root / "meter_raw/trajectories/qwen3.5-27b/seed_0.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_root / "meter_action_tokens",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path(
            "./models/"
            "Qwen3-4B-Instruct-2507"
        ),
    )
    parser.add_argument("--max-seq-length", type=int, default=16384)
    parser.add_argument("--split-seed", type=int, default=20260825)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--batch-rows", type=int, default=256)
    args = parser.parse_args()
    if args.max_seq_length <= 0:
        parser.error("sequence length must be positive")
    if not 0 < args.train_ratio < 1:
        parser.error("--train-ratio must be in (0, 1)")
    if not 0 <= args.validation_ratio < 1 - args.train_ratio:
        parser.error("--validation-ratio leaves no test split")
    return args


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hash_split(task_id: str, seed: int, train: float, validation: float) -> str:
    digest = hashlib.sha256(f"{seed}:{task_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < train:
        return "train"
    if value < train + validation:
        return "validation"
    return "test"


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                yield line_number, json.loads(line)




def parse_trajectory(record: dict[str, Any], source_kind: str) -> dict[str, Any]:
    if record.get("schema_version") != RAW_SCHEMA_VERSION:
        raise ValueError("unsupported raw trajectory schema")
    steps = record.get("steps") or []
    if not steps or len(steps) != int(record.get("num_steps", -1)):
        raise ValueError("empty trajectory or num_steps mismatch")
    if not bool(steps[-1].get("environment_done")):
        raise ValueError("final step is not terminal")
    if bool(record["success"]) != (float(record["reward"]) >= 1.0):
        raise ValueError("success/reward mismatch")
    turns = []
    for expected_index, step in enumerate(steps):
        if int(step["step_index"]) != expected_index:
            raise ValueError("non-contiguous step index")
        token_ids = step.get("actor_response_token_ids") or []
        token_count = int(step["actor_response_token_count"])
        if token_count != len(token_ids) or token_count <= 0:
            raise ValueError("invalid actor response token count")
        turns.append(
            {
                "assistant": str(step.get("actor_response") or ""),
                "projected_action": str(step.get("projected_action") or ""),
                "assistant_tokens": token_count,
                "observation": str(
                    step.get("environment_observation_after") or ""
                ),
                "action_valid": bool(step.get("action_valid")),
            }
        )
    source_messages = record.get("source_prompt_messages") or []
    system_text = ""
    for message in source_messages:
        if message.get("role") == "system":
            system_text = str(message.get("content") or "")
            break
    task_index = int(record["global_task_index"])
    return {
        "source_kind": source_kind,
        "task_id": f"papersearchqa_train_{task_index}",
        "task_index": task_index,
        "trajectory_id": str(record["trajectory_id"]),
        "success": bool(record["success"]),
        "stop_reason": str(
            steps[-1].get("environment_info_after", {}).get(
                "postprocessed_action", ""
            )
        ),
        "original_agent_system": system_text,
        "reference_model": str(record["model_label"]),
        "initial_observation": str(steps[0]["environment_observation_before"]),
        "task": str(record["task"]),
        "turns": turns,
        "token_count_source": "step.actor_response_token_count",
        "source_metadata": {
            "seed": int(record["seed"]),
            "global_task_index": task_index,
            "model_path": str(record.get("model_path") or ""),
            "rollout_settings": record["rollout_settings"],
        },
    }


def history(turns: list[dict[str, Any]], state_idx: int) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in turns[:state_idx]:
        messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append(
            {
                "role": "user",
                "content": (
                    "<environment_observation>\n"
                    f"{turn['observation']}\n"
                    "</environment_observation>"
                ),
            }
        )
    return messages


def suffix_tokens(turns: list[dict[str, Any]]) -> list[int]:
    result = [0] * len(turns)
    running = 0
    for index in range(len(turns) - 1, -1, -1):
        running += int(turns[index]["assistant_tokens"])
        result[index] = running
    return result


def sequence_length(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": False,
        "enable_thinking": False,
    }
    try:
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    return len(ids[0] if ids and isinstance(ids[0], list) else ids)


class Writers:
    def __init__(self, root: Path, batch_rows: int):
        self.root = root
        self.batch_rows = batch_rows
        self.buffers: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self.json_files: dict[tuple[str, str], Any] = {}
        self.parquet_files: dict[tuple[str, str], pq.ParquetWriter] = {}
        for kind in ("sft", "rl"):
            (root / kind).mkdir(parents=True, exist_ok=True)
            for split in ("train", "validation", "test"):
                self.json_files[(kind, split)] = (
                    root / kind / f"{split}.jsonl"
                ).open("w", encoding="utf-8")

    @staticmethod
    def schema(kind: str) -> pa.Schema:
        return SFT_SCHEMA if kind == "sft" else RL_SCHEMA

    def add(self, kind: str, split: str, row: dict[str, Any]) -> None:
        key = (kind, split)
        self.json_files[key].write(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        self.buffers[key].append(row)
        if len(self.buffers[key]) >= self.batch_rows:
            self.flush(key)

    def flush(self, key: tuple[str, str]) -> None:
        rows = self.buffers[key]
        if not rows:
            return
        kind, split = key
        writer = self.parquet_files.get(key)
        if writer is None:
            writer = pq.ParquetWriter(
                self.root / kind / f"{split}.parquet",
                self.schema(kind),
                compression="zstd",
                use_dictionary=True,
            )
            self.parquet_files[key] = writer
        writer.write_table(pa.Table.from_pylist(rows, schema=self.schema(kind)))
        rows.clear()

    def close(self) -> None:
        for key, json_file in self.json_files.items():
            self.flush(key)
            if key in self.parquet_files:
                self.parquet_files[key].close()
            else:
                kind, split = key
                pq.write_table(
                    pa.Table.from_pylist([], schema=self.schema(kind)),
                    self.root / kind / f"{split}.parquet",
                    compression="zstd",
                )
            json_file.close()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=False
    )
    writers = Writers(output_dir, args.batch_rows)
    invalid = (output_dir / "invalid_trajectories.jsonl").open(
        "w", encoding="utf-8"
    )
    overlong = (output_dir / "overlong_prefixes.jsonl").open(
        "w", encoding="utf-8"
    )
    counts: Counter[str] = Counter()
    task_splits: dict[str, str] = {}
    sequence_lengths: list[int] = []
    completed = False

    def emit(trajectory: dict[str, Any]) -> None:
        split = hash_split(
            trajectory["task_id"],
            args.split_seed,
            args.train_ratio,
            args.validation_ratio,
        )
        old_split = task_splits.setdefault(trajectory["task_id"], split)
        if old_split != split:
            raise RuntimeError("task-level split inconsistency")
        turns = trajectory["turns"]
        suffix = suffix_tokens(turns)
        states = range(len(turns))
        counts[f"source_trajectories_{trajectory['source_kind']}"] += 1
        counts[
            f"source_trajectories_{trajectory['source_kind']}_"
            + ("success" if trajectory["success"] else "failure")
        ] += 1
        counts["prefix_candidates"] += len(states)
        for state_idx in states:
            remaining = suffix[state_idx] if trajectory["success"] else None
            prompt = build_meter_messages(
                original_agent_system=trajectory["original_agent_system"],
                tools=[SEARCH_TOOL_SCHEMA],
                original_task=trajectory["task"],
                initial_observation=trajectory["initial_observation"],
                history_messages=history(turns, state_idx),
                assistant_token_counts=[
                    int(turn["assistant_tokens"]) for turn in turns[:state_idx]
                ],
                completed_turns=state_idx,
            )
            # Pool observed completion effort across sources; model and
            # decoding metadata are retained for provenance, outside the prompt.
            target = {
                "can_complete": bool(trajectory["success"]),
                "remaining_action_tokens": remaining,
            }
            target_text = build_target_text(bool(trajectory["success"]), remaining)
            sft_messages = [*prompt, {"role": "assistant", "content": target_text}]
            length = sequence_length(tokenizer, sft_messages)
            if length > args.max_seq_length:
                counts["overlong_prefixes_dropped"] += 1
                overlong.write(
                    json.dumps(
                        {
                            "trajectory_id": trajectory["trajectory_id"],
                            "source_kind": trajectory["source_kind"],
                            "state_idx": state_idx,
                            "sequence_tokens": length,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue
            sequence_lengths.append(length)
            searches_before = sum(
                turn["projected_action"].startswith("<search>")
                for turn in turns[:state_idx]
            )
            sample_id = stable_hash(
                [trajectory["trajectory_id"], state_idx, DATA_SCHEMA_VERSION]
            )
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "dataset_schema_version": DATA_SCHEMA_VERSION,
                "sample_id": sample_id,
                "task_id": trajectory["task_id"],
                "task_index": trajectory["task_index"],
                "trajectory_id": trajectory["trajectory_id"],
                "source_kind": trajectory["source_kind"],
                "source_metadata": trajectory["source_metadata"],
                "split": split,
                "state_idx": state_idx,
                "trajectory_turns": len(turns),
                "searches_before_state": searches_before,
                "source_success": bool(trajectory["success"]),
                "source_stop_reason": trajectory["stop_reason"],
                "observed_remaining_action_tokens_until_termination": suffix[state_idx],
                **target,
                "target_token_definition": TARGET_DEFINITION,
                "token_count_source": trajectory["token_count_source"],
                "reference_model": trajectory["reference_model"],
                "sequence_tokens": length,
            }
            writers.add(
                "sft",
                split,
                {
                    "messages": sft_messages,
                    "tools": [],
                    "enable_thinking": False,
                    "metadata": json.dumps(
                        metadata, ensure_ascii=False, separators=(",", ":")
                    ),
                },
            )
            writers.add(
                "rl",
                split,
                {
                    "data_source": "papersearchqa_action_token_meter",
                    "prompt": prompt,
                    "ability": "continuation_action_token_estimate",
                    "reward_model": {"style": "rule", "ground_truth": target},
                    "extra_info": {
                        "sample_id": sample_id,
                        "task_id": trajectory["task_id"],
                        "trajectory_id": trajectory["trajectory_id"],
                        "source_kind": trajectory["source_kind"],
                        "state_idx": state_idx,
                        "split": split,
                        "reference_model": trajectory["reference_model"],
                        "source_success": bool(trajectory["success"]),
                        "searches_before_state": searches_before,
                        "target_token_definition": TARGET_DEFINITION,
                    },
                },
            )
            counts[f"samples_{split}"] += 1
            counts[f"samples_{trajectory['source_kind']}"] += 1
            counts[
                "samples_" + ("success" if trajectory["success"] else "failure")
            ] += 1
            counts[f"samples_searches_before_{min(searches_before, 2)}"] += 1

    sources = (
        ("qwen3_4b", args.qwen3_4b_input),
        ("qwen35_27b", args.qwen35_27b_input),
    )
    try:
        for source_kind, path in sources:
            for line_number, record in iter_jsonl(path):
                counts[f"input_rows_{source_kind}"] += 1
                try:
                    emit(parse_trajectory(record, source_kind))
                except Exception as exc:
                    counts[f"invalid_trajectories_{source_kind}"] += 1
                    invalid.write(
                        json.dumps(
                            {
                                "source_kind": source_kind,
                                "line_number": line_number,
                                "global_task_index": record.get("global_task_index"),
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        completed = True
    finally:
        writers.close()
        invalid.close()
        overlong.close()

    sorted_lengths = sorted(sequence_lengths)
    percentile = lambda value: (
        sorted_lengths[round((len(sorted_lengths) - 1) * value)]
        if sorted_lengths
        else None
    )
    manifest = {
        "schema_version": DATA_SCHEMA_VERSION,
        "target_schema_version": SCHEMA_VERSION,
        "complete": completed,
        "inputs": {kind: str(path.resolve()) for kind, path in sources},
        "rejected_input_included": False,
        "output_dir": str(output_dir),
        "prefix_selection": {
            "strategy": "all_pre_action_states",

            "future_messages_removed": True,
            "max_sequence_tokens": args.max_seq_length,
            "overlong_strategy": "drop",
        },
        "ground_truth": {
            "can_complete": "source sampled continuation terminal success",
            "remaining_action_tokens": (
                "suffix sum of source step.actor_response_token_count for "
                "successful trajectories; null for failures"
            ),
            "observations_counted": False,
            "target_token_definition": TARGET_DEFINITION,
        },
        "split": {
            "unit": "PaperSearchQA global_task_index shared across source models",
            "seed": args.split_seed,
            "train_ratio": args.train_ratio,
            "validation_ratio": args.validation_ratio,
            "test_ratio": 1 - args.train_ratio - args.validation_ratio,
            "task_counts": dict(Counter(task_splits.values())),
        },
        "counts": dict(counts),
        "sequence_tokens": {
            "min": min(sorted_lengths) if sorted_lengths else None,
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": max(sorted_lengths) if sorted_lengths else None,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
