#!/usr/bin/env python3
"""Build ALFWorld progress-meter SFT/RL datasets.

Every valid input trajectory is cut at all pre-action boundaries. No source,
outcome, or length balancing is applied. Future messages are
discarded from the input; the label keeps only the observed terminal outcome
and the suffix sum of assistant/action tokens for successful continuations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from agent_system.environments.protocols.alfworld import PROTOCOL_VERSION, TOOL_SCHEMA  # noqa: E402
from deltattg.meter.prompt import build_meter_messages  # noqa: E402
from deltattg.meter.schema import SCHEMA_VERSION, build_target_text  # noqa: E402


TARGET_DEFINITION = (
    "future assistant-generated action tokens only; current/future "
    "tool/environment observations excluded"
)
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
                    pa.field("target_token_definition", pa.string()),
                ]
            ),
        ),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qwen-input",
        type=Path,
        default=ROOT
        / "TTGData/ALFWorld/raw/qwen3_4b_instruct_2507_alfworld_train_3x.jsonl",
    )
    parser.add_argument(
        "--expert-input",
        type=Path,
        default=ROOT / "TTGData/ALFWorld/expert/train.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "TTGData/ALFWorld/meter_action_tokens",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / "models/Qwen3-4B-Instruct-2507",
    )
    parser.add_argument("--max-seq-length", type=int, default=16384)
    parser.add_argument("--split-seed", type=int, default=20260803)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument(
        "--reference-model", default="Qwen3-4B-Instruct-2507"
    )
    args = parser.parse_args()
    if args.max_seq_length <= 0:
        parser.error("sequence length must be positive")
    if not 0 < args.train_ratio < 1:
        parser.error("--train-ratio must be between zero and one")
    if not 0 <= args.validation_ratio < 1 - args.train_ratio:
        parser.error("--validation-ratio leaves no test split")
    return args


def _hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _task_split(task_id: str, args: argparse.Namespace) -> str:
    digest = hashlib.sha256(f"{args.split_seed}:{task_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < args.train_ratio:
        return "train"
    if value < args.train_ratio + args.validation_ratio:
        return "validation"
    return "test"


def _task_id_from_game(game_file: str) -> str:
    game = Path(game_file)
    parts = game.parts
    if "train" in parts:
        index = len(parts) - 1 - list(reversed(parts)).index("train")
        return str(Path(*parts[index + 1 : -1]))
    return str(game.parent)


def _task_description(initial_observation: str) -> str:
    match = re.search(
        r"Your task is to:\s*(.+?)(?:\n\s*\n|\Z)",
        initial_observation,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("initial observation has no ALFWorld task description")
    return " ".join(match.group(1).split())


def _tool_call(action: str) -> str:
    payload = {
        "name": "env_step",
        "arguments": {"action": action},
    }
    return (
        "<tool_call>\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\n</tool_call>"
    )


def _qwen_trajectory(record: dict[str, Any]) -> dict[str, Any]:
    if int(record.get("alfworld_protocol_version", -1)) != PROTOCOL_VERSION:
        raise ValueError("unsupported ALFWorld protocol")
    initial: str | None = None
    turns: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    for event in record["trajectory"]:
        kind = event.get("type")
        if kind == "reset":
            if initial is not None:
                raise ValueError("multiple reset events")
            initial = str(event["observation"])
        elif kind == "generation":
            if pending is not None:
                raise ValueError("generation without intervening step")
            count = event.get("completion_tokens")
            if not isinstance(count, int) or count < 0:
                raise ValueError("invalid generation completion_tokens")
            pending = {
                "assistant": str(event.get("assistant") or ""),
                "assistant_tokens": count,
                "observation": None,
            }
            turns.append(pending)
        elif kind == "step":
            if pending is None:
                raise ValueError("step without preceding generation")
            pending["observation"] = str(event["observation"])
            pending = None
    if initial is None or not turns:
        raise ValueError("trajectory lacks reset or generations")
    if any(turn["observation"] is None for turn in turns[:-1]):
        raise ValueError("non-final generation lacks an observation")
    total = sum(int(turn["assistant_tokens"]) for turn in turns)
    if total != int(record["completion_tokens"]):
        raise ValueError("stored Qwen action-token total is inconsistent")
    game_file = str(record["game_file"])
    task_id = _task_id_from_game(game_file)
    return {
        "source_kind": "qwen",
        "task_id": task_id,
        "task_index": int(record["task_index"]),
        "trajectory_id": (
            f"qwen_rollout_{int(record['rollout_id'])}"
            f"__sampling_seed_{int(record['sampling_seed'])}"
        ),
        "game_file": game_file,
        "success": bool(record["success"]),
        "stop_reason": record.get("stop_reason"),
        "original_agent_system": str(record["messages"][0]["content"]),
        "initial_observation": initial,
        "task": _task_description(initial),
        "turns": turns,
        "token_count_source": "generation.completion_tokens",
        "source_metadata": {
            "rollout_id": int(record["rollout_id"]),
            "rollout_index": int(record["rollout_index"]),
            "sampling_seed": int(record["sampling_seed"]),
        },
    }


def _expert_trajectory(record: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    if not bool(record.get("success")):
        raise ValueError("expert trajectory is not successful")
    turns = []
    for step in record["steps"]:
        assistant = _tool_call(str(step["expert_action"]))
        turns.append(
            {
                "assistant": assistant,
                "assistant_tokens": len(
                    tokenizer.encode(assistant, add_special_tokens=False)
                ),
                "observation": str(step["next_observation"]),
            }
        )
    if not turns:
        raise ValueError("expert trajectory has no steps")
    return {
        "source_kind": "expert",
        "task_id": str(record["task_id"]),
        "task_index": -1,
        "trajectory_id": f"expert__{record['task_id']}",
        "game_file": str(record["game_file"]),
        "success": True,
        "stop_reason": "verified_expert_success",
        "original_agent_system": str(record["system_prompt"]),
        "initial_observation": str(record["initial_observation"]),
        "task": str(record["task"]),
        "turns": turns,
        "token_count_source": (
            "canonical env_step tool-call text tokenized with meter tokenizer"
        ),
        "source_metadata": {
            "expert_type": str(record["expert_type"]),
        },
    }


def _history(turns: list[dict[str, Any]], state_idx: int) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for turn in turns[:state_idx]:
        result.append({"role": "assistant", "content": turn["assistant"]})
        observation = turn["observation"]
        if observation is None:
            raise ValueError("selected state follows missing observation")
        result.append(
            {
                "role": "user",
                "content": (
                    "<environment_observation>\n"
                    f"{observation}\n"
                    "</environment_observation>"
                ),
            }
        )
    return result


def _suffix_tokens(turns: list[dict[str, Any]]) -> list[int]:
    result = [0] * len(turns)
    running = 0
    for index in range(len(turns) - 1, -1, -1):
        running += int(turns[index]["assistant_tokens"])
        result[index] = running
    return result


def _jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                yield line_number, json.loads(line)


class Writers:
    def __init__(self, root: Path, batch_rows: int):
        self.root = root
        self.batch_rows = batch_rows
        self.buffers: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        self.json: dict[tuple[str, str], Any] = {}
        self.parquet: dict[tuple[str, str], pq.ParquetWriter] = {}
        for kind in ("sft", "rl"):
            (root / kind).mkdir(parents=True, exist_ok=True)
            for split in ("train", "validation", "test"):
                self.json[(kind, split)] = (
                    root / kind / f"{split}.jsonl"
                ).open("w", encoding="utf-8")

    @staticmethod
    def schema(kind: str) -> pa.Schema:
        return SFT_SCHEMA if kind == "sft" else RL_SCHEMA

    def add(self, kind: str, split: str, row: dict[str, Any]) -> None:
        key = (kind, split)
        self.json[key].write(
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
        writer = self.parquet.get(key)
        if writer is None:
            writer = pq.ParquetWriter(
                self.root / kind / f"{split}.parquet",
                self.schema(kind),
                compression="zstd",
                use_dictionary=True,
            )
            self.parquet[key] = writer
        writer.write_table(pa.Table.from_pylist(rows, schema=self.schema(kind)))
        rows.clear()

    def close(self) -> None:
        for key in self.json:
            self.flush(key)
            if key in self.parquet:
                self.parquet[key].close()
            else:
                kind, split = key
                pq.write_table(
                    pa.Table.from_pylist([], schema=self.schema(kind)),
                    self.root / kind / f"{split}.parquet",
                    compression="zstd",
                )
            self.json[key].close()


def _sequence_length(tokenizer: Any, messages: list[dict[str, str]]) -> int:
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
    seq_lengths: list[int] = []
    completed = False

    def emit(trajectory: dict[str, Any]) -> None:
        split = _task_split(trajectory["task_id"], args)
        old_split = task_splits.setdefault(trajectory["task_id"], split)
        if old_split != split:
            raise ValueError("task-level split inconsistency")
        turns = trajectory["turns"]
        suffix = _suffix_tokens(turns)
        selected_states = range(len(turns))
        counts[f"source_trajectories_{trajectory['source_kind']}"] += 1
        counts[
            f"source_trajectories_{trajectory['source_kind']}_"
            f"{'success' if trajectory['success'] else 'failure'}"
        ] += 1
        counts["prefix_candidates"] += len(selected_states)
        for state_idx in selected_states:
            remaining = suffix[state_idx] if trajectory["success"] else None
            prompt = build_meter_messages(
                original_agent_system=trajectory["original_agent_system"],
                tools=[TOOL_SCHEMA],
                original_task=trajectory["task"],
                initial_observation=trajectory["initial_observation"],
                history_messages=_history(turns, state_idx),
                assistant_token_counts=[
                    int(turn["assistant_tokens"])
                    for turn in turns[:state_idx]
                ],
                completed_turns=state_idx,
            )
            ground_truth = {
                "can_complete": bool(trajectory["success"]),
                "remaining_action_tokens": remaining,
            }
            target = build_target_text(bool(trajectory["success"]), remaining)
            sft_messages = [
                *prompt,
                {"role": "assistant", "content": target},
            ]
            length = _sequence_length(tokenizer, sft_messages)
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
            seq_lengths.append(length)
            sample_id = _hash(
                [trajectory["trajectory_id"], state_idx, SCHEMA_VERSION]
            )
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "sample_id": sample_id,
                "task_id": trajectory["task_id"],
                "task_index": trajectory["task_index"],
                "trajectory_id": trajectory["trajectory_id"],
                "source_kind": trajectory["source_kind"],
                "source_metadata": trajectory["source_metadata"],
                "game_file": trajectory["game_file"],
                "split": split,
                "state_idx": state_idx,
                "trajectory_turns": len(turns),
                "source_success": bool(trajectory["success"]),
                "source_stop_reason": trajectory["stop_reason"],
                "observed_remaining_action_tokens_until_termination": suffix[
                    state_idx
                ],
                **ground_truth,
                "target_token_definition": TARGET_DEFINITION,
                "token_count_source": trajectory["token_count_source"],
                "reference_model": args.reference_model,
                "sequence_tokens": length,
            }
            sft_row = {
                "messages": sft_messages,
                "tools": [],
                "enable_thinking": False,
                "metadata": json.dumps(
                    metadata, ensure_ascii=False, separators=(",", ":")
                ),
            }
            rl_row = {
                "data_source": "alfworld_action_token_meter",
                "prompt": prompt,
                "ability": "continuation_action_token_estimate",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    "sample_id": sample_id,
                    "task_id": trajectory["task_id"],
                    "trajectory_id": trajectory["trajectory_id"],
                    "source_kind": trajectory["source_kind"],
                    "state_idx": state_idx,
                    "split": split,
                    "reference_model": args.reference_model,
                    "source_success": bool(trajectory["success"]),
                    "target_token_definition": TARGET_DEFINITION,
                },
            }
            writers.add("sft", split, sft_row)
            writers.add("rl", split, rl_row)
            counts[f"samples_{split}"] += 1
            counts[f"samples_{trajectory['source_kind']}"] += 1
            counts[f"samples_{'success' if trajectory['success'] else 'failure'}"] += 1

    try:
        for line_number, record in _jsonl(args.expert_input):
            counts["expert_input_rows"] += 1
            try:
                emit(_expert_trajectory(record, tokenizer))
            except Exception as exc:
                counts["invalid_expert_trajectories"] += 1
                invalid.write(
                    json.dumps(
                        {
                            "source": "expert",
                            "line_number": line_number,
                            "task_id": record.get("task_id"),
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        qwen_tasks: set[str] = set()
        for line_number, record in _jsonl(args.qwen_input):
            task_id = _task_id_from_game(str(record["game_file"]))
            counts["qwen_input_rows"] += 1
            qwen_tasks.add(task_id)
            try:
                emit(_qwen_trajectory(record))
            except Exception as exc:
                counts["invalid_qwen_trajectories"] += 1
                invalid.write(
                    json.dumps(
                        {
                            "source": "qwen",
                            "line_number": line_number,
                            "task_id": task_id,
                            "rollout_id": record.get("rollout_id"),
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        counts["qwen_tasks"] = len(qwen_tasks)
        completed = True
    finally:
        writers.close()
        invalid.close()
        overlong.close()

    sorted_lengths = sorted(seq_lengths)
    percentile = lambda value: (
        sorted_lengths[round((len(sorted_lengths) - 1) * value)]
        if sorted_lengths
        else None
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": completed,
        "inputs": {
            "qwen": str(args.qwen_input.resolve()),
            "expert": str(args.expert_input.resolve()),
        },
        "output_dir": str(output_dir),
        "source_mixture": {
            "selection": "all valid trajectories from each supplied source",
            "repeated_prefixes": "retained with each recorded continuation",
        },
        "prefix_selection": {
            "strategy": "all_pre_action_states",

            "future_messages_removed": True,
            "max_sequence_tokens": args.max_seq_length,
            "overlong_strategy": "drop",
        },
        "ground_truth": {
            "can_complete": "source trajectory terminal success",
            "remaining_action_tokens": (
                "suffix sum of assistant/action tokens for successful source "
                "trajectories; null for failures"
            ),
            "observations_counted": False,
            "target_token_definition": TARGET_DEFINITION,
        },
        "split": {
            "unit": "task_id",
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
        "formats": {
            "sft": "messages; loss target is the final assistant message only",
            "rl": "verl prompt plus rule reward_model.ground_truth",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
