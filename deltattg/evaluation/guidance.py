#!/usr/bin/env python3
"""Compare natural and meter-controlled Qwen ReAct rollouts on ALFWorld."""

from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import re
import statistics
import time
import traceback
from typing import Any

import httpx
from transformers import AutoTokenizer

from agent_system.environments.prompts.alfworld import (
    ALFWORLD_TEMPLATE,
    ALFWORLD_TEMPLATE_NO_HIS,
)
from agent_system.environments.protocols.alfworld import format_meter_observation
from deltattg.integration.adapters.alfworld import build_alfworld_meter_messages
from deltattg.meter.chat_template import apply_chat_template_compat
from deltattg.meter.parser import parse_meter_output


FAILURE_COST = 25_600.0
DEFAULT_METER_MAX_MODEL_LEN = 262_144


def task_from_initial(feedback: str) -> str:
    marker = "Your task is to: "
    if marker not in feedback:
        raise ValueError("ALFWorld initial observation lacks task marker")
    return feedback.split(marker, 1)[1].split("\n\n", 1)[0].strip()


def observation_without_action_catalog(observation: str) -> str:
    return observation.split("\n\nAdmissible actions:\n", 1)[0].rstrip()


def format_actions(actions: list[str]) -> str:
    return "\n ".join(f"'{action}'" for action in actions if action != "help")


def actor_prompt(
    feedback: str,
    actions: list[str],
    task: str,
    history: list[tuple[str, str]],
    history_length: int,
) -> str:
    if not history or history_length <= 0:
        return ALFWORLD_TEMPLATE_NO_HIS.format(
            current_observation=observation_without_action_catalog(feedback),
            admissible_actions=format_actions(actions),
        )
    recent = history[-history_length:]
    start = len(history) - len(recent)
    memory = "\n".join(
        f"[Observation {start + offset + 1}: "
        f"'{observation_without_action_catalog(old_feedback)}', "
        f"Action {start + offset + 1}: '{action}']"
        for offset, (old_feedback, action) in enumerate(recent)
    )
    return ALFWORLD_TEMPLATE.format(
        task_description=task,
        step_count=len(history),
        history_length=len(recent),
        action_history=memory,
        current_step=len(history) + 1,
        current_observation=observation_without_action_catalog(feedback),
        admissible_actions=format_actions(actions),
    )


def canonicalize_qwen_no_think_react(response: str) -> tuple[str, bool]:
    """Repair only the Qwen no-think chat-template boundary."""
    text = response.strip()
    if re.search(r"<think>.*?</think>\s*<action>.*?</action>", text, re.I | re.S):
        return text, False
    text = re.sub(r"^\s*</think>\s*", "", text, count=1, flags=re.I)
    action = re.search(r"<action>\s*(.*?)\s*</action>", text, re.I | re.S)
    if action is None:
        return response, False
    reasoning = text[: action.start()].strip()
    reasoning = re.sub(r"^\s*<think>\s*", "", reasoning, count=1, flags=re.I)
    reasoning = re.sub(r"\s*</think>\s*$", "", reasoning, count=1, flags=re.I)
    if not reasoning:
        reasoning = "Select the admissible action that best advances the task."
    normalized = (
        f"<think>\n{reasoning}\n</think>\n\n"
        f"<action>{action.group(1).strip()}</action>"
    )
    return normalized, normalized != response


def meter_cost(result: Any, failure_cost: float = FAILURE_COST) -> float:
    if not result.valid or not result.can_complete:
        return float(failure_cost)
    if result.remaining_action_tokens is None:
        return float(failure_cost)
    return float(result.remaining_action_tokens)


def choose_candidate(
    before_cost: float, after_costs: list[float]
) -> tuple[int, list[float]]:
    if not after_costs:
        raise ValueError("At least one candidate is required")
    progress = [float(before_cost) - float(value) for value in after_costs]
    # Ties select the first maximum; candidate 0 is selected when all scores tie.
    selected = max(range(len(progress)), key=progress.__getitem__)
    return selected, progress


def deduplicate_response_indices(responses: list[str]) -> tuple[list[int], list[int | None]]:
    """Return first-occurrence indices and the representative of each duplicate.

    Deduplication is deliberately based on the complete raw actor response.  This
    avoids changing the experimental semantics by merging merely similar actions.
    """
    first_seen: dict[str, int] = {}
    unique_indices: list[int] = []
    duplicate_of: list[int | None] = []
    for index, response in enumerate(responses):
        representative = first_seen.get(response)
        if representative is None:
            first_seen[response] = index
            unique_indices.append(index)
            duplicate_of.append(None)
        else:
            duplicate_of.append(representative)
    return unique_indices, duplicate_of


def environment_message(observation: str) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            "<environment_observation>\n"
            + observation
            + "\n</environment_observation>"
        ),
    }


class CompletionAPI:
    def __init__(self, url: str, model: str, timeout: float = 600.0):
        self.url = url.rstrip("/")
        self.model = model
        self.client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self.client.close()

    def complete(
        self,
        prompts: str | list[str],
        *,
        n: int = 1,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        seed: int,
        stop: list[str] | None = None,
    ) -> tuple[list[str], dict[str, int]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompts,
            "n": n,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        if top_k >= 0:
            payload["top_k"] = top_k
        if stop:
            payload["stop"] = stop
            payload["include_stop_str_in_output"] = True
        response = self.client.post(self.url + "/v1/completions", json=payload)
        response.raise_for_status()
        body = response.json()
        choices = sorted(body["choices"], key=lambda row: int(row["index"]))
        usage = body.get("usage") or {}
        return [str(row.get("text", "")) for row in choices], {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        }


@dataclass
class Candidate:
    index: int
    raw_response: str
    response: str
    response_normalized: bool
    response_token_count: int
    projected_action: str
    format_valid: bool
    executable: bool
    observation_after: str
    environment_reward: float
    success_after: bool
    terminated_after: bool
    truncated_after: bool
    duplicate_of: int | None = None
    meter_after: dict[str, Any] | None = None
    cost_after: float | None = None
    estimated_progress: float | None = None


@dataclass
class TokenUsage:
    actor_prompt_tokens: int = 0
    actor_generated_tokens: int = 0
    executed_action_tokens: int = 0
    meter_prompt_tokens: int = 0
    meter_generated_tokens: int = 0


@dataclass
class EpisodeState:
    task_index: int
    game_file: str
    seed: int
    mode: str
    episode: AlfWorldEpisode
    task: str
    initial_meter_observation: str
    actor_history: list[tuple[str, str]] = field(default_factory=list)
    meter_history: list[dict[str, str]] = field(default_factory=list)
    assistant_token_counts: list[int] = field(default_factory=list)
    selected_actions: list[str] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    cached_before_estimate: Any | None = None
    cached_before_cost: float | None = None


class Evaluator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.actor = CompletionAPI(args.actor_url, args.actor_served_name)
        self.meter = CompletionAPI(
            args.meter_url, args.meter_served_name
        )
        self.actor_tokenizer = AutoTokenizer.from_pretrained(
            args.actor_model, local_files_only=True
        )
        self.meter_tokenizer = AutoTokenizer.from_pretrained(
            args.meter_model, local_files_only=True
        )

    def close(self) -> None:
        self.actor.close()
        self.meter.close()

    def render_actor(self, state: EpisodeState) -> tuple[str, int]:
        prompt = actor_prompt(
            str(state.episode.state.feedback),
            state.episode.admissible_commands,
            state.task,
            state.actor_history,
            self.args.history_length,
        )
        rendered = apply_chat_template_compat(
            self.actor_tokenizer,
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = self.actor_tokenizer.encode(
            rendered, add_special_tokens=False
        )
        if len(input_ids) > self.args.max_prompt_length:
            raise RuntimeError(
                f"Actor prompt {len(input_ids)} exceeds "
                f"max_prompt_length={self.args.max_prompt_length}"
            )
        return rendered, len(input_ids)

    def meter_messages(
        self,
        state: EpisodeState,
        history: list[dict[str, str]],
        token_counts: list[int],
    ) -> list[dict[str, Any]]:
        return build_alfworld_meter_messages(
            original_task=state.task,
            initial_observation=state.initial_meter_observation,
            history_messages=history,
            assistant_token_counts=token_counts,
            completed_turns=len(token_counts),
        )

    def estimate_many(
        self, state: EpisodeState, histories: list[tuple[list[dict[str, str]], list[int]]]
    ) -> tuple[list[Any], dict[str, int]]:
        prompts = []
        for history, counts in histories:
            messages = self.meter_messages(state, history, counts)
            rendered = apply_chat_template_compat(
                self.meter_tokenizer,
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            input_ids = self.meter_tokenizer.encode(
                rendered, add_special_tokens=False
            )
            if len(input_ids) + self.args.meter_max_tokens > self.args.meter_max_model_len:
                raise RuntimeError(
                    f"Meter prompt {len(input_ids)} exceeds model budget"
                )
            prompts.append(rendered)
        outputs, usage = self.meter.complete(
            prompts,
            n=1,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=self.args.meter_max_tokens,
            seed=self.args.meter_seed,
        )
        parsed = [parse_meter_output(text) for text in outputs]
        invalid = [index for index, result in enumerate(parsed) if not result.valid]
        if invalid:
            retries, retry_usage = self.meter.complete(
                [prompts[index] for index in invalid],
                n=1,
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                max_tokens=self.args.meter_max_tokens,
                seed=self.args.meter_seed,
            )
            for key in usage:
                usage[key] += retry_usage.get(key, 0)
            for index, text in zip(invalid, retries, strict=True):
                parsed[index] = parse_meter_output(text, retried=True)
        return parsed, usage

    def simulate_candidate(
        self,
        state: EpisodeState,
        action: str,
    ) -> dict[str, Any]:
        from agent_system.environments.env_package.alfworld.runtime import AlfWorldEpisode

        branch = AlfWorldEpisode.create(
            state.game_file, max_steps=self.args.max_steps
        )
        try:
            for selected in state.selected_actions:
                branch.step(selected)
            if (
                str(branch.state.feedback) != str(state.episode.state.feedback)
                or branch.admissible_commands != state.episode.admissible_commands
            ):
                raise RuntimeError("ALFWorld replay branch diverged from live state")
            result = branch.step(action)
            observation = format_meter_observation(
                str(branch.state.feedback),
                branch.admissible_commands,
                action=action,
                steps=branch.steps,
                success=bool(result["success"]),
                finished=bool(result["terminated"] or result["truncated"]),
            )
            return {**result, "meter_observation": observation}
        finally:
            branch.close()

    def run_one(self, task_index: int, game_file: str, seed: int, mode: str) -> dict[str, Any]:
        from agent_system.environments.env_package.alfworld.runtime import AlfWorldEpisode
        from agent_system.environments.env_package.alfworld.projection import alfworld_projection

        episode = AlfWorldEpisode.create(game_file, max_steps=self.args.max_steps)
        feedback = str(episode.state.feedback)
        state = EpisodeState(
            task_index=task_index,
            game_file=game_file,
            seed=seed,
            mode=mode,
            episode=episode,
            task=task_from_initial(feedback),
            initial_meter_observation=format_meter_observation(
                feedback, episode.admissible_commands
            ),
        )
        started = time.monotonic()
        success = False
        stop_reason = "max_steps"
        try:
            while len(state.steps) < self.args.max_steps:
                step_index = len(state.steps)
                rendered, prompt_tokens = self.render_actor(state)
                candidate_count = 1 if mode == "natural" else self.args.candidates
                raw_outputs, actor_usage = self.actor.complete(
                    rendered,
                    n=candidate_count,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    top_k=self.args.top_k,
                    max_tokens=self.args.max_response_length,
                    seed=seed + (step_index + 1) * 1_000_003,
                    stop=["</action>"],
                )
                state.usage.actor_prompt_tokens += int(actor_usage["prompt_tokens"])
                state.usage.actor_generated_tokens += int(actor_usage["completion_tokens"])
                candidates: list[Candidate] = []
                projected_actions = []
                normalized_rows = []
                for raw in raw_outputs:
                    normalized, changed = canonicalize_qwen_no_think_react(raw)
                    projected, valid = alfworld_projection(
                        [normalized], [state.episode.admissible_commands]
                    )
                    normalized_rows.append((raw, normalized, changed, bool(valid[0])))
                    projected_actions.append(str(projected[0]))
                unique_indices, duplicate_of = deduplicate_response_indices(raw_outputs)
                branch_results = {
                    index: self.simulate_candidate(state, projected_actions[index])
                    for index in unique_indices
                }
                for index, (row, action) in enumerate(
                    zip(normalized_rows, projected_actions, strict=True)
                ):
                    raw, normalized, changed, valid = row
                    representative = duplicate_of[index]
                    branch = branch_results[
                        index if representative is None else representative
                    ]
                    candidates.append(
                        Candidate(
                            index=index,
                            raw_response=raw,
                            response=normalized,
                            response_normalized=changed,
                            response_token_count=len(
                                self.actor_tokenizer.encode(
                                    raw, add_special_tokens=False
                                )
                            ),
                            projected_action=action,
                            format_valid=valid,
                            executable=not bool(branch["invalid_action"]),
                            observation_after=str(branch["meter_observation"]),
                            environment_reward=float(branch["environment_reward"]),
                            success_after=bool(branch["success"]),
                            terminated_after=bool(branch["terminated"]),
                            truncated_after=bool(branch["truncated"]),
                            duplicate_of=representative,
                        )
                    )

                before_estimate = None
                cost_before = None
                meter_usage = {"prompt_tokens": 0, "completion_tokens": 0}
                selected_index = 0
                if mode == "meter":
                    if state.cached_before_estimate is None:
                        before_rows, before_usage = self.estimate_many(
                            state,
                            [(list(state.meter_history), list(state.assistant_token_counts))],
                        )
                        before_estimate = before_rows[0]
                        cost_before = meter_cost(
                            before_estimate, self.args.failure_cost
                        )
                        before_source = "fresh"
                    else:
                        before_estimate = state.cached_before_estimate
                        cost_before = state.cached_before_cost
                        before_usage = {"prompt_tokens": 0, "completion_tokens": 0}
                        before_source = "cached_selected_after"
                    after_histories = []
                    unique_candidates = [
                        candidate
                        for candidate in candidates
                        if candidate.duplicate_of is None
                    ]
                    for candidate in unique_candidates:
                        history = list(state.meter_history)
                        history.append({"role": "assistant", "content": candidate.response})
                        history.append(environment_message(candidate.observation_after))
                        after_histories.append(
                            (history, state.assistant_token_counts + [candidate.response_token_count])
                        )
                    after_rows, after_usage = self.estimate_many(state, after_histories)
                    meter_usage = {
                        key: int(before_usage.get(key, 0)) + int(after_usage.get(key, 0))
                        for key in ("prompt_tokens", "completion_tokens")
                    }
                    estimates_by_index: dict[int, Any] = {}
                    for candidate, estimate in zip(
                        unique_candidates, after_rows, strict=True
                    ):
                        estimates_by_index[candidate.index] = estimate
                    after_costs = []
                    for candidate in candidates:
                        representative = (
                            candidate.index
                            if candidate.duplicate_of is None
                            else candidate.duplicate_of
                        )
                        estimate = estimates_by_index[representative]
                        candidate.meter_after = asdict(estimate)
                        candidate.cost_after = meter_cost(
                            estimate, self.args.failure_cost
                        )
                        after_costs.append(candidate.cost_after)
                    selected_index, progress = choose_candidate(cost_before, after_costs)
                    for candidate, value in zip(candidates, progress, strict=True):
                        candidate.estimated_progress = value
                    state.usage.meter_prompt_tokens += meter_usage["prompt_tokens"]
                    state.usage.meter_generated_tokens += meter_usage["completion_tokens"]
                else:
                    before_source = None

                selected = candidates[selected_index]
                before_feedback = str(state.episode.state.feedback)
                result = state.episode.step(selected.projected_action)
                state.usage.executed_action_tokens += selected.response_token_count
                state.steps.append(
                    {
                        "step_index": step_index,
                        "actor_prompt": rendered,
                        "actor_prompt_token_count": prompt_tokens,
                        "meter_before": (
                            asdict(before_estimate) if before_estimate is not None else None
                        ),
                        "meter_before_source": before_source,
                        "cost_before": cost_before,
                        "failure_cost": self.args.failure_cost,
                        "selection_rule": "max(remaining_before-remaining_after)",
                        "selected_candidate_index": selected_index,
                        "unique_candidate_count": len(unique_indices),
                        "candidates": [asdict(candidate) for candidate in candidates],
                        "meter_api_usage": meter_usage,
                    }
                )
                state.actor_history.append((before_feedback, selected.projected_action))
                state.selected_actions.append(selected.projected_action)
                state.meter_history.append(
                    {"role": "assistant", "content": selected.response}
                )
                selected_after = format_meter_observation(
                    str(state.episode.state.feedback),
                    state.episode.admissible_commands,
                    action=selected.projected_action,
                    steps=state.episode.steps,
                    success=bool(result["success"]),
                    finished=bool(result["terminated"] or result["truncated"]),
                )
                if selected_after != selected.observation_after:
                    raise RuntimeError(
                        "Selected ALFWorld branch diverged after applying action"
                    )
                state.meter_history.append(environment_message(selected_after))
                state.assistant_token_counts.append(selected.response_token_count)
                if mode == "meter":
                    state.cached_before_estimate = estimates_by_index[
                        selected.index
                        if selected.duplicate_of is None
                        else selected.duplicate_of
                    ]
                    state.cached_before_cost = selected.cost_after
                success = bool(result["success"])
                if success:
                    stop_reason = "success"
                    break
                if result["terminated"]:
                    stop_reason = "environment_terminated"
                    break
                if result["truncated"]:
                    stop_reason = "max_steps"
                    break
        finally:
            episode.close()
        return {
            "schema_version": "alfworld_meter_controlled_rollout_v2",
            "task_index": task_index,
            "game_file": game_file,
            "task": state.task,
            "seed": seed,
            "mode": mode,
            "success": success,
            "reward": float(success),
            "num_steps": len(state.steps),
            "stop_reason": stop_reason,
            "usage": asdict(state.usage),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "settings": {
                "actor_model": self.args.actor_model,
                "meter_model": self.args.meter_model,
                "candidates": 1 if mode == "natural" else self.args.candidates,
                "history_length": self.args.history_length,
                "max_steps": self.args.max_steps,
                "max_prompt_length": self.args.max_prompt_length,
                "max_response_length": self.args.max_response_length,
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
                "top_k": self.args.top_k,
                "enable_thinking": False,
                "failure_cost": self.args.failure_cost,
                "episode_concurrency": self.args.episode_concurrency,
                "cache_selected_after_as_next_before": True,
                "deduplicate_exact_raw_responses": True,
            },
            "steps": state.steps,
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


_worker_evaluator: Evaluator | None = None


def _init_episode_worker(args: argparse.Namespace) -> None:
    """Keep TextWorld's global lock local to each process, not all episodes."""
    global _worker_evaluator
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _worker_evaluator = Evaluator(args)
    atexit.register(_worker_evaluator.close)


def _run_episode_worker(task_index: int, game_file: str, seed: int, mode: str) -> dict[str, Any]:
    if _worker_evaluator is None:
        raise RuntimeError("Episode worker was not initialized")
    return _worker_evaluator.run_one(task_index, game_file, seed, mode)


def load_games(manifest: Path) -> list[str]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    games = [str(Path(item).resolve()) for item in payload["game_files"]]
    if len(games) != int(payload["game_count"]) or len(set(games)) != len(games):
        raise RuntimeError(f"Invalid ALFWorld manifest: {manifest}")
    if any(not Path(game).is_file() for game in games):
        raise FileNotFoundError("ALFWorld manifest contains missing game files")
    return games


def safe_mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def mean_std(values: list[float]) -> dict[str, float | int | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
        "n": len(values),
    }


def summarize_records(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if row["success"]]
    return {
        "trajectories": len(rows),
        "successes": len(successes),
        "success_rate": len(successes) / len(rows) if rows else None,
        "mean_steps": safe_mean([float(row["num_steps"]) for row in rows]),
        "mean_executed_action_tokens_all": safe_mean(
            [float(row["usage"]["executed_action_tokens"]) for row in rows]
        ),
        "mean_executed_action_tokens_success": safe_mean(
            [float(row["usage"]["executed_action_tokens"]) for row in successes]
        ),
        "mean_actor_generated_tokens": safe_mean(
            [float(row["usage"]["actor_generated_tokens"]) for row in rows]
        ),
        "mean_meter_prompt_tokens": safe_mean(
            [float(row["usage"]["meter_prompt_tokens"]) for row in rows]
        ),
        "mean_meter_generated_tokens": safe_mean(
            [float(row["usage"]["meter_generated_tokens"]) for row in rows]
        ),
    }


def summarize(output_root: Path) -> None:
    grouped: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    trajectory_paths = [
        path
        for mode in ("natural", "meter")
        for path in output_root.glob(f"*/seed_*/{mode}.jsonl")
    ]
    for path in sorted(trajectory_paths):
        split = path.parents[1].name
        seed = path.parent.name
        mode = path.stem
        grouped.setdefault(split, {}).setdefault(seed, {}).setdefault(mode, []).extend(
            read_jsonl(path)
        )
    result: dict[str, Any] = {"schema_version": "alfworld_meter_controlled_summary_v1"}
    for split, seeds in grouped.items():
        split_result: dict[str, Any] = {"by_seed": {}}
        pooled: dict[str, list[dict[str, Any]]] = {}
        for seed, modes in seeds.items():
            seed_result = {
                mode: summarize_records(rows) for mode, rows in modes.items()
            }
            for mode, rows in modes.items():
                pooled.setdefault(mode, []).extend(rows)
            if "natural" in seed_result and "meter" in seed_result:
                natural = seed_result["natural"]
                guided = seed_result["meter"]
                seed_result["difference_meter_minus_natural"] = {
                    "success_rate": guided["success_rate"] - natural["success_rate"],
                    "mean_executed_action_tokens_all": (
                        guided["mean_executed_action_tokens_all"]
                        - natural["mean_executed_action_tokens_all"]
                    ),
                    "mean_executed_action_tokens_success": (
                        None
                        if guided["mean_executed_action_tokens_success"] is None
                        or natural["mean_executed_action_tokens_success"] is None
                        else guided["mean_executed_action_tokens_success"]
                        - natural["mean_executed_action_tokens_success"]
                    ),
                }
            split_result["by_seed"][seed] = seed_result

        split_result["pooled"] = {
            mode: summarize_records(rows) for mode, rows in pooled.items()
        }
        split_result["seed_statistics"] = {}
        for mode in ("natural", "meter"):
            per_seed = [
                row[mode]
                for row in split_result["by_seed"].values()
                if mode in row
            ]
            split_result["seed_statistics"][mode] = {
                metric: mean_std(
                    [float(row[metric]) for row in per_seed if row[metric] is not None]
                )
                for metric in (
                    "success_rate",
                    "mean_executed_action_tokens_all",
                    "mean_executed_action_tokens_success",
                )
            }
        differences = [
            row["difference_meter_minus_natural"]
            for row in split_result["by_seed"].values()
            if "difference_meter_minus_natural" in row
        ]
        split_result["seed_statistics"]["difference_meter_minus_natural"] = {
            metric: mean_std(
                [float(row[metric]) for row in differences if row[metric] is not None]
            )
            for metric in (
                "success_rate",
                "mean_executed_action_tokens_all",
                "mean_executed_action_tokens_success",
            )
        }
        result[split] = split_result
    destination = output_root / "summary.json"
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"SUMMARY {destination}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--actor-url", required=True)
    run.add_argument("--meter-url", required=True)
    run.add_argument("--actor-served-name", default="actor")
    run.add_argument("--meter-served-name", default="meter")
    run.add_argument("--actor-model", required=True)
    run.add_argument("--meter-model", required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--mode", choices=("natural", "meter"), required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--max-tasks", type=int, default=0)
    run.add_argument("--candidates", type=int, default=4)
    run.add_argument("--history-length", type=int, default=2)
    run.add_argument("--max-steps", type=int, default=50)
    run.add_argument("--max-prompt-length", type=int, default=2048)
    run.add_argument("--max-response-length", type=int, default=512)
    run.add_argument("--temperature", type=float, default=1.0)
    run.add_argument("--top-p", type=float, default=1.0)
    run.add_argument("--top-k", type=int, default=-1)
    run.add_argument("--failure-cost", type=float, default=FAILURE_COST)
    run.add_argument("--meter-max-tokens", type=int, default=32)
    run.add_argument(
        "--meter-max-model-len",
        type=int,
        default=DEFAULT_METER_MAX_MODEL_LEN,
    )
    run.add_argument("--meter-seed", type=int, default=0)
    run.add_argument("--episode-concurrency", type=int, default=1)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "summarize":
        summarize(args.output_root.resolve())
        return
    if args.candidates < 2 and args.mode == "meter":
        raise ValueError("Meter mode requires at least two candidates")
    if args.episode_concurrency < 1:
        raise ValueError("episode_concurrency must be positive")
    games = load_games(args.manifest.resolve())
    if args.max_tasks > 0:
        games = games[: args.max_tasks]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(args.output)
    completed = {int(row["task_index"]) for row in existing}
    errors_path = args.output.with_suffix(".errors.jsonl")
    with args.output.open("a", encoding="utf-8", buffering=1) as destination:
        for attempt in (1, 2):
            pending_tasks = [
                (task_index, game_file)
                for task_index, game_file in enumerate(games)
                if task_index not in completed
            ]
            if not pending_tasks:
                break
            print(
                f"RUN mode={args.mode} attempt={attempt} pending={len(pending_tasks)} "
                f"workers={args.episode_concurrency}",
                flush=True,
            )
            with ProcessPoolExecutor(
                max_workers=args.episode_concurrency,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_episode_worker,
                initargs=(args,),
            ) as executor, errors_path.open("a", encoding="utf-8", buffering=1) as errors:
                futures = {
                    executor.submit(
                        _run_episode_worker,
                        task_index,
                        game_file,
                        args.seed + task_index * 10_000_019,
                        args.mode,
                    ): task_index
                    for task_index, game_file in pending_tasks
                }
                for future in as_completed(futures):
                    task_index = futures[future]
                    try:
                        record = future.result()
                        if int(record["task_index"]) != task_index:
                            raise RuntimeError(f"Result task ID mismatch: {task_index}")
                        destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                        completed.add(task_index)
                        print(
                            f"ROLLOUT mode={args.mode} complete={len(completed)}/{len(games)} "
                            f"task={task_index} success={int(record['success'])} "
                            f"steps={record['num_steps']}",
                            flush=True,
                        )
                    except Exception:
                        details = traceback.format_exc()
                        errors.write(json.dumps({
                            "time_utc": datetime.now(timezone.utc).isoformat(),
                            "task_index": task_index,
                            "attempt": attempt,
                            "traceback": details,
                        }) + "\n")
                        print(f"EPISODE_ERROR task={task_index} attempt={attempt}\n{details}", flush=True)
    missing = sorted(set(range(len(games))) - completed)
    if missing:
        raise RuntimeError(f"{len(missing)} episodes failed after two attempts: {missing}; see {errors_path}")


if __name__ == "__main__":
    main()
