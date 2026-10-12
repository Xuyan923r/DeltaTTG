"""Strict generation metrics for continuation-meter SFT models."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ANSWER_PATTERN = re.compile(
    r'<answer>\{"can_complete":(true|false),"remaining_action_tokens":'
    r'(null|[0-9]+)\}</answer>'
)


@dataclass(frozen=True)
class ParsedAnswer:
    can_complete: bool
    remaining_action_tokens: int | None

    @property
    def success(self) -> bool:
        return self.can_complete



def parse_strict_answer(text: str) -> ParsedAnswer | None:
    """Parse only the exact canonical target format, with no surrounding text."""
    match = ANSWER_PATTERN.fullmatch(text)
    if match is None:
        return None
    success = match.group(1) == "true"
    token_text = match.group(2)
    if success and token_text == "null":
        return None
    if not success and token_text != "null":
        return None
    return ParsedAnswer(
        can_complete=success,
        remaining_action_tokens=int(token_text) if success else None,
    )


def load_eval_records(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record.get("messages")
            metadata = record.get("metadata")
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: metadata is not valid JSON"
                    ) from exc
                record["metadata"] = metadata
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{path}:{line_number}: missing messages")
            if messages[-1].get("role") != "assistant":
                raise ValueError(f"{path}:{line_number}: final message is not assistant")
            if not isinstance(metadata, dict):
                raise ValueError(f"{path}:{line_number}: missing metadata")
            target = parse_strict_answer(str(messages[-1].get("content", "")))
            if target is None:
                raise ValueError(f"{path}:{line_number}: invalid canonical target")
            records.append(record)
    return records


def render_eval_prompt(tokenizer, record: dict[str, Any]) -> str:
    kwargs = {
        "tools": record.get("tools"),
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if "enable_thinking" in record:
        kwargs["enable_thinking"] = bool(record["enable_thinking"])
    return tokenizer.apply_chat_template(record["messages"][:-1], **kwargs)


def score_estimates(rows: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    total = 0
    format_correct = 0
    success_correct = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    token_abs_error_sum = 0
    token_squared_error_sum = 0
    token_abs_error_count = 0

    for row in rows:
        total += 1
        target = ParsedAnswer(
            can_complete=bool(row["target_success"]),
            remaining_action_tokens=row.get(
                "target_remaining_action_tokens",
                row.get("target_remaining_tokens_if_success"),
            ),
        )
        estimate = parse_strict_answer(str(row["estimate"]))
        if estimate is None:
            continue
        format_correct += 1
        if estimate.success == target.success:
            success_correct += 1
        if estimate.success and target.success:
            true_positive += 1
            token_error = abs(
                int(estimate.remaining_action_tokens)
                - int(target.remaining_action_tokens)
            )
            token_abs_error_sum += token_error
            token_squared_error_sum += token_error**2
            token_abs_error_count += 1
        elif not estimate.success and not target.success:
            true_negative += 1
        elif estimate.success:
            false_positive += 1
        else:
            false_negative += 1

    token_mae = (
        token_abs_error_sum / token_abs_error_count
        if token_abs_error_count
        else 0.0
    )
    token_rmse = (
        (token_squared_error_sum / token_abs_error_count) ** 0.5
        if token_abs_error_count
        else 0.0
    )
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "num_samples": total,
        "format_correct": format_correct,
        "format_accuracy": format_correct / total if total else 0.0,
        "success_correct": success_correct,
        "success_accuracy": success_correct / total if total else 0.0,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "successful_token_abs_error_sum": token_abs_error_sum,
        "successful_token_squared_error_sum": token_squared_error_sum,
        "successful_token_abs_error_count": token_abs_error_count,
        "successful_token_abs_error_mae": token_mae,
        "successful_token_error_rmse": token_rmse,
        "success_precision": precision,
        "success_recall": recall,
        "success_specificity": specificity,
        "success_f1": f1,
        "success_balanced_accuracy": (recall + specificity) / 2,
    }


def estimate_row(
    record: dict[str, Any],
    estimate: str,
) -> dict[str, Any]:
    target = parse_strict_answer(record["messages"][-1]["content"])
    assert target is not None
    parsed = parse_strict_answer(estimate)
    return {
        "sample_id": record["metadata"]["sample_id"],
        "task_id": record["metadata"]["task_id"],
        "estimate": estimate,
        "format_correct": parsed is not None,
        "estimated_success": parsed.success if parsed is not None else None,
        "estimated_remaining_action_tokens": (
            parsed.remaining_action_tokens if parsed is not None else None
        ),
        "target_success": target.success,
        "target_remaining_action_tokens": target.remaining_action_tokens,
        "success_correct": (
            parsed is not None and parsed.success == target.success
        ),
        "successful_token_abs_error": (
            abs(
                int(parsed.remaining_action_tokens)
                - int(target.remaining_action_tokens)
            )
            if parsed is not None and parsed.success and target.success
            else None
        ),
    }
