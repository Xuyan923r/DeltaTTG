"""Strict parser for continuation meter generations."""

from __future__ import annotations

import math
import re

from deltattg.meter.schema import parse_strict_answer

from deltattg.types import MeterResult


_NUMBER = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
_VALUE_ANSWER = re.compile(
    rf'<answer>\s*\{{\s*"value"\s*:\s*({_NUMBER})\s*\}}\s*</answer>'
)


def parse_value_output(raw_output: str, *, retried: bool = False) -> MeterResult:
    """Accept a tagged value object or a bare finite JSON number."""
    text = raw_output.strip()
    match = _VALUE_ANSWER.fullmatch(text)
    number = match.group(1) if match else text
    value = float(number) if re.fullmatch(_NUMBER, number) else None
    valid = value is not None and math.isfinite(value)
    return MeterResult(
        # This mode has no binary feasibility decision; consumers use value.
        can_complete=False,
        remaining_action_tokens=None,
        valid=valid,
        raw_output=raw_output,
        error=None if valid else "invalid_value_answer",
        retried=retried,
        value=value if valid else None,
    )


def parse_meter_output(raw_output: str, *, retried: bool = False) -> MeterResult:
    parsed = parse_strict_answer(raw_output.strip())
    if parsed is None:
        return MeterResult(
            can_complete=False,
            remaining_action_tokens=None,
            valid=False,
            raw_output=raw_output,
            error="invalid_canonical_answer",
            retried=retried,
        )
    if parsed.can_complete and (
        parsed.remaining_action_tokens is None
        or parsed.remaining_action_tokens < 0
    ):
        return MeterResult(
            can_complete=False,
            remaining_action_tokens=None,
            valid=False,
            raw_output=raw_output,
            error="invalid_remaining_action_tokens",
            retried=retried,
        )
    return MeterResult(
        can_complete=parsed.can_complete,
        remaining_action_tokens=parsed.remaining_action_tokens,
        valid=True,
        raw_output=raw_output,
        retried=retried,
    )
