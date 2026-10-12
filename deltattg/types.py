"""Serializable schemas shared by rollout workers and the PPO driver."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AgentStep:
    step_idx: int
    prefix_messages: list[dict[str, Any]]
    assistant_message: dict[str, Any]
    assistant_output_token_ids: list[int]
    tool_or_environment_messages: list[dict[str, Any]]
    flattened_token_start: int
    flattened_token_end: int
    flattened_trainable_token_positions: list[int] | None = None

    def token_positions(self) -> list[int]:
        if self.flattened_trainable_token_positions is not None:
            return list(self.flattened_trainable_token_positions)
        return list(range(self.flattened_token_start, self.flattened_token_end))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AgentStep":
        return cls(**value)


@dataclass
class TrajectoryRecord:
    prompt_id: str
    task_id: str
    trajectory_id: str
    group_id: str
    messages: list[dict[str, Any]]
    steps: list[AgentStep]
    outcome_reward: float
    trajectory_success: bool
    termination_reason: str
    flattened_response_token_ids: list[int]
    flattened_response_trainable_mask: list[bool]
    max_assistant_generated_tokens: int
    tools: list[dict[str, Any]] = field(default_factory=list)
    original_agent_system: str = ""
    original_task: str = ""
    initial_observation: str = ""
    reference_model: str = ""
    decoding_config: dict[str, Any] = field(default_factory=dict)
    sampling_seed: int | None = None
    credit_fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrajectoryRecord":
        copied = dict(value)
        copied["steps"] = [
            item if isinstance(item, AgentStep) else AgentStep.from_dict(item)
            for item in copied["steps"]
        ]
        return cls(**copied)


@dataclass(frozen=True)
class MeterRequest:
    request_id: str
    trajectory_id: str
    step_idx: int
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    serialized_prompt: str | None = None
    input_ids: list[int] | None = None


@dataclass(frozen=True)
class MeterResult:
    can_complete: bool
    remaining_action_tokens: int | None
    valid: bool
    raw_output: str
    error: str | None = None
    retried: bool = False
    completion_log_odds: float | None = None
    completion_true_logprob: float | None = None
    completion_false_logprob: float | None = None
    # Outcome-only meters predict the final score, independently of TTG fields.
    value: float | None = None

    @property
    def success(self) -> bool:
        """Compatibility alias for existing credit metrics."""
        return self.can_complete
