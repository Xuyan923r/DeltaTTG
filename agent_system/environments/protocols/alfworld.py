"""One source of truth for standalone ALFWorld rollouts and verl GRPO."""

from __future__ import annotations

from typing import Any


PROTOCOL_VERSION = 1
METER_PREFIX_ADAPTER = "alfworld_training_envelope_v1"
MAX_ENV_STEPS = 50
MAX_TOKENS_PER_TURN = 512
TEMPERATURE = 0.7
TOP_P = 0.95
TOP_K = -1

SYSTEM_PROMPT = """You control an ALFWorld household environment through the env_step tool.

On every turn, choose exactly one command from the current "Admissible actions" list.
Copy the command exactly, including object names and numbers. Plan silently, then respond
immediately with exactly one env_step tool call and no other text. Never invent a command
that is not listed. Only the environment can declare success or end the episode.
"""

TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "env_step",
        "description": "Execute exactly one currently admissible ALFWorld command.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Copy exactly one command from the current Admissible actions list.",
                }
            },
            "required": ["action"],
        },
        "strict": True,
    },
}


def format_observation(
    feedback: str,
    admissible_commands: list[str],
    *,
    action: str | None = None,
    steps: int = 0,
    success: bool = False,
    finished: bool = False,
) -> str:
    """Render identical reset/step observations for collection and training."""

    parts = [feedback.strip(), "", "Admissible actions:"]
    parts.extend(f"- {command}" for command in admissible_commands)
    if action is not None:
        parts.extend(["", f"Last action: {action}", f"Environment steps: {steps}"])
    if success:
        parts.extend(["", "TASK SUCCESS."])
    elif finished:
        parts.extend(["", "EPISODE ENDED WITHOUT SUCCESS."])
    return "\n".join(parts)


def format_meter_observation(
    feedback: str,
    admissible_commands: list[str],
    **kwargs: Any,
) -> str:
    """Canonicalize the ALFWorld state visible to the verl-agent actor.

    Verl-agent deliberately hides ``help`` from its action list.  Excluding it
    here preserves exactly the state content available to that actor while
    using the standalone meter dataset's observation layout.
    """

    visible_commands = [
        str(command) for command in admissible_commands if str(command) != "help"
    ]
    return format_observation(str(feedback), visible_commands, **kwargs)
