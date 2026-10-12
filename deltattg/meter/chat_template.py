"""Chat-template rendering shared by Qwen, Llama, and generic tokenizers."""

from __future__ import annotations

from typing import Any


def apply_chat_template_compat(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tokenize: bool,
    add_generation_prompt: bool = True,
    enable_thinking: bool | None = None,
    tools: list[dict[str, Any]] | None = None,
):
    """Render without assuming Qwen-only template variables exist.

    Modern Transformers forwards extra variables to Jinja, so Llama safely
    ignores ``enable_thinking``.  Some older/custom tokenizers reject that
    keyword at the Python boundary; for those, retry without only that optional
    knob.  Required arguments and message content are never rewritten.
    """

    kwargs: dict[str, Any] = {
        "tokenize": tokenize,
        "add_generation_prompt": add_generation_prompt,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    if tools is not None:
        kwargs["tools"] = tools
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as error:
        message = str(error)
        if "enable_thinking" not in kwargs or not (
            "unexpected keyword" in message or "enable_thinking" in message
        ):
            raise
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)
