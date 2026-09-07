"""Thin async LLM client on the OpenAI SDK.

Works with OpenAI directly and with Gemini via Google's OpenAI-compatible
endpoint. Configuration comes from environment variables (see resolve_config).

Parameter notes (verified against provider docs, Sep 2026):
- max_completion_tokens is used for both providers: GPT-5.x rejects max_tokens,
  and Gemini's compatibility layer accepts max_completion_tokens. On both
  providers the cap includes reasoning/thinking tokens, so keep it generous.
- temperature is never sent unless LLM_TEMPERATURE is set (GPT-5.x reasoning
  models reject non-default values).
- reasoning_effort is only sent when LLM_REASONING_EFFORT is set.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import cache
from typing import Any

import openai
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODELS = {
    "openai": "gpt-5.4-mini",
    "gemini": "gemini-3.8-flash",
}
PROVIDER_ALIASES = {"google": "gemini"}
NOT_CONFIGURED_MESSAGE = "LLM not configured: set OPENAI_API_KEY or GOOGLE_API_KEY"


class LLMError(RuntimeError):
    """An LLM request failed (network, provider error, empty answer)."""


class LLMNotConfiguredError(LLMError):
    """No usable provider / API key in the environment."""


class OutputCapError(LLMError):
    """The model hit max_completion_tokens before producing any text or tool call."""


def _env(name: str, default: str = "") -> str:
    """Environment value with whitespace stripped; empty string means unset."""
    return os.environ.get(name, default).strip()


# AgentBeats/Amber may deliver a secret under a role-prefixed name (the participant role on
# Pi-Bench is "agent"). Deterministic lookup order, no scanning of the whole environment.
_ALIAS_PREFIXES = ("", "AGENT_", "AMBER_CONFIG_", "AMBER_CONFIG_AGENT_")


def _env_any(name: str) -> str:
    for prefix in _ALIAS_PREFIXES:
        value = _env(prefix + name)
        if value:
            return value
    return ""


def log_env_diagnostic() -> None:
    """Log the NAMES of auth-relevant environment variables (never their values)."""
    markers = ("OPENAI", "GOOGLE", "GEMINI", "API_KEY", "AMBER", "SECRET", "LLM_", "PIBENCH")
    names = sorted(n for n in os.environ if any(m in n.upper() for m in markers))
    logger.info("auth/config-relevant env var names: %s", names)


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str
    base_url: str | None
    timeout_s: float = 120.0
    max_retries: int = 2
    max_output_tokens: int = 8192
    reasoning_effort: str | None = None
    temperature: float | None = None

    def describe(self) -> str:
        return f"{self.provider}:{self.model}"


def resolve_config() -> LLMConfig:
    """Pick provider, model and key from the environment.

    Raises LLMNotConfiguredError when no usable key is present.
    """
    openai_key = _env_any("OPENAI_API_KEY")
    gemini_key = _env_any("GOOGLE_API_KEY") or _env_any("GEMINI_API_KEY")

    provider = _env("LLM_PROVIDER").lower()
    provider = PROVIDER_ALIASES.get(provider, provider)
    if not provider:  # auto-select
        if openai_key:
            provider = "openai"
        elif gemini_key:
            provider = "gemini"
        else:
            raise LLMNotConfiguredError(NOT_CONFIGURED_MESSAGE)
    if provider not in DEFAULT_MODELS:
        raise LLMNotConfiguredError(
            f"LLM not configured: unknown LLM_PROVIDER={provider!r} "
            "(expected 'openai' or 'gemini')"
        )

    if provider == "openai":
        api_key = openai_key
        base_url = None  # the SDK honors OPENAI_BASE_URL on its own
        if not api_key:
            raise LLMNotConfiguredError(
                "LLM not configured: LLM_PROVIDER=openai but OPENAI_API_KEY is not set"
            )
    else:
        api_key = gemini_key
        base_url = _env("GEMINI_BASE_URL") or GEMINI_BASE_URL
        if not api_key:
            raise LLMNotConfiguredError(
                "LLM not configured: LLM_PROVIDER=gemini but GOOGLE_API_KEY "
                "(or GEMINI_API_KEY) is not set"
            )

    temperature = _env("LLM_TEMPERATURE")
    return LLMConfig(
        provider=provider,
        model=_env("LLM_MODEL") or DEFAULT_MODELS[provider],
        api_key=api_key,
        base_url=base_url,
        timeout_s=float(_env("LLM_TIMEOUT_S") or 120),
        max_retries=int(_env("LLM_MAX_RETRIES") or 2),
        max_output_tokens=int(_env("LLM_MAX_OUTPUT_TOKENS") or 8192),
        reasoning_effort=_env("LLM_REASONING_EFFORT") or None,
        temperature=float(temperature) if temperature else None,
    )


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON string exactly as returned by the model


@dataclass
class ChatResult:
    text: str
    finish_reason: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

    def assistant_message(self) -> dict[str, Any]:
        """This turn as a chat-format dict suitable for appending to history."""
        msg: dict[str, Any] = {"role": "assistant", "content": self.text}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in self.tool_calls
            ]
        return msg


class LLMClient:
    """Async chat client. One instance per process is enough (shared HTTP pool)."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_s,
            max_retries=config.max_retries,  # SDK retries 408/409/429/5xx + connection errors
        )

    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def model(self) -> str:
        return self.config.model

    def describe(self) -> str:
        return self.config.describe()

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        seed: int | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> ChatResult:
        """One chat completion. Raises LLMError on failure or a capped/blocked empty answer.

        `timeout` and `max_retries` override the client defaults for this call only (the HTTP
        pool is shared); `seed` is forwarded only when given.
        """
        cfg = self.config
        cap = max_tokens or cfg.max_output_tokens
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "max_completion_tokens": cap,
        }
        # None = use the configured default; "" = send no reasoning_effort at all.
        effort = cfg.reasoning_effort if reasoning_effort is None else (reasoning_effort or None)
        if effort:
            kwargs["reasoning_effort"] = effort
        if cfg.temperature is not None:
            kwargs["temperature"] = cfg.temperature
        if response_format:
            kwargs["response_format"] = response_format
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        if seed is not None:
            kwargs["seed"] = seed
        client = self._client
        overrides = {k: v for k, v in {"timeout": timeout, "max_retries": max_retries}.items() if v is not None}
        if overrides:
            client = client.with_options(**overrides)

        try:
            completion = await client.chat.completions.create(**kwargs)
        except openai.APIStatusError as e:
            raise LLMError(
                f"{self.describe()} request failed: HTTP {e.status_code}: {e.message}"
            ) from e
        except openai.OpenAIError as e:  # timeouts, connection errors, etc.
            raise LLMError(
                f"{self.describe()} request failed: {type(e).__name__}: {e}"
            ) from e

        if not completion.choices:
            raise LLMError(f"{self.describe()} returned no choices")
        choice = completion.choices[0]
        text = (choice.message.content or "").strip()
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
            for tc in (choice.message.tool_calls or [])
            if getattr(tc, "type", "function") == "function"
        ]

        usage: dict[str, int] = {}
        if completion.usage:
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }
            details = getattr(completion.usage, "completion_tokens_details", None)
            reasoning = getattr(details, "reasoning_tokens", None)
            if reasoning is not None:
                usage["reasoning_tokens"] = reasoning

        if not text and not tool_calls:
            if choice.finish_reason == "length":
                raise OutputCapError(
                    f"{self.describe()} hit the output cap ({cap} tokens, reasoning included) "
                    "before answering; raise LLM_MAX_OUTPUT_TOKENS or lower LLM_REASONING_EFFORT"
                )
            if choice.finish_reason == "content_filter":
                raise LLMError(f"{self.describe()} response blocked by provider content filter")

        logger.info("%s finish=%s usage=%s", self.describe(), choice.finish_reason, usage)
        return ChatResult(
            text=text, finish_reason=choice.finish_reason, tool_calls=tool_calls, usage=usage
        )


@cache
def get_llm() -> LLMClient:
    """Lazily built, process-wide client. Raises LLMNotConfiguredError if unusable.

    functools.cache does not cache exceptions, so an unconfigured process keeps
    reporting the error on every task instead of poisoning the cache.
    """
    config = resolve_config()
    logger.info(
        "LLM configured: %s (timeout=%ss retries=%s max_output_tokens=%s reasoning_effort=%s)",
        config.describe(), config.timeout_s, config.max_retries,
        config.max_output_tokens, config.reasoning_effort,
    )
    return LLMClient(config)


def describe_llm() -> str:
    """Provider:model string that never raises (for startup logs)."""
    try:
        return resolve_config().describe()
    except LLMNotConfiguredError:
        return "not configured"
