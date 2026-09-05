"""Keyless unit tests for LLM configuration resolution (no network)."""

import pytest

from llm import DEFAULT_MODELS, GEMINI_BASE_URL, LLMNotConfiguredError, resolve_config

ENV_VARS = [
    "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI_BASE_URL",
    "LLM_PROVIDER", "LLM_MODEL", "LLM_TEMPERATURE", "LLM_REASONING_EFFORT",
    "LLM_MAX_OUTPUT_TOKENS", "LLM_TIMEOUT_S", "LLM_MAX_RETRIES",
    # role-prefixed aliases that AgentBeats/Amber may use for the same secrets
    "AGENT_OPENAI_API_KEY", "AMBER_CONFIG_OPENAI_API_KEY", "AMBER_CONFIG_AGENT_OPENAI_API_KEY",
    "AGENT_GOOGLE_API_KEY", "AMBER_CONFIG_GOOGLE_API_KEY", "AMBER_CONFIG_AGENT_GOOGLE_API_KEY",
    "AGENT_GEMINI_API_KEY", "AMBER_CONFIG_GEMINI_API_KEY", "AMBER_CONFIG_AGENT_GEMINI_API_KEY",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_not_configured_without_keys():
    with pytest.raises(LLMNotConfiguredError, match="OPENAI_API_KEY or GOOGLE_API_KEY"):
        resolve_config()


def test_auto_selects_openai_when_both_keys_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "g-test")
    cfg = resolve_config()
    assert cfg.provider == "openai"
    assert cfg.model == DEFAULT_MODELS["openai"]
    assert cfg.base_url is None
    assert cfg.api_key == "sk-test"


def test_auto_selects_gemini_when_only_google_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "g-test")
    cfg = resolve_config()
    assert cfg.provider == "gemini"
    assert cfg.model == DEFAULT_MODELS["gemini"]
    assert cfg.base_url == GEMINI_BASE_URL


def test_gemini_api_key_alias(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-alias")
    cfg = resolve_config()
    assert cfg.provider == "gemini" and cfg.api_key == "g-alias"


def test_role_prefixed_key_aliases(monkeypatch):
    monkeypatch.setenv("AMBER_CONFIG_AGENT_OPENAI_API_KEY", "sk-prefixed")
    cfg = resolve_config()
    assert cfg.provider == "openai" and cfg.api_key == "sk-prefixed"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-plain")
    assert resolve_config().api_key == "sk-plain"  # the plain name wins over aliases


def test_empty_strings_count_as_unset(monkeypatch):
    # Amber fills omitted optional config with "" (see amber-manifest.json5).
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER", "")
    monkeypatch.setenv("LLM_MODEL", "")
    cfg = resolve_config()
    assert cfg.provider == "openai" and cfg.model == DEFAULT_MODELS["openai"]


def test_explicit_provider_with_missing_key_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(LLMNotConfiguredError, match="GOOGLE_API_KEY"):
        resolve_config()


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(LLMNotConfiguredError, match="unknown LLM_PROVIDER"):
        resolve_config()


def test_overrides(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "g-test")
    monkeypatch.setenv("LLM_PROVIDER", "google")  # alias
    monkeypatch.setenv("LLM_MODEL", "gemini-3.5-flash-lite")
    monkeypatch.setenv("LLM_REASONING_EFFORT", "low")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "512")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")
    cfg = resolve_config()
    assert cfg.provider == "gemini"
    assert cfg.model == "gemini-3.5-flash-lite"
    assert cfg.reasoning_effort == "low"
    assert cfg.max_output_tokens == 512
    assert cfg.temperature == 0.2


@pytest.mark.asyncio
async def test_chat_reasoning_effort_none_uses_config_and_empty_string_omits(monkeypatch):
    """None = configured default (LLM_REASONING_EFFORT); "" = send no reasoning_effort at all."""
    from types import SimpleNamespace

    from llm import LLMClient, LLMConfig

    seen = []

    async def create(**kwargs):
        seen.append(kwargs)
        message = SimpleNamespace(content="ok", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None)

    client = LLMClient(LLMConfig(provider="openai", model="m", api_key="k", base_url=None, reasoning_effort="medium"))
    monkeypatch.setattr(client._client.chat.completions, "create", create)
    await client.chat([{"role": "user", "content": "hi"}])
    await client.chat([{"role": "user", "content": "hi"}], reasoning_effort="")
    await client.chat([{"role": "user", "content": "hi"}], reasoning_effort="low", seed=7)
    assert seen[0]["reasoning_effort"] == "medium"
    assert "reasoning_effort" not in seen[1]
    assert seen[2]["reasoning_effort"] == "low" and seen[2]["seed"] == 7 and "seed" not in seen[0]
