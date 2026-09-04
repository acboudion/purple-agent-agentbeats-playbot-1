"""Keyless unit tests for LLM configuration resolution (no network)."""

import pytest

from llm import DEFAULT_MODELS, GEMINI_BASE_URL, LLMNotConfiguredError, resolve_config

ENV_VARS = [
    "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI_BASE_URL",
    "LLM_PROVIDER", "LLM_MODEL", "LLM_TEMPERATURE", "LLM_REASONING_EFFORT",
    "LLM_MAX_OUTPUT_TOKENS", "LLM_TIMEOUT_S", "LLM_MAX_RETRIES",
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
