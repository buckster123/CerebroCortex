"""Tests for the LLM client abstraction."""

import pytest

from cerebro.utils.llm import (
    LLMClient,
    LLMResponse,
    OllamaProvider,
    AnthropicProvider,
)


class TestLLMResponse:
    def test_basic_response(self):
        resp = LLMResponse(text="Hello", provider="test", model="test-model")
        assert resp.text == "Hello"
        assert resp.provider == "test"
        assert resp.tokens_used == 0
        assert resp.was_fallback is False

    def test_fallback_response(self):
        resp = LLMResponse(text="Hi", provider="ollama", model="phi3:mini", was_fallback=True)
        assert resp.was_fallback is True


class TestLLMClientInit:
    def test_default_init(self):
        client = LLMClient()
        assert client.primary_name == "anthropic"
        assert client.fallback_name == "ollama"
        assert client.total_calls == 0

    def test_explicit_provider(self):
        client = LLMClient(provider="ollama", model="phi3:mini")
        assert client.primary_name == "ollama"

    def test_invalid_provider(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMClient(provider="invalid")

    def test_stats(self):
        client = LLMClient()
        s = client.stats()
        assert s["total_calls"] == 0
        assert s["primary"] == "anthropic"
        assert s["fallback"] == "ollama"


class TestLLMClientFailover:
    def test_both_fail_raises(self):
        """When both providers fail, RuntimeError is raised."""
        client = LLMClient()

        # Mock both providers to fail
        class FailProvider:
            def generate(self, *args, **kwargs):
                raise ConnectionError("No connection")

        client.primary = FailProvider()
        client.fallback = FailProvider()

        with pytest.raises(RuntimeError, match="All LLM providers failed"):
            client.generate("test prompt")

        assert client.total_calls == 1

    def test_fallback_on_primary_failure(self):
        """When primary fails, fallback is used."""
        client = LLMClient()

        class FailProvider:
            def generate(self, *args, **kwargs):
                raise ConnectionError("No connection")

        class SuccessProvider:
            def generate(self, prompt, system=None, max_tokens=1024, temperature=0.7):
                return LLMResponse(text="fallback response", provider="mock", model="mock")

        client.primary = FailProvider()
        client.fallback = SuccessProvider()

        resp = client.generate("test")
        assert resp.text == "fallback response"
        assert resp.was_fallback is True
        assert client.fallback_count == 1

    def test_primary_success(self):
        """When primary works, fallback is not used."""
        client = LLMClient()

        class SuccessProvider:
            def generate(self, prompt, system=None, max_tokens=1024, temperature=0.7):
                return LLMResponse(text="primary response", provider="mock", model="mock", tokens_used=50)

        client.primary = SuccessProvider()

        resp = client.generate("test")
        assert resp.text == "primary response"
        assert resp.was_fallback is False
        assert client.total_tokens == 50
        assert client.fallback_count == 0
