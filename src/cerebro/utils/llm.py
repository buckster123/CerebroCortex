"""LLM client abstraction for CerebroCortex.

Supports multiple providers with automatic failover:
- Anthropic (Claude API) - primary, highest quality
- Ollama (local) - fallback, no cost/network needed

Usage:
    client = LLMClient()
    response = client.generate("Extract patterns from these memories...")

    # Or with explicit provider:
    client = LLMClient(provider="ollama", model="phi3:mini")
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol

import requests

from cerebro.config import (
    LLM_FALLBACK_MODEL,
    LLM_FALLBACK_PROVIDER,
    LLM_MAX_TOKENS,
    LLM_PRIMARY_MODEL,
    LLM_PRIMARY_PROVIDER,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
)

logger = logging.getLogger("cerebro-llm")


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    text: str
    provider: str
    model: str
    tokens_used: int = 0
    was_fallback: bool = False


class LLMProvider(Protocol):
    """Protocol for LLM provider backends."""

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
    ) -> LLMResponse: ...


# =============================================================================
# Anthropic provider
# =============================================================================

class AnthropicProvider:
    """Anthropic Claude API provider."""

    def __init__(self, model: str = LLM_PRIMARY_MODEL):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except ImportError:
                raise RuntimeError("anthropic package not installed: pip install anthropic")
            except Exception as e:
                raise RuntimeError(f"Anthropic client init failed: {e}")
        return self._client

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
    ) -> LLMResponse:
        client = self._get_client()

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system

        response = client.messages.create(**kwargs)

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        tokens = 0
        if hasattr(response, "usage"):
            tokens = getattr(response.usage, "input_tokens", 0) + getattr(response.usage, "output_tokens", 0)

        return LLMResponse(
            text=text,
            provider="anthropic",
            model=self.model,
            tokens_used=tokens,
        )


# =============================================================================
# Ollama provider
# =============================================================================

class OllamaProvider:
    """Ollama local LLM provider (HTTP API)."""

    def __init__(
        self,
        model: str = LLM_FALLBACK_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
    ) -> LLMResponse:
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system

        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        return LLMResponse(
            text=data.get("response", ""),
            provider="ollama",
            model=self.model,
            tokens_used=data.get("eval_count", 0),
        )


# =============================================================================
# Unified LLM client with auto-failover
# =============================================================================

class LLMClient:
    """Unified LLM client with primary + fallback provider.

    Auto-failover: if primary fails, transparently retries with fallback.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        fallback_provider: Optional[str] = None,
        fallback_model: Optional[str] = None,
    ):
        prov = provider or LLM_PRIMARY_PROVIDER
        mod = model or LLM_PRIMARY_MODEL

        self.primary = self._make_provider(prov, mod)
        self.primary_name = prov

        fb_prov = fallback_provider or LLM_FALLBACK_PROVIDER
        fb_mod = fallback_model or LLM_FALLBACK_MODEL
        self.fallback = self._make_provider(fb_prov, fb_mod)
        self.fallback_name = fb_prov

        self.total_calls = 0
        self.total_tokens = 0
        self.fallback_count = 0

    @staticmethod
    def _make_provider(provider: str, model: str) -> LLMProvider:
        if provider == "anthropic":
            return AnthropicProvider(model=model)
        elif provider == "ollama":
            return OllamaProvider(model=model)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
    ) -> LLMResponse:
        """Generate text with auto-failover."""
        self.total_calls += 1

        # Try primary
        try:
            resp = self.primary.generate(prompt, system, max_tokens, temperature)
            self.total_tokens += resp.tokens_used
            return resp
        except Exception as e:
            logger.warning(f"Primary LLM ({self.primary_name}) failed: {e}")

        # Fallback
        try:
            logger.info(f"Falling back to {self.fallback_name}")
            resp = self.fallback.generate(prompt, system, max_tokens, temperature)
            resp.was_fallback = True
            self.total_tokens += resp.tokens_used
            self.fallback_count += 1
            return resp
        except Exception as e:
            logger.error(f"Fallback LLM ({self.fallback_name}) also failed: {e}")
            raise RuntimeError(
                f"All LLM providers failed. Primary: {self.primary_name}, Fallback: {self.fallback_name}"
            ) from e

    def stats(self) -> dict:
        """LLM usage statistics."""
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "fallback_count": self.fallback_count,
            "primary": self.primary_name,
            "fallback": self.fallback_name,
        }
