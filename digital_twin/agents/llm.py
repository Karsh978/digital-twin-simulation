"""Pluggable LLM backend with OpenRouter integration and deterministic mock fallback.

- MockLLM: deterministic, persona-driven heuristic scorer for test environments.
- OpenRouterLLM: OpenRouter-compatible chat API, supporting internal mixture-of-models across providers.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional


class LLMBackend:
    name = "base"

    def complete(self, system: str, prompt: str, temperature: float = 0.7,
                 model: Optional[str] = None) -> str:
        raise NotImplementedError

    def score(self, system: str, prompt: str, temperature: float = 0.7,
              model: Optional[str] = None) -> float:
        """Ask for a JSON {"score": 0..1} and parse it."""
        raw = self.complete(
            system + '\nRespond with ONLY a JSON object: {"score": <float 0..1>}.',
            prompt, temperature, model)
        try:
            return max(0.0, min(1.0, float(json.loads(raw)["score"])))
        except (ValueError, KeyError, json.JSONDecodeError):
            return 0.5


class MockLLM(LLMBackend):
    """Deterministic stand-in for testing without API keys."""

    name = "mock"

    def complete(self, system: str, prompt: str, temperature: float = 0.7,
                 model: Optional[str] = None) -> str:
        return json.dumps({"score": self.score(system, prompt, temperature, model)})

    def score(self, system: str, prompt: str, temperature: float = 0.7,
              model: Optional[str] = None) -> float:
        h = hashlib.sha256(f"{model}|{system}|{prompt}".encode()).digest()
        base = int.from_bytes(h[:4], "big") / 0xFFFFFFFF
        return max(0.0, min(1.0, 0.5 + (base - 0.5) * (0.4 + temperature)))


class OpenRouterLLM(LLMBackend):
    """OpenRouter API Backend replacing direct OpenAI endpoints."""

    name = "openrouter"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 default_model: str = "openai/gpt-4o-mini") -> None:
        from openai import OpenAI
        
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        
        if not self.api_key:
            # Fallback to empty key string if running without keys (Mock mode will handle fallback)
            self.api_key = "missing_key"

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers={
                "HTTP-Referer": "https://failsafe.ai",
                "X-Title": "Failsafe Digital Twin Engine",
            }
        )
        self.default_model = default_model

    def complete(self, system: str, prompt: str, temperature: float = 0.7,
                 model: Optional[str] = None) -> str:
        resp = self.client.chat.completions.create(
            model=model or self.default_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


# Backward Compatibility Alias so legacy agent imports do not break
OpenAILLM = OpenRouterLLM


_default: Optional[LLMBackend] = None


def get_default_backend() -> LLMBackend:
    """Returns OpenRouter backend when key is present, else MockLLM."""
    global _default
    if _default is None:
        if os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"):
            try:
                _default = OpenRouterLLM()
            except Exception:
                _default = MockLLM()
        else:
            _default = MockLLM()
    return _default