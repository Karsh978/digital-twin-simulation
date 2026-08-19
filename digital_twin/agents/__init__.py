"""Layer 4 — agent cognitive layer: two-speed architecture, consequential-
decision gate, internal multi-model debate, pluggable LLM backend."""

from .llm import LLMBackend, MockLLM, OpenAILLM, get_default_backend
from .debate import DebateOutcome, internal_debate
from .base import Agent, AgentMemory
from .vc_agents import FounderAgent, LPAgent, VCPartnerAgent

__all__ = [
    "LLMBackend", "MockLLM", "OpenAILLM", "get_default_backend",
    "DebateOutcome", "internal_debate", "Agent", "AgentMemory",
    "FounderAgent", "VCPartnerAgent", "LPAgent",
]
