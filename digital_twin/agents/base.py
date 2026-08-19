"""Two-speed agent base (§1.2 Layer 4).

- Fast/reactive module: handles routine ticks cheaply (heuristics or one
  light model call).
- Slow/reflective module: periodically updates strategy and memory from
  accumulated experience — this is what makes memory PATH-DEPENDENT, so
  agents diverge over time instead of resetting to a generic prior (§1.2
  Layer 5 anti-herding lever).
- Consequential-decision gate: defined per slice; routes flagged decisions
  through the internal debate before committing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

from ..environment import Action, ActionRecord
from ..personas import Persona
from .debate import DebateOutcome, internal_debate
from .llm import LLMBackend, get_default_backend

if TYPE_CHECKING:
    from ..environment import EnvironmentServer


@dataclass
class AgentMemory:
    """Path-dependent memory: experiences accumulate and bias future decisions."""

    experiences: List[dict] = field(default_factory=list)
    strategy: Dict[str, float] = field(default_factory=dict)   # reflective state
    relationships: Dict[str, float] = field(default_factory=dict)  # agent_id -> trust

    def record(self, week: int, kind: str, detail: dict) -> None:
        self.experiences.append({"week": week, "kind": kind, **detail})

    def nudge_relationship(self, other_id: str, delta: float) -> None:
        self.relationships[other_id] = max(-1.0, min(1.0,
            self.relationships.get(other_id, 0.0) + delta))


class Agent:
    """Base agent. Subclasses implement `fast_path` and `reflect`."""

    #: actions this agent type treats as consequential (slice-defined gate)
    consequential_actions: Set[Action] = set()

    def __init__(
        self,
        agent_id: str,
        persona: Persona,
        backend: Optional[LLMBackend] = None,
        reflect_every_weeks: int = 8,
        debate_models: Optional[Dict[str, str]] = None,
        seed: int = 0,
    ) -> None:
        self.agent_id = agent_id
        self.persona = persona
        self.backend = backend or get_default_backend()
        self.memory = AgentMemory()
        self.reflect_every_weeks = reflect_every_weeks
        self.debate_models = debate_models
        self.rng = random.Random(seed)
        self.env: Optional["EnvironmentServer"] = None

    def bind_environment(self, env: "EnvironmentServer") -> None:
        self.env = env

    # -- main entry ---------------------------------------------------------
    def act(self, week: int, perception: dict) -> List[ActionRecord]:
        records: List[ActionRecord] = []
        if week % self.reflect_every_weeks == 0:
            self.reflect(week, perception)
        for action, target, params in self.fast_path(week, perception):
            debate: Optional[DebateOutcome] = None
            if action in self.consequential_actions:
                debate = self._gate(action, target, params, perception)
                if debate.decision == "decline":
                    self.memory.record(week, "debate_declined",
                                       {"action": action.value, "target": target,
                                        "ambivalence": debate.ambivalence})
                    continue
            assert self.env is not None
            records.append(self.env.submit(self.agent_id, action, target,
                                         debate_trace=(
                                             {"decision": debate.decision,
                                              "confidence": debate.confidence,
                                              "ambivalence": debate.ambivalence}
                                             if debate else None),
                                         **params))
        return records

    # -- the consequential-decision gate --------------------------------------
    def _gate(self, action: Action, target: Optional[str], params: dict,
              perception: dict) -> DebateOutcome:
        context = (
            f"Week {perception['week']}. Market heat {perception['market_heat']:.2f}, "
            f"rate environment {perception['rate_environment']:.2f}. "
            f"Recent shocks: {perception['recent_shocks']}. "
            f"Action params: {params}. "
            f"Agent strategy state: {self.memory.strategy}."
        )
        return internal_debate(
            self.backend,
            persona_summary=self.persona_summary(),
            decision_description=f"{action.value} targeting {target}",
            context=context,
            models=self.debate_models,
            temperature=self.persona.temperature,
        )

    def persona_summary(self) -> str:
        t = self.persona.traits
        return (
            f"Synthetic persona {self.persona.persona_id} (archetype "
            f"'{self.persona.archetype}'). Traits: risk_tolerance="
            f"{t.get('risk_tolerance', 0.5):.2f}, thesis_conviction="
            f"{t.get('thesis_conviction', 0.5):.2f}, herd_sensitivity="
            f"{t.get('herd_sensitivity', 0.5):.2f}. "
            f"Thesis type: {self.persona.categorical.get('thesis_type', 'generalist')}."
        )

    # -- subclass hooks -------------------------------------------------------
    def fast_path(self, week: int, perception: dict) -> List[tuple]:
        """Routine decisions. Return [(action, target, params), ...]."""
        return []

    def reflect(self, week: int, perception: dict) -> None:
        """Slow module: update strategy from accumulated experience."""
        n_wins = sum(1 for e in self.memory.experiences if e.get("kind") == "success")
        n_losses = sum(1 for e in self.memory.experiences if e.get("kind") == "failure")
        total = max(1, n_wins + n_losses)
        # path dependence: recent outcomes shift strategy, never reset it
        self.memory.strategy["aggression"] = max(0.0, min(1.0,
            self.memory.strategy.get("aggression", 0.5)
            + 0.1 * (n_wins - n_losses) / total))
