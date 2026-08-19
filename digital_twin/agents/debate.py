"""Internal mixture-of-models debate (§1.2 Layer 4, new in v2).

Consequential decisions route through an advocate / challenger / arbitrator
pass — potentially on DIFFERENT underlying models — to model genuine internal
ambivalence rather than one flat LLM opinion. Deliberately gated: it is not
affordable to run this per agent per tick, so only the consequential-decision
gate (defined per slice) calls it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .llm import LLMBackend


@dataclass
class DebateOutcome:
    decision: str                      # e.g. "proceed" | "decline"
    confidence: float                  # arbitrator's conviction, 0..1
    advocate_score: float
    challenger_score: float
    ambivalence: float                 # |advocate - challenger| — high = genuinely torn
    trace: Dict[str, str] = field(default_factory=dict)  # full transcript for traceability


def internal_debate(
    backend: LLMBackend,
    persona_summary: str,
    decision_description: str,
    context: str,
    models: Optional[Dict[str, str]] = None,
    temperature: float = 0.7,
    proceed_threshold: float = 0.5,
) -> DebateOutcome:
    """Run the three-role internal debate.

    `models` may map role -> model name, e.g.
        {"advocate": "gpt-4o", "challenger": "claude-sonnet", "arbitrator": "gpt-4o-mini"}
    In mock mode, distinct model names still produce genuinely distinct scores.
    """
    models = models or {}

    def role_model(role: str) -> Optional[str]:
        return models.get(role)

    advocate_prompt = (
        f"You are the internal ADVOCATE for this persona:\n{persona_summary}\n\n"
        f"Decision under consideration: {decision_description}\nContext:\n{context}\n\n"
        "Argue the strongest case FOR proceeding. Score how compelling the case is."
    )
    challenger_prompt = (
        f"You are the internal CHALLENGER for this persona:\n{persona_summary}\n\n"
        f"Decision under consideration: {decision_description}\nContext:\n{context}\n\n"
        "Argue the strongest case AGAINST proceeding — risks, conflicts with thesis, "
        "better uses of capital/time. Score how compelling the objection is."
    )

    adv = backend.score("You argue in favor of the decision.",
                        advocate_prompt, temperature, role_model("advocate"))
    chal = backend.score("You argue against the decision.",
                         challenger_prompt, temperature, role_model("challenger"))

    arbitrator_prompt = (
        f"You are the internal ARBITRATOR for this persona:\n{persona_summary}\n\n"
        f"Decision: {decision_description}\nContext:\n{context}\n\n"
        f"The advocate's case scored {adv:.2f}/1.00; the challenger's objection "
        f"scored {chal:.2f}/1.00. Weigh both, accounting for this persona's risk "
        "posture and thesis. Score the final conviction to PROCEED."
    )
    arb = backend.score("You weigh both sides and decide.",
                      arbitrator_prompt, temperature * 0.7, role_model("arbitrator"))

    return DebateOutcome(
        decision="proceed" if arb >= proceed_threshold else "decline",
        confidence=arb,
        advocate_score=adv,
        challenger_score=chal,
        ambivalence=abs(adv - chal),
        trace={"advocate": advocate_prompt, "challenger": challenger_prompt,
               "arbitrator": arbitrator_prompt},
    )
