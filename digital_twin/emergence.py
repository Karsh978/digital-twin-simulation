"""Layer 5 — emergent behavior surface.

Agents freely choose among legal actions each tick, constrained by Layer 3,
informed by Layer 4. This module provides:

1. The anti-herding levers, stacked (§1.2) — none relied on alone:
   - persona diversity (Layer 1 synthesis)
   - temperature/sampling variance (per-persona temperature)
   - staggered decision timing (environment shuffles agent order each tick)
   - real friction/cost on imitation actions (ImitationFriction below)
   - path-dependent memory (Layer 4 reflective module)
   - the internal debate gate (Layer 4)
2. Herding measurement: dispersion metrics over the agent population, so you
   can verify the levers produce dispersion, not just different clustering.
3. The scripted-vs-emergent comparison harness: same seeded scenario run
   with scripted heuristics vs. full agent decision-making; the delta is the
   measured contribution of emergent behavior.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

from .environment import Action, ActionRecord


@dataclass
class ImitationFriction:
    """Real cost on imitation actions (§1.2 Layer 5).

    Copying another investor into a hot round is not free: each additional
    follower pays more (higher effective price / worse terms). `follow_cost`
    maps the number of existing followers to a cost multiplier.
    """

    base_cost: float = 0.05           # 5% penalty for the first follower
    per_follower_increment: float = 0.03

    def follow_cost(self, n_existing_followers: int) -> float:
        return self.base_cost + self.per_follower_increment * n_existing_followers


def herding_index(scores: Sequence[float]) -> float:
    """0 = maximal dispersion, 1 = perfect herding (all agents agree).

    Implemented as 1 - normalized variance of decision scores across the
    population. A tightly-correlated population herds at ANY size (§v2
    correction) — this metric, not the population cap, is how you detect it.
    """
    if len(scores) < 2:
        return 0.0
    var = statistics.pvariance(scores)
    uniform_var = 1.0 / 12.0  # variance of a uniform [0,1] population
    return max(0.0, 1.0 - var / uniform_var)


def decision_dispersion(records: Sequence[ActionRecord],
                        action: Action = Action.ISSUE_TERM_SHEET) -> Dict[str, float]:
    """Dispersion of evaluation scores across the whole agent population.

    Measures over ALL scored evaluations (term sheets AND passes) — scoring
    only issued sheets would truncate the distribution at the threshold and
    fake a herding signal."""
    rel = [r for r in records
           if r.action in (action, Action.PASS) and "score" in r.params]
    if not rel:
        return {"n_decisions": 0, "herding_index": 0.0, "timing_spread_weeks": 0.0}
    scores = [float(r.params.get("score", 0.5)) for r in rel]
    weeks = [r.week for r in rel]
    return {
        "n_decisions": float(len(rel)),
        "herding_index": herding_index(scores),
        "score_std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "timing_spread_weeks": float(max(weeks) - min(weeks)),
    }


@dataclass
class ComparisonResult:
    """Delta between scripted and emergent runs of the same seeded scenario."""

    scenario: str
    scripted_kpis: Dict[str, float]
    emergent_kpis: Dict[str, float]
    delta: Dict[str, float]

    def summary(self) -> str:
        lines = [f"scripted-vs-emergent comparison: {self.scenario}"]
        for k in self.emergent_kpis:
            s, e = self.scripted_kpis.get(k, 0.0), self.emergent_kpis[k]
            lines.append(f"  {k}: scripted={s:.4g} emergent={e:.4g} "
                         f"delta={self.delta.get(k, 0.0):+.4g}")
        return "\n".join(lines)


def run_comparison(
    scenario_name: str,
    build_and_run: Callable[[bool, int], Dict[str, float]],
    seed: int = 0,
) -> ComparisonResult:
    """Run the same seeded scenario twice (§1.2 Layer 5 / §2.5).

    `build_and_run(emergent, seed)` must construct the full scenario and
    return its KPIs — once with scripted heuristics (emergent=False), once
    with full agent decision-making (emergent=True). The delta is the
    measured contribution of the agent layer: signal or noise.
    """
    scripted = build_and_run(False, seed)
    emergent = build_and_run(True, seed)
    delta = {k: emergent.get(k, 0.0) - scripted.get(k, 0.0)
             for k in set(scripted) | set(emergent)}
    return ComparisonResult(scenario_name, scripted, emergent, delta)
