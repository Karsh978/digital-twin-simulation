"""Agent types for the U.S. VC slice (§2.4): founders, VC partners, LPs.

- Founders: manage runway; when the Layer 3 runway model says they're
  approaching the market window, they pitch. Accepting a term sheet is
  consequential -> internal debate.
- VC partners: evaluate pitches on the fast path (thesis fit + persona
  traits + market heat); issuing a term sheet and leading vs. following a
  competitive round are consequential -> internal debate.
- LPs: lighter-weight (§2.4) — they mostly constrain fund behavior via
  capital pacing and pressure signals.
"""

from __future__ import annotations

from typing import List, Optional

from ..environment import Action
from ..rules.fund import Fund, Stage
from ..rules.runway import RunwayModel
from .base import Agent


class FounderAgent(Agent):
    consequential_actions = {Action.NEGOTIATE_TERMS}  # accepting/negotiating a term sheet

    def __init__(self, agent_id, persona, company_id: str,
                 runway: RunwayModel, raise_window_weeks: int = 26, **kw) -> None:
        super().__init__(agent_id, persona, **kw)
        self.company_id = company_id
        self.runway = runway
        self.raise_window_weeks = raise_window_weeks
        self.raising = False

    def fast_path(self, week: int, perception: dict) -> List[tuple]:
        out: List[tuple] = []
        if self.runway.is_out_of_runway():
            out.append((Action.RUN_OUT_OF_RUNWAY, self.company_id, {}))
            return out
        # go to market when runway enters the window; urgency rises as it shrinks
        if not self.raising and self.runway.needs_raise_within(self.raise_window_weeks):
            self.raising = True
            out.append((Action.PITCH, self.company_id,
                        {"urgency": 1.0 - self.runway.runway_weeks() / max(1, self.raise_window_weeks * 2)}))
        elif self.raising:
            out.append((Action.PITCH, self.company_id, {"urgency": 0.8}))
        # publish updates — cheap signal, feeds narrative momentum
        if self.rng.random() < 0.1:
            out.append((Action.PUBLISH_UPDATE, self.company_id, {}))
        return out

    def reflect(self, week: int, perception: dict) -> None:
        super().reflect(week, perception)
        rejections = sum(1 for e in self.memory.experiences
                         if e.get("kind") == "pitch_passed")
        if rejections > 3:
            # path-dependent: repeated rejection makes founders cut burn
            self.memory.strategy["cost_discipline"] = min(1.0,
                self.memory.strategy.get("cost_discipline", 0.3) + 0.1)


class VCPartnerAgent(Agent):
    consequential_actions = {Action.ISSUE_TERM_SHEET, Action.CO_INVEST}

    def __init__(self, agent_id, persona, fund: Fund, **kw) -> None:
        super().__init__(agent_id, persona, **kw)
        self.fund = fund

    def fast_path(self, week: int, perception: dict) -> List[tuple]:
        out: List[tuple] = []
        heat = perception["market_heat"]
        for company_id, round_info in perception["open_rounds"].items():
            meeting = (Action.TAKE_MEETING, company_id, {})
            out.append(meeting)
            score = self._evaluate(company_id, round_info, heat)
            threshold = 0.55 - 0.15 * self.persona.traits.get("risk_tolerance", 0.5)
            if score >= threshold:
                stage = Stage(round_info.get("stage", "seed"))
                check = self._size_check(round_info, stage)
                if check > 0:
                    # lead vs. follow: lead when conviction high & round uncompetitive
                    lead = score > 0.75 and not round_info.get("competitive", False)
                    out.append((Action.ISSUE_TERM_SHEET, company_id,
                                {"check": check, "lead": lead, "score": score}))
            else:
                out.append((Action.PASS, company_id, {"score": score}))
        return out

    def _evaluate(self, company_id: str, round_info: dict, heat: float) -> float:
        """Fast-path evaluation: thesis fit + persona traits + momentum.
        Herd sensitivity is a persona trait — FOMO is modeled, not assumed."""
        herd = self.persona.traits.get("herd_sensitivity", 0.5)
        conviction = self.persona.traits.get("thesis_conviction", 0.5)
        thesis_fit = 1.0 if round_info.get("sector") == \
            self.persona.categorical.get("thesis_type") else 0.4
        momentum = round_info.get("narrative_momentum", 0.5)
        score = (0.45 * thesis_fit
                 + 0.25 * conviction
                 + 0.30 * (herd * (0.5 * momentum + 0.5 * heat)
                           + (1 - herd) * round_info.get("traction", 0.5)))
        # temperature/sampling variance across the population (anti-herding lever)
        score += self.rng.gauss(0, 0.05 * self.persona.temperature)
        return max(0.0, min(1.0, score))

    def _size_check(self, round_info: dict, stage: Stage) -> float:
        band = self.fund.check_bands.get(stage)
        if not band:
            return 0.0
        target = 0.35 * round_info.get("round_size", band.max_check)
        check = band.clamp(target)
        return check if self.fund.can_write_initial(stage, check) else 0.0

    def reflect(self, week: int, perception: dict) -> None:
        super().reflect(week, perception)
        # slow layer: update thesis view from market conditions
        heat = perception["market_heat"]
        self.memory.strategy["market_view"] = (
            0.7 * self.memory.strategy.get("market_view", 0.5) + 0.3 * heat)


class LPAgent(Agent):
    """Lighter-weight (§2.4): constrains fund behavior rather than playing
    the deal game. Quarterly pacing decisions; pressure on dry powder."""

    consequential_actions = set()  # LP decisions rarely hit the debate gate in v1

    def __init__(self, agent_id, persona, fund: Fund, **kw) -> None:
        super().__init__(agent_id, persona, **kw)
        self.fund = fund

    def fast_path(self, week: int, perception: dict) -> List[tuple]:
        # Quarterly (13-week) pacing review: push the fund to deploy or slow down.
        if week % 13 != 0:
            return []
        heat = perception["market_heat"]
        pressure = self.persona.traits.get("risk_tolerance", 0.5) * (1.2 - heat)
        self.fund.reserve_ratio = min(0.7, max(0.3,
            self.fund.reserve_ratio + (0.05 if heat < 0.4 else -0.05)))
        self.memory.record(week, "pacing_review",
                           {"reserve_ratio": self.fund.reserve_ratio,
                            "pressure": pressure})
        return []
