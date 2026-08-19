"""Layer 2 — World & Environment Server (Multi-Industry Generalized).

Simulation state, entity references, async event-bus hookup, Failsafe Master Agent API
data ingestion layer, and a configurable time engine.
"""

from __future__ import annotations

import json
import logging
import os
import random
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from .event_bus import EventBus, ShockEvent

log = logging.getLogger(__name__)


class Action(str, Enum):
    """Fallback Base Action space for VC slice. Slices register custom action strings dynamically."""
    PITCH = "pitch"
    TAKE_MEETING = "take_meeting"
    ISSUE_TERM_SHEET = "issue_term_sheet"
    NEGOTIATE_TERMS = "negotiate_terms"
    PASS = "pass_"
    CO_INVEST = "co_invest"
    INTRO = "intro_founder_to_investor"
    PUBLISH_UPDATE = "publish_update"
    HIRE = "hire"
    RUN_OUT_OF_RUNWAY = "run_out_of_runway"


@dataclass
class ActionRecord:
    week: int
    agent_id: str
    action: Union[Action, str]
    target: Optional[str]
    params: Dict[str, Any]
    outcome: str = "submitted"
    debate_trace: Optional[dict] = None


@dataclass
class WorldState:
    """Generalized Industry World State."""
    week: int = 0
    industry_type: str = "generic"
    market_heat: float = 0.5
    rate_environment: float = 0.5
    failsafe_telemetry: Dict[str, Any] = field(default_factory=dict)
    entities: Dict[str, dict] = field(default_factory=dict)
    open_transactions: Dict[str, dict] = field(default_factory=dict)
    shock_log: List[dict] = field(default_factory=list)


class EnvironmentServer:
    """Generalized Environment Server ingesting Failsafe Master Agent API data."""

    def __init__(self, slice_name: str, industry_type: str = "vc_finance", 
                 bus: Optional[EventBus] = None, seed: int = 0, weeks_per_year: int = 52) -> None:
        self.slice_name = slice_name
        self.industry_type = industry_type
        self.bus = bus or EventBus()
        self.state = WorldState(industry_type=industry_type)
        self.rng = random.Random(seed)
        self.weeks_per_year = weeks_per_year
        self.action_log: List[ActionRecord] = []
        self._agents: Dict[str, Any] = {}
        self._resolvers: Dict[str, Callable[[ActionRecord], str]] = {}
        self.failsafe_api_url = os.environ.get("FAILSAFE_MASTER_AGENT_URL", "https://api.failsafe.ai/v1/master-agent")
        self.failsafe_api_key = os.environ.get("FAILSAFE_API_KEY", "")

        self.bus.subscribe(slice_name, self._on_shock)
        self.sync_failsafe_data_layer()

    # -- Failsafe Data Layer Ingestion --------------------------------------
    def sync_failsafe_data_layer(self) -> None:
        """Pull live context and parameters from Failsafe Master Agent API instead of 100s of manual APIs."""
        if not self.failsafe_api_key:
            log.info("FAILSAFE_API_KEY not found. Operating with local default state.")
            return

        try:
            req = urllib.request.Request(
                f"{self.failsafe_api_url}/telemetry?industry={self.industry_type}",
                headers={
                    "Authorization": f"Bearer {self.failsafe_api_key}",
                    "Content-Type": "application/json"
                }
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    self.state.failsafe_telemetry = data.get("telemetry", {})
                    self.state.market_heat = data.get("market_heat", self.state.market_heat)
                    log.info(f"Successfully synced Layer 1 data from Failsafe API for {self.industry_type}")
        except Exception as e:
            log.warning(f"Failsafe Master Agent API sync skipped/failed: {e}")

    # -- Registration ------------------------------------------------------
    def register_agent(self, agent: Any) -> None:
        self._agents[agent.agent_id] = agent
        agent.bind_environment(self)

    def register_resolver(self, action: Union[Action, str], fn: Callable[[ActionRecord], str]) -> None:
        act_key = action.value if isinstance(action, Action) else action
        self._resolvers[act_key] = fn

    # -- Shocks -------------------------------------------------------------
    def _on_shock(self, ev: ShockEvent) -> None:
        self.state.shock_log.append({
            "week": self.state.week, "type": ev.shock_type.value,
            "magnitude": ev.magnitude, "origin": ev.origin_slice,
            "confidence": ev.confidence.value, "description": ev.description,
        })
        if ev.shock_type.value == "rate_change":
            self.state.rate_environment = min(1.0, max(0.0, self.state.rate_environment + 0.5 * ev.magnitude))
            self.state.market_heat = min(1.0, max(0.0, self.state.market_heat - 0.3 * ev.magnitude))
        elif ev.shock_type.value in ("demand_shift", "liquidity_shock", "major_failure"):
            self.state.market_heat = min(1.0, max(0.0, self.state.market_heat + 0.4 * ev.magnitude))

    # -- Main Loop -----------------------------------------------------------
    def tick(self, weeks: int = 1) -> None:
        for _ in range(weeks):
            self.state.week += 1
            self.bus.dispatch_due(self.state.week)
            
            order = list(self._agents.values())
            self.rng.shuffle(order)
            for agent in order:
                perception = self.perceive(agent.agent_id)
                for rec in agent.act(self.state.week, perception):
                    self.resolve(rec)
            
            self._advance_world()

    def run(self, weeks: int) -> None:
        self.tick(weeks)

    def _advance_world(self) -> None:
        if hasattr(self, "world_dynamics") and self.world_dynamics:
            self.world_dynamics(self.state)

    world_dynamics: Optional[Callable[[WorldState], None]] = None

    # -- Perception / Action ------------------------------------------------
    def perceive(self, agent_id: str) -> dict:
        return {
            "week": self.state.week,
            "industry": self.state.industry_type,
            "market_heat": self.state.market_heat,
            "rate_environment": self.state.rate_environment,
            "failsafe_telemetry": self.state.failsafe_telemetry,
            "recent_shocks": self.state.shock_log[-5:],
            "open_transactions": dict(self.state.open_transactions),
        }

    def submit(self, agent_id: str, action: Union[Action, str], target: Optional[str] = None,
               debate_trace: Optional[dict] = None, **params) -> ActionRecord:
        rec = ActionRecord(week=self.state.week, agent_id=agent_id, action=action,
                           target=target, params=params, debate_trace=debate_trace)
        self.action_log.append(rec)
        return rec

    def resolve(self, rec: ActionRecord) -> str:
        act_key = rec.action.value if isinstance(rec.action, Action) else rec.action
        fn = self._resolvers.get(act_key)
        rec.outcome = fn(rec) if fn else "no_resolver"
        return rec.outcome

    # -- Traceability Export --------------------------------------------------
    def export_log(self, path: str) -> None:
        with open(path, "w") as f:
            for rec in self.action_log:
                act_str = rec.action.value if isinstance(rec.action, Action) else str(rec.action)
                f.write(json.dumps({
                    "week": rec.week, "agent": rec.agent_id,
                    "action": act_str, "target": rec.target,
                    "params": rec.params, "outcome": rec.outcome,
                    "debate": rec.debate_trace,
                }) + "\n")