"""Worked example (Part 2): U.S. seed-to-Series-A software slice.

Independently built and validated per §1.1 — a U.K. slice would be a
SEPARATE slice with its own sourcing and rules, not a "country" field here.

This module wires the six layers into a runnable slice:
  L1  knowledge graph + persona synthesis (mock adapters stand in for
      Crunchbase/EDGAR/GDELT; real adapters implement the same interface)
  L2  environment server, weekly ticks, action space, state logging
  L3  procedural rules: cap table, runway, fund mechanics
  L4  founders / VC partners / LPs with the consequential-decision gate
  L5  emergent behavior + scripted-vs-emergent comparison
  L6  KPI extraction + historical replay validation
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..agents import FounderAgent, LPAgent, VCPartnerAgent
from ..emergence import ImitationFriction, decision_dispersion, run_comparison, ComparisonResult
from ..environment import Action, ActionRecord, EnvironmentServer
from ..event_bus import EventBus
from ..knowledge_graph import KnowledgeGraph, LicenseStatus, SourceRecord
from ..personas import PersonaSynthesizer, SourceProfile, build_archetype, cluster_profiles
from ..rules.cap_table import CapTable, ShareClass, issue_round
from ..rules.fund import CheckSizeBand, Fund, Stage
from ..rules.runway import RunwayModel
from ..validation import KPITarget, validate_episode, ValidationReport

SECTORS = ["devtools", "fintech", "healthtech", "consumer", "ai_infra"]


# --------------------------------------------------------------------------
# Layer 1 — mock source adapters (real adapters implement this interface)
# --------------------------------------------------------------------------

def mock_source_records(n_companies: int, n_investors: int, seed: int = 0) -> List[SourceRecord]:
    """Stand-in for Crunchbase/EDGAR adapters. Licensing status is attached
    per record — the ingestion gate enforces it before anything enters the
    graph (§1.5)."""
    rng = random.Random(seed)
    records: List[SourceRecord] = []
    for i in range(n_companies):
        records.append(SourceRecord(
            source="mock_crunchbase", record_type="company",
            license_status=LicenseStatus.API_LICENSED,
            data={"name": f"Startup {i:03d}",
                  "sector": rng.choice(SECTORS),
                  "traction": rng.betavariate(2, 5),
                  "narrative_momentum": rng.betavariate(2, 4)}))
    fund_names = ["Meridian", "Cascade", "Harborlight", "Ironwood", "Bluebird",
                  "Summit", "Foxglove", "Northgate", "Ember", "Lattice",
                  "Wildcat", "Juniper", "Osprey", "Cobalt", "Redwood"]
    for i in range(n_investors):
        records.append(SourceRecord(
            source="mock_edgar_form_d", record_type="fund",
            license_status=LicenseStatus.FREE,
            data={"name": f"{fund_names[i % len(fund_names)]} Capital"}))
    return records


def mock_source_profiles(n: int, seed: int = 0) -> List[SourceProfile]:
    """Stand-in for the persona source pool (§1.4): bucketed features only,
    never raw text. Pool size comfortably above the minimum gate."""
    rng = random.Random(seed)
    profiles = []
    for _ in range(n):
        profiles.append(SourceProfile(
            features={
                "risk_tolerance": min(1, max(0, rng.gauss(0.5, 0.18))),
                "thesis_conviction": min(1, max(0, rng.gauss(0.55, 0.15))),
                "herd_sensitivity": min(1, max(0, rng.gauss(0.45, 0.2))),
            },
            categorical={"thesis_type": rng.choice(SECTORS)}))
    return profiles


# --------------------------------------------------------------------------
# Slice assembly
# --------------------------------------------------------------------------

@dataclass
class SliceConfig:
    n_companies: int = 40
    n_funds: int = 12
    persona_pool: int = 120          # source individuals per synthesis run (§1.4)
    min_pool: int = 24
    sim_weeks: int = 104             # two years of weekly ticks
    seed: int = 0
    emergent: bool = True            # False -> scripted heuristics (L5 comparison)


@dataclass
class SliceKPIs:
    """Slice-specific KPIs (§2.6): round size distribution, time-to-next-round,
    valuation step-up multiples, syndicate composition."""
    round_sizes: List[float] = field(default_factory=list)
    step_ups: List[float] = field(default_factory=list)
    syndicate_sizes: List[int] = field(default_factory=list)
    time_to_next_round_weeks: List[int] = field(default_factory=list)

    def as_scalars(self) -> Dict[str, float]:
        return {
            "mean_round_size_m": (statistics.fmean(self.round_sizes) / 1e6
                                  if self.round_sizes else 0.0),
            "mean_step_up": (statistics.fmean(self.step_ups)
                             if self.step_ups else 0.0),
            "mean_syndicate_size": (statistics.fmean(self.syndicate_sizes)
                                    if self.syndicate_sizes else 0.0),
            "mean_time_to_next_round_wks": (statistics.fmean(self.time_to_next_round_weeks)
                                            if self.time_to_next_round_weeks else 0.0),
        }


class USVentureCapitalSlice:
    name = "us_vc_seed_to_a"

    def __init__(self, config: Optional[SliceConfig] = None,
                 bus: Optional[EventBus] = None) -> None:
        self.config = config or SliceConfig()
        self.rng = random.Random(self.config.seed)
        self.bus = bus or EventBus()
        self.env = EnvironmentServer(self.name, bus=self.bus, seed=self.config.seed)
        self.graph = KnowledgeGraph()
        self.kpis = SliceKPIs()
        self.friction = ImitationFriction()
        self.companies: Dict[str, dict] = {}
        self.funds: Dict[str, Fund] = {}
        self.cap_tables: Dict[str, CapTable] = {}
        self.founders: List[FounderAgent] = []
        self.partners: List[VCPartnerAgent] = []
        self._last_round_week: Dict[str, int] = {}
        self._term_sheets: Dict[str, List[ActionRecord]] = {}
        self._build()

    # -- construction ------------------------------------------------------
    def _build(self) -> None:
        cfg = self.config
        # L1: ingest through the gate, build the graph
        self.graph.ingest(mock_source_records(cfg.n_companies, cfg.n_funds, cfg.seed))
        # L1: persona synthesis (statistical, pool-gated, outlier-filtered)
        profiles = mock_source_profiles(cfg.persona_pool, cfg.seed)
        clusters = cluster_profiles(profiles, n_archetypes=4, seed=cfg.seed)
        # merge undersized clusters into the largest: the pool-size gate (§1.4)
        # is a hard floor, never silently bypassed
        clusters.sort(key=len, reverse=True)
        while len(clusters) > 1 and len(clusters[-1]) < cfg.min_pool:
            clusters[0].extend(clusters.pop())
        archetypes = [build_archetype(f"arch-{i}", c) for i, c in enumerate(clusters)]
        synth = PersonaSynthesizer(min_pool=cfg.min_pool, seed=cfg.seed)
        founder_personas = synth.synthesize(
            max(archetypes, key=lambda a: a.size), cfg.n_companies)
        partner_personas = synth.synthesize(
            min(archetypes, key=lambda a: a.size), cfg.n_funds)

        # L2/L3: companies with cap tables + runway models
        for i, ent in enumerate(self.graph.of_type("company")):
            cash = self.rng.uniform(0.8e6, 2.5e6)
            ct = CapTable()
            ct.add_class(ShareClass("founders", 8_000_000))
            ct.add_class(ShareClass("option_pool", 2_000_000))
            self.cap_tables[ent.id] = ct
            self.companies[ent.id] = {
                "name": ent.name, "sector": ent.attrs.get("sector", "devtools"),
                "traction": ent.attrs.get("traction", 0.3),
                "narrative_momentum": ent.attrs.get("narrative_momentum", 0.3),
                "last_post_money": 0.0, "dead": False,
            }
            runway = RunwayModel(cash=cash, headcount=self.rng.randint(4, 12))
            founder = FounderAgent(f"founder-{i}", founder_personas[i], ent.id,
                                   runway, seed=cfg.seed + i)
            self.founders.append(founder)
            self.env.register_agent(founder)

        # L3: funds with check bands and reserves
        for i, ent in enumerate(self.graph.of_type("fund")):
            size = self.rng.uniform(30e6, 150e6)
            fund = Fund(name=ent.name, size=size, reserve_ratio=0.5,
                        check_bands={
                            Stage.SEED: CheckSizeBand(Stage.SEED, 0.5e6, 3e6),
                            Stage.SERIES_A: CheckSizeBand(Stage.SERIES_A, 3e6, 12e6),
                        })
            self.funds[ent.id] = fund
            partner = VCPartnerAgent(f"partner-{i}", partner_personas[i % len(partner_personas)],
                                     fund, seed=cfg.seed + 1000 + i)
            self.partners.append(partner)
            self.env.register_agent(partner)
            lp = LPAgent(f"lp-{i}", partner_personas[(i + 1) % len(partner_personas)],
                         fund, seed=cfg.seed + 2000 + i)
            self.env.register_agent(lp)

        # L2: resolvers apply Layer 3 rules — agents never touch state directly
        self.env.register_resolver(Action.PITCH, self._resolve_pitch)
        self.env.register_resolver(Action.ISSUE_TERM_SHEET, self._resolve_term_sheet)
        self.env.register_resolver(Action.PASS, self._resolve_pass)
        self.env.register_resolver(Action.RUN_OUT_OF_RUNWAY, self._resolve_death)
        self.env.register_resolver(Action.PUBLISH_UPDATE, self._resolve_update)
        self.env.world_dynamics = self._world_dynamics

    # -- resolvers (Layer 3 application) -------------------------------------
    def _resolve_pitch(self, rec: ActionRecord) -> str:
        company_id = rec.target
        if self.companies.get(company_id, {}).get("dead"):
            return "dead"
        c = self.companies[company_id]
        stage = "seed" if c["last_post_money"] == 0 else "series_a"
        round_size = (2.5e6 if stage == "seed" else 10e6) * (0.7 + 0.6 * c["traction"])
        self.env.state.open_rounds[company_id] = {
            "stage": stage, "round_size": round_size, "sector": c["sector"],
            "traction": c["traction"],
            "narrative_momentum": c["narrative_momentum"],
            "competitive": c["narrative_momentum"] > 0.6 and
                           self.env.state.market_heat > 0.6,
        }
        return "round_opened"

    def _resolve_term_sheet(self, rec: ActionRecord) -> str:
        company_id = rec.target
        if company_id not in self.env.state.open_rounds:
            return "stale"
        self._term_sheets.setdefault(company_id, []).append(rec)
        # close the round once enough sheets cover the round size
        info = self.env.state.open_rounds[company_id]
        sheets = self._term_sheets[company_id]
        committed = sum(s.params["check"] for s in sheets)
        if committed >= info["round_size"]:
            return self._close_round(company_id, sheets)
        return "sheet_outstanding"

    def _close_round(self, company_id: str, sheets: List[ActionRecord]) -> str:
        info = self.env.state.open_rounds.pop(company_id)
        c = self.companies[company_id]
        # friction on imitation: followers pay an effective premium (§1.2 L5)
        n_followers = sum(1 for s in sheets if not s.params.get("lead"))
        heat_uplift = 1.0 + 0.5 * self.env.state.market_heat * c["narrative_momentum"]
        friction_uplift = 1.0 + self.friction.follow_cost(n_followers)
        pre_money = info["round_size"] * 4 * heat_uplift * friction_uplift
        ct = self.cap_tables[company_id]
        # allocations are scaled to exactly fill the round (oversubscription is
        # allocated pro-rata, as in a real competitive round)
        total_committed = sum(s.params["check"] for s in sheets)
        scale = info["round_size"] / total_committed
        investors: Dict[str, float] = {}
        for s in sheets:  # aggregate — one partner may hold multiple sheets
            investors[s.agent_id] = investors.get(s.agent_id, 0.0) + s.params["check"] * scale
        issue_round(ct, info["stage"], pre_money, info["round_size"], investors,
                    pool_topup_pct=0.10)
        # fund mechanics: checks come out of the right budgets (L3 constraint)
        for s in sheets:
            fund = next((p.fund for p in self.partners if p.agent_id == s.agent_id), None)
            if fund:
                stage = Stage(info["stage"])
                check = s.params["check"] * scale
                if company_id in fund.portfolio:
                    fund.write_follow_on(company_id, check)
                else:
                    fund.write_initial(company_id, stage, check)
        # KPI capture (§2.6)
        self.kpis.round_sizes.append(info["round_size"])
        self.kpis.syndicate_sizes.append(len(sheets))
        if c["last_post_money"] > 0:
            self.kpis.step_ups.append((pre_money + info["round_size"]) / c["last_post_money"])
            self.kpis.time_to_next_round_weeks.append(
                self.env.state.week - self._last_round_week.get(company_id, 0))
        c["last_post_money"] = pre_money + info["round_size"]
        self._last_round_week[company_id] = self.env.state.week
        # founders stop raising; cash replenishes the runway model
        for f in self.founders:
            if f.company_id == company_id:
                f.raising = False
                f.runway.cash += info["round_size"]
                f.memory.record(self.env.state.week, "success",
                              {"round_size": info["round_size"]})
        del self._term_sheets[company_id]
        return "round_closed"

    def _resolve_pass(self, rec: ActionRecord) -> str:
        for f in self.founders:
            if f.company_id == rec.target:
                f.memory.record(self.env.state.week, "pitch_passed",
                                {"by": rec.agent_id})
        return "passed"

    def _resolve_death(self, rec: ActionRecord) -> str:
        self.companies[rec.target]["dead"] = True
        self.env.state.open_rounds.pop(rec.target, None)
        return "died"

    def _resolve_update(self, rec: ActionRecord) -> str:
        c = self.companies.get(rec.target)
        if c:
            c["narrative_momentum"] = min(1.0, c["narrative_momentum"] + 0.05)
        return "published"

    def _world_dynamics(self, state) -> None:
        """Burn advances even with no agents in the loop (§2.9 milestone)."""
        for f in self.founders:
            if not self.companies[f.company_id]["dead"]:
                f.runway.tick(1)
        # market heat mean-reverts slowly
        state.market_heat += (0.5 - state.market_heat) * 0.01

    # -- running -------------------------------------------------------------
    def run(self) -> Dict[str, float]:
        if not self.config.emergent:
            self._scripted_mode()
        self.env.run(self.config.sim_weeks)
        return self.kpis.as_scalars()

    def _scripted_mode(self) -> None:
        """Scripted heuristics for the L5 comparison: replace deliberation with
        fixed rules (all partners evaluate identically, no debate gate)."""
        for p in self.partners:
            p.consequential_actions = set()          # no internal debate
            p.persona.temperature = 0.0              # no sampling variance
            p._evaluate = lambda cid, ri, heat: (  # identical scoring for all
                0.45 * (1.0 if ri.get("sector") == "devtools" else 0.4)
                + 0.25 * 0.55
                + 0.30 * ri.get("traction", 0.5))

    def dispersion_report(self) -> Dict[str, float]:
        return decision_dispersion(self.env.action_log)

    # -- Layer 6 --------------------------------------------------------------
    def validate_2021_surge(self, n_replications: int = 20) -> ValidationReport:
        """Replay the 2021 funding surge with real starting conditions and
        score against slice KPIs (§2.6). Historical targets are placeholders —
        calibrate them from the Layer 1 graph once real data is ingested."""
        targets = [
            KPITarget("mean_round_size_m", historical_value=9.0, tolerance=0.6),
            KPITarget("mean_step_up", historical_value=2.2, tolerance=0.8),
            KPITarget("mean_syndicate_size", historical_value=3.0, tolerance=0.8),
        ]

        def replication(seed: int) -> Dict[str, float]:
            cfg = SliceConfig(**{**self.config.__dict__, "seed": seed})
            slice_ = USVentureCapitalSlice(cfg, bus=EventBus())
            slice_.env.state.market_heat = 0.85  # 2021 starting conditions
            return slice_.run()

        return validate_episode("2021_funding_surge", targets, replication,
                                n_replications=n_replications)


def run_scripted_vs_emergent(seed: int = 0) -> ComparisonResult:
    """The L5 comparison for this slice (§2.5): VC has a real herding
    phenomenon (hot rounds) — distinguish realistic momentum from
    LLM-population artifact."""
    def build_and_run(emergent: bool, s: int) -> Dict[str, float]:
        cfg = SliceConfig(seed=s, emergent=emergent)
        return USVentureCapitalSlice(cfg).run()

    return run_comparison("us_vc_seed_to_a", build_and_run, seed=seed)
