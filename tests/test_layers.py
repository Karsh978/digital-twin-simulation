"""Tests for the meta-architecture and Layers 1, 2, 4, 5, 6."""

import pytest

from digital_twin.agents.debate import internal_debate
from digital_twin.agents.llm import MockLLM
from digital_twin.emergence import herding_index, run_comparison
from digital_twin.environment import Action, EnvironmentServer
from digital_twin.event_bus import Confidence, EventBus, ShockType
from digital_twin.knowledge_graph import (KnowledgeGraph, LicenseStatus,
                                          SourceRecord)
from digital_twin.personas import (PersonaSynthesizer, PoolTooThinError,
                                   SourceProfile, build_archetype)
from digital_twin.validation import (DivergenceLayer, KPITarget,
                                     distribution_shape_error, validate_episode)


# -- event bus ------------------------------------------------------------------

def test_bus_delivers_after_delay_and_not_to_origin():
    bus = EventBus()
    received = []
    bus.subscribe("slice_b", received.append)
    bus.publish(ShockType.RATE_CHANGE, "slice_a", 0.5, current_week=0, delay_weeks=4)
    assert bus.dispatch_due(3) == []
    due = bus.dispatch_due(4)
    assert len(due) == 1 and len(received) == 1


def test_bus_does_not_echo_to_origin():
    bus = EventBus()
    received = []
    bus.subscribe("slice_a", received.append)
    bus.publish(ShockType.DEMAND_SHIFT, "slice_a", 0.5, current_week=0)
    bus.dispatch_due(0)
    assert received == []


def test_cross_slice_confidence_is_forced():
    bus = EventBus()
    ev = bus.publish(ShockType.MAJOR_FAILURE, "a", 1.0, 0,
                     confidence=Confidence.SLICE_VALIDATED)
    assert ev.confidence == Confidence.CROSS_SLICE  # §1.1 lower-confidence layer


# -- Layer 1: ingestion gate ------------------------------------------------------

def test_gate_rejects_unlicensed_content():
    g = KnowledgeGraph()
    with pytest.raises(PermissionError):
        g.ingest([SourceRecord("hbs_cases", "company",
                               {"name": "CaseCo"}, LicenseStatus.UNCONFIRMED)])


def test_entity_resolution_merges_fuzzy_duplicates():
    g = KnowledgeGraph()
    a = g.add_entity("company", "Acme, Inc.", source="crunchbase")
    b = g.add_entity("company", "Acme Inc", source="edgar")
    assert a.id == b.id
    assert set(a.sources) == {"crunchbase", "edgar"}


# -- Layer 1: persona synthesis discipline ----------------------------------------

def _profiles(n, seed=0):
    import random
    rng = random.Random(seed)
    return [SourceProfile(features={"risk_tolerance": rng.random(),
                                    "thesis_conviction": rng.random(),
                                    "herd_sensitivity": rng.random()},
                          categorical={"thesis_type": "devtools"})
            for _ in range(n)]


def test_pool_size_gate():
    arch = build_archetype("thin", _profiles(10))
    with pytest.raises(PoolTooThinError):
        PersonaSynthesizer(min_pool=24).synthesize(arch, 5)


def test_personas_are_synthetic_and_diverse():
    arch = build_archetype("wide", _profiles(60))
    personas = PersonaSynthesizer(min_pool=24, seed=1).synthesize(arch, 20)
    assert len({p.persona_id for p in personas}) == 20       # unique synthetic IDs
    assert len({round(p.traits["risk_tolerance"], 3) for p in personas}) > 5  # diversity
    assert all(0.6 <= p.temperature <= 1.0 for p in personas)


# -- Layer 4: internal debate ------------------------------------------------------

def test_debate_uses_distinct_models_and_reports_ambivalence():
    out = internal_debate(MockLLM(), "persona summary", "issue term sheet",
                          "context",
                          models={"advocate": "m1", "challenger": "m2",
                                  "arbitrator": "m3"})
    assert out.decision in ("proceed", "decline")
    assert 0.0 <= out.confidence <= 1.0
    assert out.ambivalence == abs(out.advocate_score - out.challenger_score)
    assert set(out.trace) == {"advocate", "challenger", "arbitrator"}


# -- Layer 2: environment -----------------------------------------------------------

def test_environment_ticks_and_logs():
    env = EnvironmentServer("test_slice")
    env.run(5)
    assert env.state.week == 5


def test_environment_perceives_shock():
    bus = EventBus()
    env = EnvironmentServer("receiving_slice", bus=bus)
    heat_before = env.state.market_heat
    bus.publish(ShockType.RATE_CHANGE, "origin_slice", 0.8, current_week=0,
                delay_weeks=1)
    env.run(2)
    assert env.state.market_heat < heat_before
    assert env.state.shock_log[0]["confidence"] == "cross_slice"


# -- Layer 5 ---------------------------------------------------------------------

def test_herding_index():
    assert herding_index([0.5, 0.5, 0.5]) == pytest.approx(1.0)   # perfect herding
    assert herding_index([0.0, 1.0]) == pytest.approx(0.0)        # max dispersion


def test_comparison_delta():
    result = run_comparison("demo",
                            lambda emergent, seed: {"x": 1.0 if emergent else 0.5})
    assert result.delta["x"] == pytest.approx(0.5)


# -- Layer 6 ----------------------------------------------------------------------

def test_validation_requires_15_to_30_replications():
    with pytest.raises(ValueError):
        validate_episode("ep", [KPITarget("k", 1.0)], lambda s: {"k": 1.0},
                         n_replications=5)


def test_validation_pass_and_diagnosis():
    targets = [KPITarget("k", historical_value=1.0, tolerance=0.2)]
    rep = validate_episode("ep", targets, lambda s: {"k": 1.05}, n_replications=15)
    assert rep.passed
    assert rep.diagnoses["k"] == DivergenceLayer.OK

    bad = validate_episode("ep", targets, lambda s: {"k": 3.0}, n_replications=15)
    assert not bad.passed
    assert bad.diagnoses["k"] == DivergenceLayer.LAYER_3_RULES


def test_distribution_shape_error():
    assert distribution_shape_error([1, 2, 3], [1, 2, 3]) == 0.0
    assert distribution_shape_error([1, 1, 1], [3, 3, 3]) == pytest.approx(2.0)
