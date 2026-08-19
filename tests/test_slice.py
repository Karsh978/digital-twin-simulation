"""Integration tests: the U.S. VC slice runs end-to-end (mock LLM)."""

import pytest

from digital_twin.event_bus import EventBus, ShockType
from digital_twin.slices.us_vc import (SliceConfig, USVentureCapitalSlice,
                                     run_scripted_vs_emergent)


def _quick_slice(seed=0, weeks=52):
    cfg = SliceConfig(n_companies=12, n_funds=5, persona_pool=60,
                      sim_weeks=weeks, seed=seed)
    return USVentureCapitalSlice(cfg, bus=EventBus())


def test_slice_builds_graph_and_agents():
    s = _quick_slice()
    stats = s.graph.stats()
    assert stats["by_type"]["company"] == 12
    assert stats["by_type"]["fund"] == 5
    assert len(s.founders) == 12 and len(s.partners) == 5


def test_slice_runs_and_produces_kpis():
    s = _quick_slice(weeks=78)
    kpis = s.run()
    assert set(kpis) == {"mean_round_size_m", "mean_step_up",
                         "mean_syndicate_size", "mean_time_to_next_round_wks"}
    assert s.env.state.week == 78
    # some activity happened
    assert len(s.env.action_log) > 0


def test_consequential_decisions_hit_the_debate_gate():
    s = _quick_slice(weeks=78)
    s.run()
    sheets = [r for r in s.env.action_log if r.action.value == "issue_term_sheet"]
    assert sheets, "expected at least one term sheet in 78 weeks"
    assert all(r.debate_trace is not None for r in sheets)


def test_scripted_mode_skips_debate():
    cfg = SliceConfig(n_companies=12, n_funds=5, persona_pool=60,
                      sim_weeks=52, seed=0, emergent=False)
    s = USVentureCapitalSlice(cfg, bus=EventBus())
    s.run()
    sheets = [r for r in s.env.action_log if r.action.value == "issue_term_sheet"]
    assert all(r.debate_trace is None for r in sheets)


def test_cross_slice_shock_propagates_with_lower_confidence():
    bus = EventBus()
    s = _quick_slice()
    s.bus = bus
    s.env.bus = bus
    bus.subscribe(s.name, s.env._on_shock)
    bus.publish(ShockType.RATE_CHANGE, "public_markets", 0.9,
                current_week=0, delay_weeks=5, description="hike")
    s.run()
    assert any(sh["origin"] == "public_markets"
               for sh in s.env.state.shock_log)
    assert all(sh["confidence"] == "cross_slice"
               for sh in s.env.state.shock_log)


def test_emergent_differs_from_scripted():
    result = run_scripted_vs_emergent(seed=3)
    # the whole point of the comparison: the agent layer must move SOME kpi
    assert any(abs(d) > 1e-9 for d in result.delta.values())
