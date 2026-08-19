"""End-to-end Multi-Industry Digital Twin Generator (Failsafe Platform Engine).

Supports all 23 verticals using OpenRouter API cognition and Failsafe Master Agent API.

Usage:
    python examples/run_industry_simulation.py --industry osint_cyber --weeks 52 --shock
    python examples/run_industry_simulation.py --industry vc_finance --weeks 104
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add root package to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from digital_twin.event_bus import Confidence, EventBus, ShockType
from digital_twin.slices.factory import (SUPPORTED_VERTICALS,
                                          UniversalIndustrySlice,
                                          UniversalSliceConfig)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    ap = argparse.ArgumentParser(description="Failsafe Multi-Industry Digital Twin Engine")
    ap.add_argument("--industry", type=str, default="vc_finance", choices=SUPPORTED_VERTICALS,
                    help="Target vertical slice to simulate")
    ap.add_argument("--weeks", type=int, default=52, help="Number of simulated weeks")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    ap.add_argument("--shock", action="store_true", help="Inject dynamic cross-slice shock event")
    args = ap.parse_args()

    cfg = UniversalSliceConfig(sim_weeks=args.weeks, seed=args.seed, industry_type=args.industry)
    bus = EventBus()

    print(f"\n=======================================================")
    print(f" FAILSAFE DIGITAL TWIN ENGINE v2 — [{args.industry.upper()}]")
    print(f" LLM Backend: OpenRouter API | Ingestion: Failsafe Master API")
    print(f"=======================================================\n")

    slice_instance = UniversalIndustrySlice(cfg, bus=bus)

    if args.shock:
        bus.publish(
            ShockType.RATE_CHANGE,
            origin_slice="failsafe_macro_orchestrator",
            magnitude=0.75,
            current_week=0,
            delay_weeks=8,
            description=f"Macro disturbance injection into {args.industry}",
            confidence=Confidence.CROSS_SLICE
        )

    kpis = slice_instance.run()

    print("\n-- Industry Slice KPIs --")
    for k, v in kpis.items():
        print(f"  {k}: {v:.4g}")

    print("\n-- Anti-Herding & Agent Dispersion --")
    for k, v in slice_instance.dispersion_report().items():
        print(f"  {k}: {v:.4g}")

    n_debates = sum(1 for r in slice_instance.env.action_log if r.debate_trace)
    print(f"\n-- Layer 4 Debate Gate --\n  Consequential decisions routed: {n_debates}")

    if args.shock:
        print("\n-- Shock Event Log --")
        for s in slice_instance.env.state.shock_log:
            print(f"  Week {s['week']}: {s['type']} (mag={s['magnitude']}) from {s['origin']}")

    log_path = Path(__file__).parent / f"action_log_{args.industry}.jsonl"
    slice_instance.env.export_log(str(log_path))
    print(f"\nTraceability Log Saved: {log_path}\n")


if __name__ == "__main__":
    main()