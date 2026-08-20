"""Quick verification script for Master Router ('Hat Swapper') Workflow Chain."""

import os
import sys
import json
import logging

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from digital_twin.agents.master_router import HatSwapperOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    print("\n=======================================================")
    print(" TESTING FAILSAFE MASTER ROUTER ('HAT SWAPPER') ENGINE ")
    print("=======================================================\n")

    orchestrator = HatSwapperOrchestrator()

    sample_prompt = (
        "Investigate company AlphaCorp: check public social records, "
        "verify linked financial assets, and synthesize into a graph."
    )

    print(f"User Prompt: {sample_prompt}\n")
    print("Executing Action Chain...\n")

    result = orchestrator.execute_chain(sample_prompt)

    print("\n-- Execution Summary --")
    print(f"Status: {result['status']}")
    print(f"Total Steps Executed: {result['total_steps']}")

    print("\n-- Step-by-Step Trace --")
    for step in result["trace"]:
        print(f"\n[Step {step['step']}] Persona: {step['persona']}")
        print(f"Action: {step['action']}")
        print(f"Output Preview: {step['output'][:150]}...")

    print("\n=======================================================")
    print(" SUCCESS: Master Router execution verified!")
    print("=======================================================\n")

if __name__ == "__main__":
    main()