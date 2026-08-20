"""Master Orchestrator — 'Hat Swapper' Dynamic Workflow Engine.

Decomposes high-level user intents into step-by-step action chains, swapping system prompts
(Agency Agent personas) and calling deployed tool APIs at each execution step.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any, Dict, List, Optional

from .llm import get_default_backend

log = logging.getLogger(__name__)

# Registry mapping tool domains to deployed Render services
DEPLOYED_TOOLS = {
    "osint": os.environ.get("OPENOSINT_API", "https://openosint-g4q8.onrender.com"),
    "social_search": os.environ.get("BLACKBIRD_API", "https://blackbird-ap9o.onrender.com"),
    "finance": os.environ.get("WHITEROCK_API", "https://whiterock-55be.onrender.com"),
    "hedge": os.environ.get("AUTOHEDGE_API", "https://autohedge-wzfa.onrender.com"),
    "graph_db": os.environ.get("ARCADEDB_URL", "https://arcadedb-1.onrender.com")
}


class HatSwapperOrchestrator:
    """Maintains a single conversation thread while dynamically swapping personas and tool bindings."""

    def __init__(self) -> None:
        self.llm = get_default_backend()
        self.agency_api = os.environ.get("AGENCY_AGENTS_API", "https://agency-agents-api.onrender.com")

    def _get_fallback_plan(self) -> List[Dict[str, Any]]:
        return [
            {"step_id": 1, "persona_type": "osint_cyber", "required_tools": ["osint", "social_search"], "action_description": "Search public records and social presence."},
            {"step_id": 2, "persona_type": "finance_accountant", "required_tools": ["finance", "hedge"], "action_description": "Inspect linked financial assets and market positions."},
            {"step_id": 3, "persona_type": "graph_analyst", "required_tools": ["graph_db"], "action_description": "Synthesize entity nodes into knowledge graph."}
        ]

    def decompose_prompt(self, user_prompt: str) -> List[Dict[str, Any]]:
        """Decomposes an end-to-end user request into an ordered sequence of domain actions."""
        planner_system = (
            "You are the Failsafe Master Task Planner. Decompose the user request into an ordered JSON array of steps.\n"
            "Each step must specify: 'step_id', 'persona_type' (osint_cyber, finance_accountant, legal_lawyer, graph_analyst), "
            "'required_tools' (list of tools), and 'action_description'.\n"
            "Respond ONLY with a valid JSON array."
        )
        plan_raw = self.llm.complete(planner_system, user_prompt, temperature=0.2)
        
        try:
            parsed = json.loads(plan_raw)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict) and "steps" in parsed and isinstance(parsed["steps"], list):
                return parsed["steps"]
        except Exception:
            pass
            
        return self._get_fallback_plan()

    def fetch_persona_prompt(self, persona_type: str) -> str:
        """Pulls agent prompt/workflow configuration from Agency Agents endpoint."""
        try:
            req = urllib.request.Request(f"{self.agency_api}/personas/{persona_type}")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    return data.get("system_prompt", f"You are acting as an expert {persona_type}.")
        except Exception:
            log.warning(f"Failed to fetch persona '{persona_type}' from Agency Agents API. Using inline template.")
        
        return f"You are the specialized Failsafe Agent for {persona_type.upper()}. Perform the step precisely."

    def execute_chain(self, user_prompt: str) -> Dict[str, Any]:
        """Executes the workflow step-by-step, passing context down the pipeline."""
        plan = self.decompose_prompt(user_prompt)
        execution_trace = []
        accumulated_context = f"User Request: {user_prompt}\n"

        for step in plan:
            if not isinstance(step, dict):
                continue

            persona_type = step.get("persona_type", "generic_agent")
            step_id = step.get("step_id", 1)
            action_desc = step.get("action_description", "Execute task")
            req_tools = step.get("required_tools", [])

            persona_prompt = self.fetch_persona_prompt(persona_type)
            step_prompt = (
                f"{accumulated_context}\n"
                f"CURRENT STEP ({step_id}): {action_desc}\n"
                f"Available Tools: {req_tools}\n"
                f"Perform analysis for this stage and output structured observations."
            )

            # Swap system prompt ("Hat Swapper") while preserving thread context
            step_result = self.llm.complete(system=persona_prompt, prompt=step_prompt, temperature=0.5)
            
            trace_item = {
                "step": step_id,
                "persona": persona_type,
                "action": action_desc,
                "output": step_result
            }
            execution_trace.append(trace_item)
            accumulated_context += f"\n--- Step {step_id} ({persona_type}) Output ---\n{step_result}\n"

        return {
            "status": "completed",
            "total_steps": len(execution_trace),
            "final_synthesis": accumulated_context,
            "trace": execution_trace
        }