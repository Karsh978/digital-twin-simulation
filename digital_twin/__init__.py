"""Industry Digital Twin — micro-slice + event-bus meta-architecture (v2).

Layers (per slice):
    1. Data & document ingestion  -> digital_twin.knowledge_graph, .personas
    2. World & environment server -> digital_twin.environment
    3. Procedural rule engine     -> digital_twin.rules
    4. Agent cognitive layer      -> digital_twin.agents
    5. Emergent behavior surface  -> digital_twin.emergence
    6. Validation & calibration   -> digital_twin.validation

Meta-architecture (cross-slice):
    digital_twin.event_bus
"""

__version__ = "0.2.0"
