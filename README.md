# Industry Digital Twin — Reference Implementation

A runnable Python implementation of the **Industry Digital Twin Architecture v2**:
micro-slices on an async event bus, with U.S. venture capital (seed → Series A
software) as the worked example.

The whole system runs out of the box with a **deterministic mock LLM** — no API
key, no network. Set `OPENAI_API_KEY` to route agent cognition and internal
debates through a real OpenAI-compatible model.

## Quick start

```bash
cd digital-twin
pip install -e .          # or just run from the repo root; no required deps
python examples/run_us_vc_demo.py                  # 2-year simulation, ~seconds
python examples/run_us_vc_demo.py --compare        # + scripted-vs-emergent (L5)
python examples/run_us_vc_demo.py --shock          # + cross-slice rate shock
python examples/run_us_vc_demo.py --validate       # + Layer 6 historical replay
python -m pytest tests/ -q                         # 38 tests
```

## Architecture map

| Doc section | Concept | Code |
|---|---|---|
| §1.1 | Micro-slices, async event bus, lower-confidence cross-slice output | `digital_twin/event_bus.py` |
| §1.2 L1 | Ingestion gate (licensing enforced *before* the graph), entity resolution, knowledge graph | `digital_twin/knowledge_graph.py` |
| §1.4 | Persona synthesis: statistical archetypes, pool-size gate, outlier filtering, synthetic IDs only | `digital_twin/personas.py` |
| §1.2 L2 | Environment server, weekly time engine, action space, full state logging | `digital_twin/environment.py` |
| §1.2 L3 | Procedural rules — pure functions, no LLM: cap tables, SAFE conversion, liquidation prefs, fund mechanics, runway | `digital_twin/rules/` |
| §1.2 L4 | Two-speed agents, consequential-decision gate, advocate/challenger/arbitrator debate (multi-model) | `digital_twin/agents/` |
| §1.2 L5 | Emergent surface, stacked anti-herding levers, herding metrics, scripted-vs-emergent comparison | `digital_twin/emergence.py` |
| §1.2 L6 | Historical replay, 15–30 replications, KPI scoring, divergence diagnosis by layer | `digital_twin/validation.py` |
| Part 2 | U.S. VC slice wiring all six layers | `digital_twin/slices/us_vc.py` |

## Design rules encoded in code (not just docs)

- **Ingestion gate, not cleanup.** `IngestionGate` raises on any record with
  `LicenseStatus.UNCONFIRMED` — unlicensed content never enters the graph (§1.5).
- **Personas are statistical syntheses.** `PersonaSynthesizer` refuses
  archetypes built from fewer than `min_pool` source individuals
  (`PoolTooThinError`), filters distinctive outliers before generation, and
  issues synthetic IDs only — never real names, never 1:1 identity binding (§1.4).
- **The debate is gated, not per-tick.** Only slice-defined consequential
  actions (term sheets, lead/follow, exits) route through the internal
  advocate/challenger/arbitrator pass; routine ticks stay on the fast path (§1.2 L4).
- **Cross-slice output is labeled lower-confidence.** The bus forces
  `Confidence.CROSS_SLICE` on anything it propagates, and slices log shocks
  with that label (§1.1).
- **The population cap controls compute, not correlation.** Herding is
  *measured* (`herding_index`, decision dispersion) and countered with stacked
  levers — persona diversity, per-persona temperature, shuffled (staggered)
  tick order, imitation friction, path-dependent memory, the debate gate (§v2
  correction, §1.2 L5).
- **Agents never mutate state.** They submit actions; resolvers apply Layer 3
  rules. Every action is logged with its debate trace for traceability.

## What the demo shows

A 104-week run with 40 startups and 12 funds produces the slice KPIs from
§2.6 (round sizes, step-up multiples, syndicate composition, time-to-next-round),
a dispersion report, and a count of decisions that went through the internal
debate with their mean ambivalence. `--validate` replays the 2021 surge across
15 replications. Expect it to **FAIL some KPIs out of the box** — that is the
harness working: historical targets are placeholders until you calibrate Layer 3
against real data ingested through Layer 1, and the report tells you which layer
to fix (wrong shape → L3, convergence → L5, timing → L2).

## Extending it

- **Real data**: implement adapters returning `SourceRecord`s (Crunchbase,
  EDGAR Form D, GDELT per §2.1) and feed them to `KnowledgeGraph.ingest` —
  the licensing gate applies automatically.
- **Real LLM**: `pip install -e .[llm]`, set `OPENAI_API_KEY` (and optionally
  `OPENAI_BASE_URL` for compatible providers). Pass `debate_models` to give
  advocate/challenger/arbitrator different underlying models.
- **A second slice** (e.g. U.K. VC): new module under `digital_twin/slices/`
  with its own sourcing and rules — not a "country" field on the U.S. slice.
  Connect slices only through `EventBus` shock events, and only after the new
  slice passes its own Layer 6 validation (§1.3 step 8).

## Layout

```
digital_twin/
  event_bus.py       # meta-architecture: shocks between slices
  knowledge_graph.py # L1: graph, entity resolution, ingestion gate
  personas.py        # L1: persona synthesis + privacy discipline
  environment.py     # L2: world server, time engine, action space, logging
  rules/             # L3: cap_table, valuation, liquidation, fund, runway
  agents/            # L4: llm backends, debate, two-speed base, VC agents
  emergence.py       # L5: anti-herding levers, metrics, comparison harness
  validation.py      # L6: replay, replications, KPI scoring, diagnosis
  slices/us_vc.py    # worked example slice
examples/run_us_vc_demo.py
tests/               # 38 tests: rule engine canonical cases + layer behavior
```
