"""Layer 1 — knowledge graph + entity resolution.

Grounds every persona and every piece of world state in real, slice-specific
data. In this reference implementation the graph is an in-memory typed graph;
swap the storage backend for Neo4j/etc. without changing the interface.

Entity types (per the U.S. VC worked example, §2.1):
    Company, Person, Fund, Round
Edge types:
    invested_in, worked_at, co_invested_with, acquired

Ingestion discipline (§1.4/§1.5) is enforced at the gate: every record carries
a `SourceRecord` with a licensing status, and `IngestionGate` refuses records
whose licensing is unconfirmed. This is a gate, not a cleanup step.
"""

from __future__ import annotations

import itertools
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple


class LicenseStatus(str, Enum):
    FREE = "free"                      # e.g. SEC EDGAR, GDELT
    API_LICENSED = "api_licensed"      # e.g. Crunchbase/PitchBook under a current agreement
    CONFIRMED = "confirmed"            # rights holder confirmed in writing (e.g. HBS cases)
    UNCONFIRMED = "unconfirmed"        # NOT ingestible — gate rejects


@dataclass
class SourceRecord:
    """One raw record from a source adapter, before it enters the graph."""

    source: str
    record_type: str                   # "company" | "person" | "fund" | "round"
    data: dict
    license_status: LicenseStatus


class IngestionGate:
    """Non-negotiable gate on Layer 1 (§1.4/§1.5). Rejects unlicensed content."""

    def check(self, record: SourceRecord) -> None:
        if record.license_status == LicenseStatus.UNCONFIRMED:
            raise PermissionError(
                f"Refusing to ingest record from '{record.source}': licensing "
                "status unconfirmed. Confirm with the rights holder first (§1.5)."
            )

    def filter(self, records: Iterable[SourceRecord]) -> List[SourceRecord]:
        ok = []
        for r in records:
            self.check(r)
            ok.append(r)
        return ok


@dataclass
class Entity:
    id: str
    type: str                          # company | person | fund | round
    name: str
    attrs: dict = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)


@dataclass
class Edge:
    src: str
    dst: str
    type: str                          # invested_in | worked_at | co_invested_with | acquired
    attrs: dict = field(default_factory=dict)


def _normalize(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).lower()
    name = re.sub(r"[^a-z0-9 ]", "", name)
    for suffix in (" inc", " llc", " ltd", " corp", " co", " the"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return re.sub(r"\s+", " ", name).strip()


class EntityResolver:
    """Fuzzy-match entity resolution (§2.1).

    Merges incoming records onto existing entities when normalized names are
    similar enough. Threshold is deliberately conservative — a false merge
    corrupts ground truth; a false split just duplicates a node.
    """

    def __init__(self, threshold: float = 0.92) -> None:
        self.threshold = threshold

    def match(self, name: str, candidates: Iterable[Entity]) -> Optional[Entity]:
        norm = _normalize(name)
        best: Tuple[float, Optional[Entity]] = (0.0, None)
        for c in candidates:
            score = SequenceMatcher(None, norm, _normalize(c.name)).ratio()
            if score > best[0]:
                best = (score, c)
        return best[1] if best[0] >= self.threshold else None


class KnowledgeGraph:
    """Typed in-memory graph grounding world state and personas."""

    def __init__(self, resolver: Optional[EntityResolver] = None) -> None:
        self.entities: Dict[str, Entity] = {}
        self.edges: List[Edge] = []
        self.resolver = resolver or EntityResolver()
        self._ids = itertools.count(1)
        self._gate = IngestionGate()

    # -- ingestion ---------------------------------------------------------
    def ingest(self, records: Iterable[SourceRecord]) -> List[Entity]:
        out = []
        for rec in self._gate.filter(records):
            out.append(self.add_entity(rec.record_type, rec.data.get("name", "unknown"),
                                       source=rec.source, **{k: v for k, v in rec.data.items() if k != "name"}))
        return out

    # -- graph primitives --------------------------------------------------
    def add_entity(self, type_: str, name: str, source: str = "", **attrs) -> Entity:
        same_type = [e for e in self.entities.values() if e.type == type_]
        existing = self.resolver.match(name, same_type)
        if existing is not None:
            existing.attrs.update(attrs)
            if source and source not in existing.sources:
                existing.sources.append(source)
            return existing
        e = Entity(id=f"{type_[:3]}-{next(self._ids)}", type=type_, name=name,
                   attrs=attrs, sources=[source] if source else [])
        self.entities[e.id] = e
        return e

    def add_edge(self, src: str, dst: str, type_: str, **attrs) -> Edge:
        for e in self.edges:
            if e.src == src and e.dst == dst and e.type == type_:
                e.attrs.update(attrs)
                return e
        edge = Edge(src=src, dst=dst, type=type_, attrs=attrs)
        self.edges.append(edge)
        return edge

    # -- queries -----------------------------------------------------------
    def of_type(self, type_: str) -> List[Entity]:
        return [e for e in self.entities.values() if e.type == type_]

    def neighbors(self, entity_id: str, edge_type: Optional[str] = None) -> List[Entity]:
        ids = []
        for e in self.edges:
            if edge_type and e.type != edge_type:
                continue
            if e.src == entity_id:
                ids.append(e.dst)
            elif e.dst == entity_id:
                ids.append(e.src)
        return [self.entities[i] for i in ids if i in self.entities]

    def co_investors(self, fund_id: str) -> List[Entity]:
        return self.neighbors(fund_id, "co_invested_with")

    def stats(self) -> dict:
        by_type: Dict[str, int] = {}
        for e in self.entities.values():
            by_type[e.type] = by_type.get(e.type, 0) + 1
        return {"entities": len(self.entities), "edges": len(self.edges), "by_type": by_type}
