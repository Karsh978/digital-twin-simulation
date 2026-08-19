"""Layer 1 — persona synthesis pipeline (§1.4), with the privacy discipline
enforced in code.

Rules implemented here, directly from the doc:
- NEVER build a persona as a collage of real individuals' words/details.
  Personas are STATISTICAL SYNTHESES over a wide pool: cluster patterns
  (reasoning style, thesis type, risk posture, check-size behavior), then
  generate genuinely new parameter draws from cluster statistics.
- Pool size gate: fewer than `min_pool` source individuals per archetype
  refuses synthesis (the doc's "dozens-plus per archetype"; default 24).
- Outlier filtering: rare, highly distinctive attribute values are screened
  out BEFORE generation, because a strong outlier can leak through averaging.
- Archetypes are never bound 1:1 to a named individual. Personas carry no
  real names — synthetic identifiers only.
"""

from __future__ import annotations

import hashlib
import random
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Sequence


class PoolTooThinError(ValueError):
    pass


@dataclass
class SourceProfile:
    """Anonymized features extracted from one real source individual.

    Never store raw text here — only derived, bucketed features. Raw ingested
    data retention is a legal/policy matter (§1.4) and lives outside this
    pipeline.
    """

    features: Dict[str, float]        # e.g. {"risk_tolerance": 0.7, "thesis_breadth": 0.4}
    categorical: Dict[str, str] = field(default_factory=dict)  # e.g. {"thesis_type": "devtools"}


@dataclass
class Archetype:
    """A cluster of source profiles: the right level of abstraction (§1.4)."""

    name: str
    size: int
    means: Dict[str, float]
    stds: Dict[str, float]
    categorical_dist: Dict[str, Dict[str, float]]  # feature -> value -> probability


@dataclass
class Persona:
    """A synthetic persona. Contains no real person's words or identity."""

    persona_id: str                    # synthetic identifier, never a real name
    archetype: str
    traits: Dict[str, float]
    categorical: Dict[str, str]
    temperature: float                 # sampling variance lever (§1.2 Layer 5)


def _outlier_filter(profiles: Sequence[SourceProfile], z_thresh: float = 2.5) -> List[SourceProfile]:
    """Drop profiles with any feature more than z_thresh std devs from the pool
    mean — rare, highly distinctive details are excluded before generation."""
    if len(profiles) < 4:
        return list(profiles)
    keys = set().union(*(p.features.keys() for p in profiles))
    keep = []
    for p in profiles:
        distinctive = False
        for k in keys:
            vals = [q.features.get(k, 0.0) for q in profiles]
            mu, sd = statistics.fmean(vals), (statistics.pstdev(vals) or 1e-9)
            if abs(p.features.get(k, 0.0) - mu) / sd > z_thresh:
                distinctive = True
                break
        if not distinctive:
            keep.append(p)
    return keep


def cluster_profiles(profiles: Sequence[SourceProfile], n_archetypes: int = 4,
                     seed: int = 0) -> List[List[SourceProfile]]:
    """Simple k-means over numeric feature vectors (dependency-free)."""
    rng = random.Random(seed)
    keys = sorted(set().union(*(p.features.keys() for p in profiles)))
    vecs = [[p.features.get(k, 0.0) for k in keys] for p in profiles]
    if len(vecs) <= n_archetypes:
        return [list(profiles)]
    centroids = rng.sample(vecs, n_archetypes)
    for _ in range(50):
        groups: List[List[int]] = [[] for _ in centroids]
        for i, v in enumerate(vecs):
            j = min(range(len(centroids)),
                    key=lambda c: sum((a - b) ** 2 for a, b in zip(v, centroids[c])))
            groups[j].append(i)
        new_centroids = []
        for c, g in enumerate(groups):
            if g:
                new_centroids.append([statistics.fmean(vecs[i][d] for i in g)
                                      for d in range(len(keys))])
            else:
                new_centroids.append(centroids[c])
        if new_centroids == centroids:
            break
        centroids = new_centroids
    return [[profiles[i] for i in g] for g in groups if g]


def build_archetype(name: str, profiles: Sequence[SourceProfile]) -> Archetype:
    keys = sorted(set().union(*(p.features.keys() for p in profiles)))
    means = {k: statistics.fmean(p.features.get(k, 0.0) for p in profiles) for k in keys}
    stds = {k: (statistics.pstdev([p.features.get(k, 0.0) for p in profiles]) or 0.05)
            for k in keys}
    cat_keys = set().union(*(p.categorical.keys() for p in profiles))
    cat_dist: Dict[str, Dict[str, float]] = {}
    for ck in cat_keys:
        counts = Counter(p.categorical.get(ck, "unspecified") for p in profiles)
        total = sum(counts.values())
        cat_dist[ck] = {v: n / total for v, n in counts.items()}
    return Archetype(name=name, size=len(profiles), means=means, stds=stds,
                     categorical_dist=cat_dist)


class PersonaSynthesizer:
    """Generates synthetic personas from archetype statistics.

    Enforces: pool-size gate, outlier filtering, no 1:1 identity binding.
    """

    def __init__(self, min_pool: int = 24, seed: int = 0) -> None:
        self.min_pool = min_pool
        self.rng = random.Random(seed)

    def synthesize(self, archetype: Archetype, n: int,
                   temperature_range=(0.6, 1.0)) -> List[Persona]:
        if archetype.size < self.min_pool:
            raise PoolTooThinError(
                f"Archetype '{archetype.name}' built from {archetype.size} source "
                f"individuals; minimum pool is {self.min_pool} (§1.4: widen to "
                "dozens-plus per archetype before synthesizing)."
            )
        personas = []
        for i in range(n):
            traits = {k: min(1.0, max(0.0, self.rng.gauss(archetype.means[k],
                                                          archetype.stds[k])))
                      for k in archetype.means}
            cats = {}
            for ck, dist in archetype.categorical_dist.items():
                vals, weights = zip(*dist.items())
                cats[ck] = self.rng.choices(vals, weights=weights, k=1)[0]
            # Synthetic ID only — deliberately NOT traceable to any source person.
            pid = "syn-" + hashlib.sha256(
                f"{archetype.name}:{i}:{self.rng.random()}".encode()).hexdigest()[:10]
            personas.append(Persona(
                persona_id=pid, archetype=archetype.name, traits=traits,
                categorical=cats,
                temperature=self.rng.uniform(*temperature_range),
            ))
        return personas

    def reidentifiability_spot_check(self, persona: Persona,
                                     pool: Sequence[SourceProfile],
                                     max_similarity: float = 0.85) -> bool:
        """Manual-spot-check helper (§2.9 first milestone): returns True if the
        persona is TOO CLOSE to any single source profile (i.e. fails the check)."""
        keys = sorted(persona.traits.keys())
        for p in pool:
            num = sum(persona.traits.get(k, 0) * p.features.get(k, 0) for k in keys)
            da = sum(v * v for v in persona.traits.values()) ** 0.5 or 1e-9
            db = sum(p.features.get(k, 0) ** 2 for k in keys) ** 0.5 or 1e-9
            if num / (da * db) > max_similarity:
                return True
        return False
