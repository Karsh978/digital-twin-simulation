"""Layer 6 — validation & calibration harness.

Replay a real historical episode specific to the slice, run 15–30 independent
replications, score against slice-specific KPIs using real historical data,
and diagnose divergence BY LAYER (§1.2):

    wrong distribution shape -> Layer 3 (procedural rules)
    excessive convergence    -> herding; back to Layer 5 (anti-herding levers)
    wrong timing             -> Layer 2 (time engine) or the slice's
                                runway/dynamics model

Build this in parallel with Layer 5, not after it — and keep it decoupled
from the agent builders for independent scoring (§2.8).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Sequence


class DivergenceLayer(str, Enum):
    LAYER_2_TIMING = "layer_2_time_engine_or_dynamics"
    LAYER_3_RULES = "layer_3_procedural_rules"
    LAYER_5_HERDING = "layer_5_herding"
    OK = "within_tolerance"


@dataclass
class KPITarget:
    """A slice-specific KPI with its real historical value and tolerance."""

    name: str
    historical_value: float
    tolerance: float = 0.25            # relative tolerance, e.g. 0.25 = ±25%
    kind: str = "level"                # "level" | "distribution_shape" | "timing"


@dataclass
class ReplicationResult:
    seed: int
    kpis: Dict[str, float]


@dataclass
class ValidationReport:
    episode: str
    n_replications: int
    kpi_means: Dict[str, float]
    kpi_stds: Dict[str, float]
    targets: List[KPITarget]
    diagnoses: Dict[str, DivergenceLayer]
    passed: bool

    def summary(self) -> str:
        lines = [f"validation: {self.episode} "
                 f"({self.n_replications} replications) — "
                 f"{'PASS' if self.passed else 'FAIL'}"]
        for t in self.targets:
            mu = self.kpi_means.get(t.name, float("nan"))
            sd = self.kpi_stds.get(t.name, float("nan"))
            diag = self.diagnoses[t.name]
            lines.append(f"  {t.name}: sim={mu:.4g}±{sd:.3g} "
                         f"historical={t.historical_value:.4g} "
                         f"(±{t.tolerance:.0%}) -> {diag.value}")
        return "\n".join(lines)


def _diagnose(target: KPITarget, sim_mean: float, sim_std: float,
              replication_kpis: List[Dict[str, float]]) -> DivergenceLayer:
    hist = target.historical_value
    if hist == 0:
        rel_err = abs(sim_mean) 
    else:
        rel_err = abs(sim_mean - hist) / abs(hist)
    if rel_err <= target.tolerance:
        return DivergenceLayer.OK
    if target.kind == "timing":
        return DivergenceLayer.LAYER_2_TIMING
    # herding signature: the population converges (near-zero dispersion on a
    # distribution-shaped KPI) AND lands in the wrong place (§1.2 Layer 6)
    if target.kind == "distribution_shape" and sim_std < 0.05 * abs(hist or 1.0):
        return DivergenceLayer.LAYER_5_HERDING
    return DivergenceLayer.LAYER_3_RULES


def validate_episode(
    episode: str,
    targets: List[KPITarget],
    run_replication: Callable[[int], Dict[str, float]],
    n_replications: int = 20,
    base_seed: int = 1000,
) -> ValidationReport:
    """Run `n_replications` independent replications (15–30 per §1.2) of a
    historical replay and score against the KPI targets."""
    if not 15 <= n_replications <= 30:
        raise ValueError("replications must be in [15, 30] (§1.2 Layer 6)")
    results: List[ReplicationResult] = []
    for i in range(n_replications):
        results.append(ReplicationResult(base_seed + i, run_replication(base_seed + i)))

    kpi_means: Dict[str, float] = {}
    kpi_stds: Dict[str, float] = {}
    diagnoses: Dict[str, DivergenceLayer] = {}
    for t in targets:
        vals = [r.kpis.get(t.name, 0.0) for r in results]
        mu = statistics.fmean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        kpi_means[t.name], kpi_stds[t.name] = mu, sd
        diagnoses[t.name] = _diagnose(t, mu, sd, [r.kpis for r in results])

    passed = all(d == DivergenceLayer.OK for d in diagnoses.values())
    return ValidationReport(episode, n_replications, kpi_means, kpi_stds,
                            targets, diagnoses, passed)


def distribution_shape_error(simulated: Sequence[float],
                             historical: Sequence[float]) -> float:
    """Wasserstein-1 distance between two samples (distribution-shape KPI).
    Pure Python; fine for the sample sizes here."""
    if not simulated or not historical:
        return float("inf")
    s, h = sorted(simulated), sorted(historical)
    n = max(len(s), len(h))
    qs = [s[min(len(s) - 1, int(i * len(s) / n))] for i in range(n)]
    qh = [h[min(len(h) - 1, int(i * len(h) / n))] for i in range(n)]
    return sum(abs(a - b) for a, b in zip(qs, qh)) / n
