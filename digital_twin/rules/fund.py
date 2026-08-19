"""Fund mechanics (§2.3): fund size, reserve ratios, check size bands by
stage — the actual constraint on follow-on behavior."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class Stage(str, Enum):
    SEED = "seed"
    SERIES_A = "series_a"
    SERIES_B = "series_b"


@dataclass
class CheckSizeBand:
    stage: Stage
    min_check: float
    max_check: float

    def clamp(self, amount: float) -> float:
        return min(self.max_check, max(self.min_check, amount))


@dataclass
class Fund:
    """A VC fund's deterministic constraints.

    reserve_ratio: fraction of fund size held back for follow-ons.
    """

    name: str
    size: float
    reserve_ratio: float = 0.5
    check_bands: Dict[Stage, CheckSizeBand] = field(default_factory=dict)
    deployed_initial: float = 0.0      # initial checks written
    deployed_followon: float = 0.0
    portfolio: Dict[str, float] = field(default_factory=dict)  # company -> invested

    @property
    def initial_budget(self) -> float:
        return self.size * (1 - self.reserve_ratio)

    @property
    def reserve_budget(self) -> float:
        return self.size * self.reserve_ratio

    @property
    def initial_remaining(self) -> float:
        return self.initial_budget - self.deployed_initial

    @property
    def reserves_remaining(self) -> float:
        return self.reserve_budget - self.deployed_followon

    def can_write_initial(self, stage: Stage, amount: float) -> bool:
        band = self.check_bands.get(stage)
        if band and not (band.min_check <= amount <= band.max_check):
            return False
        return amount <= self.initial_remaining

    def write_initial(self, company: str, stage: Stage, amount: float) -> float:
        """Write an initial check; returns the amount actually written (0 if
        outside the band or beyond remaining budget)."""
        if not self.can_write_initial(stage, amount):
            return 0.0
        self.deployed_initial += amount
        self.portfolio[company] = self.portfolio.get(company, 0.0) + amount
        return amount

    def can_follow_on(self, company: str, amount: float) -> bool:
        """Follow-ons come out of reserves, and only into existing portfolio —
        this is what actually constrains follow-on behavior (§2.3)."""
        return company in self.portfolio and amount <= self.reserves_remaining

    def write_follow_on(self, company: str, amount: float) -> float:
        if not self.can_follow_on(company, amount):
            return 0.0
        self.deployed_followon += amount
        self.portfolio[company] += amount
        return amount

    def dry_powder(self) -> float:
        return self.initial_remaining + self.reserves_remaining
