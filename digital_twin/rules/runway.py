"""Runway/burn model (§2.3): cash depletion as a function of headcount and
spend — what forces founders back to market. Weekly resolution to match the
environment server's weekly ticks (§2.2)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunwayModel:
    cash: float
    headcount: int
    avg_monthly_cost_per_head: float = 18_000.0   # fully loaded, U.S. software
    other_monthly_spend: float = 25_000.0         # infra, tools, legal, etc.
    monthly_revenue: float = 0.0

    WEEKS_PER_MONTH = 52.0 / 12.0

    @property
    def monthly_burn(self) -> float:
        gross = self.headcount * self.avg_monthly_cost_per_head + self.other_monthly_spend
        return max(0.0, gross - self.monthly_revenue)

    @property
    def weekly_burn(self) -> float:
        return self.monthly_burn / self.WEEKS_PER_MONTH

    def runway_weeks(self) -> float:
        burn = self.weekly_burn
        if burn <= 0:
            return float("inf")
        return self.cash / burn

    def tick(self, weeks: int = 1) -> float:
        """Advance time; returns remaining cash. Floors at zero."""
        self.cash = max(0.0, self.cash - self.weekly_burn * weeks)
        return self.cash

    def is_out_of_runway(self) -> bool:
        return self.cash <= 0.0

    def needs_raise_within(self, weeks: int) -> bool:
        """Founders typically go back to market with ~6 months (26 weeks) left."""
        return self.runway_weeks() <= weeks
