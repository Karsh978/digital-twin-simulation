"""Cap table math (§2.3): dilution per round, option pool top-ups, pro-rata
rights. Deterministic; unit-tested against real historical rounds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ShareClass:
    name: str                          # e.g. "founders", "seed", "series_a", "option_pool"
    shares: float
    invested: float = 0.0              # total dollars paid in (for preference stacks)
    pro_rata: bool = False             # holds pro-rata rights


@dataclass
class CapTable:
    classes: Dict[str, ShareClass] = field(default_factory=dict)

    def add_class(self, sc: ShareClass) -> None:
        if sc.name in self.classes:
            self.classes[sc.name].shares += sc.shares
            self.classes[sc.name].invested += sc.invested
        else:
            self.classes[sc.name] = sc

    @property
    def total_shares(self) -> float:
        return sum(sc.shares for sc in self.classes.values())

    def ownership(self, name: str) -> float:
        total = self.total_shares
        if total <= 0:
            raise ValueError("cap table is empty")
        return self.classes[name].shares / total

    def dilution(self, name: str, new_shares: float) -> float:
        """Ownership percentage points lost by `name` if `new_shares` are issued."""
        before = self.ownership(name)
        after = self.classes[name].shares / (self.total_shares + new_shares)
        return before - after


def apply_option_pool_topup(table: CapTable, target_pool_pct: float,
                            pool_class: str = "option_pool") -> float:
    """Top up the option pool to `target_pool_pct` of the POST-topup total.

    Pool top-ups come out of the pre-money (founder dilution), standard in
    U.S. term sheets. Returns shares added to the pool.
    """
    if not 0 < target_pool_pct < 1:
        raise ValueError("target pool percentage must be in (0, 1)")
    current = table.classes.get(pool_class, ShareClass(pool_class, 0.0)).shares
    non_pool = table.total_shares - current
    # Solve: (current + x) / (non_pool + current + x) = target  =>
    # x = (target * (non_pool + current) - current) / (1 - target)
    x = (target_pool_pct * (non_pool + current) - current) / (1 - target_pool_pct)
    x = max(0.0, x)
    table.add_class(ShareClass(pool_class, x))
    return x


def issue_round(
    table: CapTable,
    round_name: str,
    pre_money_val: float,
    new_money: float,
    investors: Dict[str, float],
    pool_topup_pct: float | None = None,
) -> Dict[str, float]:
    """Issue a priced round. `investors` maps investor name -> dollars invested.

    Order of operations matches standard U.S. practice:
      1. (optional) option pool top-up out of pre-money
      2. price per share = pre-money / fully-diluted shares (post-topup)
      3. new shares = investment / price
    Returns {investor: shares_issued}.
    """
    if new_money <= 0:
        raise ValueError("new money must be positive")
    if abs(sum(investors.values()) - new_money) > 1e-6 * new_money:
        raise ValueError("investor allocations must sum to new_money")
    if pool_topup_pct is not None:
        apply_option_pool_topup(table, pool_topup_pct)
    pps = pre_money_val / table.total_shares
    issued: Dict[str, float] = {}
    for name, amount in investors.items():
        shares = amount / pps
        table.add_class(ShareClass(name, shares, invested=amount))
        issued[name] = shares
    return issued


def pro_rata_allocation(table: CapTable, investor: str, round_new_shares: float) -> float:
    """Shares an investor with pro-rata rights must buy to maintain ownership."""
    if not table.classes.get(investor, ShareClass(investor, 0)).pro_rata:
        return 0.0
    own = table.ownership(investor)
    return own * (table.total_shares + round_new_shares) - table.classes[investor].shares
