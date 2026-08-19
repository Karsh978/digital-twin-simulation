"""Valuation mechanics (§2.3): pre/post-money, price per share, SAFE/note
conversion with caps and discounts, step-up multiples. Pure functions."""

from __future__ import annotations


def post_money(pre_money_val: float, new_money: float) -> float:
    if pre_money_val < 0 or new_money < 0:
        raise ValueError("valuations and investment must be non-negative")
    return pre_money_val + new_money


def pre_money(post_money_val: float, new_money: float) -> float:
    pm = post_money_val - new_money
    if pm < 0:
        raise ValueError("new money exceeds post-money valuation")
    return pm


def price_per_share(pre_money_val: float, fully_diluted_shares: float) -> float:
    if fully_diluted_shares <= 0:
        raise ValueError("fully diluted share count must be positive")
    return pre_money_val / fully_diluted_shares


def valuation_step_up(current_post: float, previous_post: float) -> float:
    """Step-up multiple between rounds (Layer 6 KPI, §2.6)."""
    if previous_post <= 0:
        raise ValueError("previous post-money must be positive")
    return current_post / previous_post


def safe_conversion_shares(
    investment: float,
    valuation_cap: float | None = None,
    discount: float | None = None,
    round_price_per_share: float = 0.0,
    fully_diluted_shares: float = 0.0,
) -> float:
    """Shares issued to a SAFE/note holder on conversion.

    Conversion price is the MORE investor-favorable of:
      - cap price   = valuation_cap / fully_diluted_shares
      - discount price = round_price * (1 - discount)
    At least one of cap/discount must be set.
    """
    if round_price_per_share <= 0 or fully_diluted_shares <= 0:
        raise ValueError("round price and share count must be positive")
    if valuation_cap is None and discount is None:
        raise ValueError("SAFE needs a valuation cap, a discount, or both")
    candidates = []
    if valuation_cap is not None:
        candidates.append(valuation_cap / fully_diluted_shares)
    if discount is not None:
        if not 0 < discount < 1:
            raise ValueError("discount must be in (0, 1)")
        candidates.append(round_price_per_share * (1 - discount))
    conversion_price = min(candidates)
    return investment / conversion_price
