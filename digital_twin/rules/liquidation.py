"""Liquidation preferences (§2.3): 1x non-participating default, configurable
multiples and participation. Computes the exit payout waterfall."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class LiquidationPreference:
    holder: str
    invested: float
    multiple: float = 1.0              # 1x default
    participating: bool = False        # non-participating default
    seniority: int = 0                 # higher = paid first (standard: later rounds senior)


def payout_waterfall(
    exit_value: float,
    preferences: List[LiquidationPreference],
    common_shares: Dict[str, float],
    preferred_shares: Dict[str, float],
) -> Dict[str, float]:
    """Compute exit proceeds per holder.

    Non-participating preferred takes the GREATER of (preference) or
    (as-converted common value). Participating preferred takes preference AND
    shares the remainder pro-rata. Seniority orders the preference stack.
    """
    if exit_value < 0:
        raise ValueError("exit value must be non-negative")
    payouts: Dict[str, float] = {h: 0.0 for h in
                                 list(common_shares) + [p.holder for p in preferences]}
    remaining = exit_value
    total_shares = sum(common_shares.values()) + sum(preferred_shares.values())
    if total_shares <= 0:
        raise ValueError("no shares outstanding")

    as_converted_ps = exit_value / total_shares  # per-share value if all convert

    converted: List[str] = []
    for pref in sorted(preferences, key=lambda p: -p.seniority):
        preference_amount = pref.invested * pref.multiple
        as_converted = preferred_shares.get(pref.holder, 0.0) * as_converted_ps
        if not pref.participating and as_converted >= preference_amount:
            converted.append(pref.holder)  # better to convert; paid in common split below
            continue
        take = min(remaining, preference_amount)
        payouts[pref.holder] += take
        remaining -= take
        if pref.participating:
            # share remaining pro-rata alongside common (capped at what's left)
            pass  # handled in the common split below via their shares

    # Split the remainder among common + converted (and participating) holders.
    residual_shares = dict(common_shares)
    for h in converted:
        residual_shares[h] = residual_shares.get(h, 0.0) + preferred_shares.get(h, 0.0)
    for pref in preferences:
        if pref.participating:
            residual_shares[pref.holder] = residual_shares.get(pref.holder, 0.0) + \
                preferred_shares.get(pref.holder, 0.0)
    rs_total = sum(residual_shares.values())
    if rs_total > 0 and remaining > 0:
        for h, sh in residual_shares.items():
            payouts[h] = payouts.get(h, 0.0) + remaining * sh / rs_total
    return payouts
