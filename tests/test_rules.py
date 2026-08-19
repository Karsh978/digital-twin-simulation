"""Layer 3 unit tests — the rule engine is ground truth (§1.2). In production
these are validated against 10 real historical U.S. rounds (§2.3); here we
test against hand-computed canonical cases."""

import math

import pytest

from digital_twin.rules.cap_table import (CapTable, ShareClass,
                                          apply_option_pool_topup, issue_round,
                                          pro_rata_allocation)
from digital_twin.rules.fund import CheckSizeBand, Fund, Stage
from digital_twin.rules.liquidation import LiquidationPreference, payout_waterfall
from digital_twin.rules.runway import RunwayModel
from digital_twin.rules.valuation import (post_money, pre_money, price_per_share,
                                          safe_conversion_shares, valuation_step_up)


# -- valuation ---------------------------------------------------------------

def test_pre_post_money():
    assert post_money(8e6, 2e6) == 10e6
    assert pre_money(10e6, 2e6) == 8e6
    with pytest.raises(ValueError):
        pre_money(1e6, 2e6)


def test_price_per_share():
    assert price_per_share(10e6, 5e6) == 2.0


def test_step_up():
    assert valuation_step_up(30e6, 10e6) == 3.0


def test_safe_conversion_cap_beats_discount():
    # cap $5M on 5M shares = $1.00; round price $2.00 with 20% discount = $1.60
    # cap is more investor-favorable -> converts at $1.00
    shares = safe_conversion_shares(500_000, valuation_cap=5e6, discount=0.20,
                                    round_price_per_share=2.0,
                                    fully_diluted_shares=5e6)
    assert shares == pytest.approx(500_000)


def test_safe_conversion_discount_beats_cap():
    # cap $40M on 5M shares = $8.00; discount price = $1.60 -> discount wins
    shares = safe_conversion_shares(500_000, valuation_cap=40e6, discount=0.20,
                                    round_price_per_share=2.0,
                                    fully_diluted_shares=5e6)
    assert shares == pytest.approx(312_500)


def test_safe_requires_cap_or_discount():
    with pytest.raises(ValueError):
        safe_conversion_shares(1e6, round_price_per_share=2.0,
                               fully_diluted_shares=5e6)


# -- cap table -----------------------------------------------------------------

def _seed_table() -> CapTable:
    ct = CapTable()
    ct.add_class(ShareClass("founders", 8_000_000))
    ct.add_class(ShareClass("option_pool", 2_000_000))
    return ct


def test_issue_round_dilution():
    ct = _seed_table()
    # $2M on $8M pre -> post $10M; new investor owns 20%
    issue_round(ct, "seed", pre_money_val=8e6, new_money=2e6,
                investors={"seed_fund": 2e6})
    assert ct.ownership("seed_fund") == pytest.approx(0.20)
    assert ct.ownership("founders") == pytest.approx(0.64)


def test_option_pool_topup_comes_out_of_premoney():
    ct = _seed_table()  # pool currently 20%
    added = apply_option_pool_topup(ct, 0.20)
    assert added == 0.0  # already at target
    ct2 = CapTable()
    ct2.add_class(ShareClass("founders", 9_000_000))
    ct2.add_class(ShareClass("option_pool", 1_000_000))  # 10%
    apply_option_pool_topup(ct2, 0.20)
    assert ct2.ownership("option_pool") == pytest.approx(0.20)


def test_round_allocations_must_sum():
    ct = _seed_table()
    with pytest.raises(ValueError):
        issue_round(ct, "seed", 8e6, 2e6, investors={"a": 1e6})


def test_pro_rata():
    ct = _seed_table()
    issue_round(ct, "seed", 8e6, 2e6, investors={"seed_fund": 2e6})
    ct.classes["seed_fund"].pro_rata = True
    needed = pro_rata_allocation(ct, "seed_fund", round_new_shares=1_000_000)
    # maintains 20% ownership post-round
    assert needed / (ct.total_shares + 1_000_000) + \
        ct.ownership("seed_fund") * ct.total_shares / (ct.total_shares + 1_000_000) \
        == pytest.approx(0.20, abs=1e-9)


# -- liquidation -----------------------------------------------------------------

def test_nonparticipating_preference_small_exit():
    # 1x non-participating, $5M invested, exit $6M, investor holds 50% ->
    # preference ($5M) beats as-converted ($3M)
    prefs = [LiquidationPreference("series_a", invested=5e6)]
    payouts = payout_waterfall(6e6, prefs,
                                 common_shares={"founders": 5e6},
                                 preferred_shares={"series_a": 5e6})
    assert payouts["series_a"] == pytest.approx(5e6)
    assert payouts["founders"] == pytest.approx(1e6)


def test_nonparticipating_converts_on_large_exit():
    # exit $100M; as-converted 50% = $50M beats $5M preference
    prefs = [LiquidationPreference("series_a", invested=5e6)]
    payouts = payout_waterfall(100e6, prefs,
                                 common_shares={"founders": 5e6},
                                 preferred_shares={"series_a": 5e6})
    assert payouts["series_a"] == pytest.approx(50e6)
    assert payouts["founders"] == pytest.approx(50e6)


def test_seniority_stack():
    prefs = [LiquidationPreference("seed", invested=2e6, seniority=0),
             LiquidationPreference("series_a", invested=8e6, seniority=1)]
    payouts = payout_waterfall(9e6, prefs,
                                 common_shares={"founders": 10e6},
                                 preferred_shares={"seed": 2e6, "series_a": 8e6})
    # series_a (senior) takes $8M first; seed takes remaining $1M; founders $0
    assert payouts["series_a"] == pytest.approx(8e6)
    assert payouts["seed"] == pytest.approx(1e6)
    assert payouts["founders"] == pytest.approx(0.0)


# -- fund mechanics ---------------------------------------------------------------

def test_fund_bands_and_reserves():
    fund = Fund("Test Fund", size=100e6, reserve_ratio=0.5,
                check_bands={Stage.SEED: CheckSizeBand(Stage.SEED, 0.5e6, 3e6)})
    assert fund.write_initial("co1", Stage.SEED, 2e6) == 2e6
    assert fund.write_initial("co2", Stage.SEED, 5e6) == 0.0   # outside band
    assert fund.can_follow_on("co1", 1e6)
    assert not fund.can_follow_on("co2", 1e6)                  # not in portfolio
    assert fund.write_follow_on("co1", 1e6) == 1e6
    assert fund.dry_powder() == pytest.approx(100e6 - 3e6)


def test_fund_budget_exhaustion():
    fund = Fund("Small", size=10e6, reserve_ratio=0.5,
                check_bands={Stage.SEED: CheckSizeBand(Stage.SEED, 0.1e6, 10e6)})
    assert fund.write_initial("a", Stage.SEED, 4e6) == 4e6
    assert fund.write_initial("b", Stage.SEED, 2e6) == 0.0     # only 1e6 left


# -- runway ---------------------------------------------------------------------

def test_runway_burn_and_death():
    r = RunwayModel(cash=1e6, headcount=10,
                    avg_monthly_cost_per_head=18_000, other_monthly_spend=20_000)
    assert r.monthly_burn == pytest.approx(200_000)
    weeks = r.runway_weeks()
    assert weeks == pytest.approx(1e6 / (200_000 / (52 / 12)))
    assert r.needs_raise_within(26)
    r.tick(int(weeks) + 2)
    assert r.is_out_of_runway()


def test_runway_revenue_extends():
    r = RunwayModel(cash=1e6, headcount=10, monthly_revenue=100_000)
    r2 = RunwayModel(cash=1e6, headcount=10, monthly_revenue=0.0)
    assert r.runway_weeks() > r2.runway_weeks()
