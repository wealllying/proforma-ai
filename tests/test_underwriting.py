# tests/test_underwriting.py
import math
import numpy as np
from app import compute_amort_schedule, robust_irr, apply_periodic_waterfall, settle_final_distribution

def test_amort_schedule_zero_bal_after_amort():
    loan = 10_000_000
    rate = 0.05
    amort_years = 10
    balances, interests, principals, payments = compute_amort_schedule(loan, rate, amort_years, amort_years)
    assert balances[0] == loan
    assert abs(sum(principals) - loan) < 1.0  # principal sum ≈ loan

def test_robust_irr_simple():
    cfs = [-1000, 0, 0, 2000]
    irr = robust_irr(cfs)
    # expected IRR ~ 0.1487
    assert abs(irr - 0.1487) < 1e-3

def test_waterfall_and_settlement_simple():
    # LP invested 100, gets 200 final residual, pref 0, no tiers
    lp_so_far = [-100.0]
    gp_so_far = [-0.0]
    lp_add, gp_add = settle_final_distribution(lp_so_far, gp_so_far, 200.0, 100.0, None)
    # default split 80/20
    assert abs(lp_add - 160.0) < 1e-6
    assert abs(gp_add - 40.0) < 1e-6

def test_periodic_waterfall_pref_roc():
    dist = 100.0
    lp_roc = 80.0
    lp_pref = 5.0
    equity_lp = 80.0
    lp_paid, gp_paid, lp_roc_rem, lp_pref_rem, residual = apply_periodic_waterfall(dist, lp_roc, lp_pref, equity_lp, 0.05, 0.0)
    # first, ROC should be paid 80 => remaining 20, pref 5 paid => remaining 15 resid
    assert abs(lp_paid - 85.0) < 1e-6
    assert abs(residual - 15.0) < 1e-6
