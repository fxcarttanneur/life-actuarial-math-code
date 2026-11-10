"""Unit tests for src/finmath.py — every figure quoted in Chapter 1 is checked here."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finmath import (  # noqa: E402
    accumulated_annuity_due,
    amortization_schedule,
    annuity_arrears,
    annuity_due,
    annuity_fractional,
    commercial_coefficient,
    deferred_annuity_arrears,
    discount_factor,
    level_payment,
    periodic_rate,
    perpetuity_arrears,
    perpetuity_due,
    solve_rate,
)


# --- Section 1.1 -----------------------------------------------------------

def test_capitalization_bond_single_premium():
    assert round(100_000 * discount_factor(0.035) ** 8, 2) == 75_941.16


def test_periodic_rate_equivalence():
    i, k = 0.035, 12
    assert (1 + periodic_rate(i, k)) ** k == pytest.approx(1 + i, abs=1e-14)
    assert periodic_rate(i, k) == pytest.approx(0.00287090, abs=1e-8)


# --- Section 1.2 -----------------------------------------------------------

def test_annuity_values_at_5pct():
    assert annuity_arrears(20, 0.05) == pytest.approx(12.462210, abs=1e-6)
    assert annuity_due(20, 0.05) == pytest.approx(13.085321, abs=1e-6)


@pytest.mark.parametrize("n", [1, 5, 20, 40])
@pytest.mark.parametrize("i", [0.01, 0.035, 0.05, 0.10])
def test_annuity_due_identities(n, i):
    v = discount_factor(i)
    ad = annuity_due(n, i)
    assert ad == pytest.approx(sum(v**t for t in range(n)))
    assert ad == pytest.approx((1 - v**n) / (1 - v))
    assert ad == pytest.approx(1 + annuity_arrears(n - 1, i))


def test_perpetuities_and_deferred():
    assert perpetuity_arrears(0.05) == pytest.approx(20.0)
    assert perpetuity_due(0.05) == pytest.approx(21.0)
    assert annuity_arrears(2000, 0.05) == pytest.approx(perpetuity_arrears(0.05))
    d = deferred_annuity_arrears(5, 10, 0.05)
    assert d == pytest.approx(6.050181, abs=1e-6)
    assert d == pytest.approx(annuity_arrears(15, 0.05) - annuity_arrears(5, 0.05))


def test_zero_rate():
    assert annuity_arrears(10, 0.0) == 10.0
    assert annuity_fractional(10, 0.0, 12) == 10.0


# --- Section 1.3 -----------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 4, 12])
def test_fractional_exact_matches_direct_sum(k):
    n, i = 20, 0.05
    v = discount_factor(i)
    direct = sum(v ** (j / k) for j in range(1, n * k + 1)) / k
    assert annuity_fractional(n, i, k) == pytest.approx(direct, rel=1e-12)
    direct_due = sum(v ** (j / k) for j in range(n * k)) / k
    assert annuity_fractional(n, i, k, due=True) == pytest.approx(direct_due, rel=1e-12)
    assert annuity_fractional(n, i, k) == pytest.approx(
        commercial_coefficient(i, k) * annuity_arrears(n, i), rel=1e-12
    )


def test_commercial_coefficient_above_one():
    for k in (2, 4, 12):
        assert commercial_coefficient(0.05, k) > 1
    assert commercial_coefficient(0.05, 12) == pytest.approx(1.022715, abs=1e-6)


@pytest.mark.parametrize("due", [False, True])
@pytest.mark.parametrize("k", [2, 4, 12])
@pytest.mark.parametrize("n", [1, 10, 20, 50])
@pytest.mark.parametrize("i", [0.005, 0.02, 0.05, 0.10])
def test_fractional_approximation_error_bounded(i, n, k, due):
    """0 <= approx - exact <= (delta^2 / 8) * ä_n  (chord above a convex curve)."""
    exact = annuity_fractional(n, i, k, due=due)
    approx = annuity_fractional(n, i, k, due=due, approx=True)
    bound = math.log1p(i) ** 2 / 8 * annuity_due(n, i)
    err = approx - exact
    assert -1e-12 <= err <= bound


def test_monthly_premium_example():
    ratio = annuity_due(20, 0.05) / annuity_fractional(20, 0.05, 12, due=True)
    assert round(1000 * ratio / 12, 2) == 85.21


# --- Section 1.4 -----------------------------------------------------------

def test_loan_level_payment():
    assert round(200_000 / annuity_arrears(20, 0.05), 2) == 16_048.52
    assert round(level_payment(200_000, 0.05, 20), 2) == 16_048.52


def test_amortization_schedule_properties():
    df = amortization_schedule(200_000, 0.05, 20)
    R = level_payment(200_000, 0.05, 20)
    assert len(df) == 20
    assert df["principal_repaid"].sum() == pytest.approx(200_000)
    assert df["interest"].sum() == pytest.approx(120_970.35, abs=0.01)
    assert df["closing_balance"].iloc[-1] == 0.0
    assert df["interest"].iloc[0] == pytest.approx(10_000.0)
    assert round(df["principal_repaid"].iloc[0], 2) == 6_048.52
    for t, row in df.iterrows():
        year = row["year"]
        # prospective balance and K_t = R v^(n-t+1)
        assert row["opening_balance"] == pytest.approx(R * annuity_arrears(20 - year + 1, 0.05))
        assert row["principal_repaid"] == pytest.approx(R * discount_factor(0.05) ** (20 - year + 1))


# --- Exercises -------------------------------------------------------------

def test_exercise_1():
    assert round(100_000 / accumulated_annuity_due(8, 0.035), 2) == 10_674.07


def test_exercise_2():
    i12 = periodic_rate(0.05, 12)
    assert round(level_payment(200_000, i12, 240), 2) == 1_307.67
    assert round(level_payment(200_000, 0.05 / 12, 240), 2) == 1_319.91
    assert (1 + 0.05 / 12) ** 12 - 1 == pytest.approx(0.051162, abs=1e-6)


def test_exercise_3():
    R = level_payment(200_000, 0.05, 20)
    j = solve_rate(200_000, R, 20, due=True)
    assert j == pytest.approx(0.056585, abs=1e-6)
    assert j == pytest.approx(solve_rate(200_000 - R, R, 19), abs=1e-9)
