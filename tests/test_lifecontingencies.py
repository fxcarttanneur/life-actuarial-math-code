"""Tests for Chapters 2-10: every figure quoted in the manuscript is asserted here."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finmath import annuity_due as annuity_due_certain  # noqa: E402
from lifecon import Basis, standard_basis  # noqa: E402
from mortality import (  # noqa: E402
    LifeTable, complete_expectation_numeric, de_moivre_tpx, gompertz_tpx,
    integrate_mu, makeham_mu, makeham_tpx,
)
from multilife import JointLife  # noqa: E402
from pricing import (  # noqa: E402
    Loadings, commercial_level_premium, commercial_single_premium,
    deferred_annuity_premium, net_level_premium,
)
from reserves import Contract, policy_loan_balance  # noqa: E402
from toolbox import borrower_insurance, linear_interpolate, secant_solve  # noqa: E402

TH = LifeTable.builtin("tm2026")
TF = LifeTable.builtin("tf2026")
B = standard_basis("tm2026", 0.02)
BM = B.with_timing("mid")
J = JointLife(TH, TF, 0.02)
E = Contract.level_premium(B, 40, 20, death=100_000, survival=100_000)
T = Contract.level_premium(B, 40, 20, death=100_000)
A6 = dict(abs=5e-7)


# --- Chapter 2 -------------------------------------------------------------

def test_table_extract():
    assert TH.omega == 110 and TF.omega == 112
    assert TH.l(65) == 79_926 and round(TH.d(65)) == 1_374
    assert TH.q(65) == pytest.approx(0.017191, **A6)
    assert TF.q(65) == pytest.approx(0.006972, **A6)


def test_ch2_probabilities():
    assert TH.tpx(10, 40) == pytest.approx(0.962301, **A6)
    assert TH.tqx(20, 40) == pytest.approx(0.112391, **A6)
    assert TH.deferred_q(5, 10, 40) == pytest.approx(0.054862, **A6)
    assert TH.expected_survivors(25, 40, 10_000) == pytest.approx(8_293.7, abs=0.05)
    assert math.sqrt(10_000 * TH.tpx(25, 40) * TH.tqx(25, 40)) == pytest.approx(37.6, abs=0.05)
    assert TH.tpx(15, 65) == pytest.approx(0.592923, **A6)
    assert TH.deferred_q(15, 10, 65) == pytest.approx(0.411956, **A6)


def test_multiplication_rule():
    assert TH.tpx(25, 40) == pytest.approx(TH.tpx(10, 40) * TH.tpx(15, 50))


def test_force_and_expectation():
    assert TH.mu(65) == pytest.approx(0.016614, **A6)
    assert TH.complete_expectation(65) == pytest.approx(16.90, abs=0.005)
    assert TF.complete_expectation(65) == pytest.approx(21.42, abs=0.005)
    assert TH.complete_expectation(0) == pytest.approx(75.51, abs=0.005)


def test_analytical_laws():
    A, Bm, c = 0.00022, 2.7e-6, 1.124
    assert makeham_mu(60, A, Bm, c) == pytest.approx(0.003222, **A6)
    p = makeham_tpx(10, 60, A, Bm, c)
    assert p == pytest.approx(0.942549, **A6)
    assert integrate_mu(lambda s: makeham_mu(s, A, Bm, c), 60, 10) == pytest.approx(p, rel=1e-10)
    assert gompertz_tpx(10, 60, Bm, c) == pytest.approx(0.944625, **A6)
    assert complete_expectation_numeric(lambda t: makeham_tpx(t, 60, A, Bm, c), 80) == pytest.approx(27.21, abs=0.005)
    assert de_moivre_tpx(10, 40, 100) == pytest.approx(5 / 6)
    assert complete_expectation_numeric(lambda t: de_moivre_tpx(t, 40, 100), 60) == pytest.approx(30.0)


# --- Chapter 3 -------------------------------------------------------------

def test_commutation_and_annuities():
    assert B.D(40) == pytest.approx(43_644.60, abs=0.005)
    assert B.N(65) == pytest.approx(316_913.20, abs=0.005)
    assert B.nEx(25, 40) == pytest.approx(0.505529, **A6)
    assert B.annuity_due(65) == pytest.approx(14.363604, **A6)
    assert B.annuity_arrears(65) == pytest.approx(B.annuity_due(65) - 1)
    assert B.annuity_due(40, 25) == pytest.approx(18.856187, **A6)
    assert B.annuity_due(40, None, 25) == pytest.approx(7.261224, **A6)
    assert B.annuity_due(40) == pytest.approx(B.annuity_due(40, 25) + B.annuity_due(40, None, 25))
    assert B.annuity_due(40) == pytest.approx(26.117411, **A6)
    assert Basis(TF, 0.02).annuity_due(65) == pytest.approx(17.498181, **A6)


def test_annuity_vs_certain():
    assert (1 - 1.02 ** -16.40) / 0.02 == pytest.approx(13.865, abs=0.001)


def test_variable_and_fractional_annuities():
    assert B.increasing_annuity_due(65, 10) == pytest.approx(42.811737, **A6)
    direct = sum((k + 1) * B.nEx(k, 65) for k in range(10))
    assert B.increasing_annuity_due(65, 10) == pytest.approx(direct)
    assert B.geometric_annuity_due(65, 0.01) == pytest.approx(15.775511, **A6)
    assert B.annuity_fractional(65, 12) == pytest.approx(13.905271, **A6)
    assert B.annuity_fractional(40, 12, None, 25) == pytest.approx(7.029523, **A6)
    assert round(12_000 * B.annuity_fractional(40, 12, None, 25), 2) == 84_354.28


# --- Chapter 4 -------------------------------------------------------------

def test_insurances():
    assert B.whole_life(40) == pytest.approx(0.487894, **A6)
    assert B.term(40, 20) == pytest.approx(0.087895, **A6)
    assert B.endowment(40, 20) == pytest.approx(0.685230, **A6)
    assert B.increasing_term(40, 20) == pytest.approx(1.064981, **A6)
    assert B.increasing_term(40) == pytest.approx(16.808450, **A6)
    assert BM.term(40, 20) == pytest.approx(0.088769, **A6)
    assert BM.term(40, 20) / B.term(40, 20) == pytest.approx(1.02 ** 0.5)


@pytest.mark.parametrize("x", [20, 40, 65, 90])
def test_key_relation(x):
    d = 0.02 / 1.02
    assert B.whole_life(x) == pytest.approx(1 - d * B.annuity_due(x))
    assert B.endowment(x, 10) == pytest.approx(1 - d * B.annuity_due(x, 10))


# --- Chapters 5 and 6 --------------------------------------------------------

def test_joint_and_last_survivor():
    assert J.tpxy(10, 65, 62) == pytest.approx(0.704154, **A6)
    assert J.tp_last(10, 65, 62) == pytest.approx(0.981068, **A6)
    assert J.annuity_due_joint(65, 62) == pytest.approx(J.N(65, 62) / J.D(65, 62))
    assert J.annuity_due_joint(65, 62) == pytest.approx(12.835419, **A6)
    assert J.annuity_due_last(65, 62) == pytest.approx(20.652061, **A6)
    assert J.A_joint(65, 62) == pytest.approx(0.748325, **A6)
    assert J.A_last(65, 62) == pytest.approx(0.595058, **A6)
    assert J.A_last(65, 62) == pytest.approx(1 - 0.02 / 1.02 * J.annuity_due_last(65, 62))
    assert J.annuity_due_single(TH, 65) == pytest.approx(B.annuity_due(65))


def test_last_survivor_is_not_markov():
    assert J.tp_last(20, 65, 62) == pytest.approx(0.819977, **A6)
    assert J.tp_last(10, 65, 62) * J.tp_last(10, 75, 72) == pytest.approx(0.868506, **A6)
    assert J.tpxy(20, 65, 62) == pytest.approx(J.tpxy(10, 65, 62) * J.tpxy(10, 75, 72))


def test_contingent_and_reversionary():
    a1 = J.A_contingent_first(65, 62)
    assert a1 == pytest.approx(0.545568, **A6)
    a1_rev = JointLife(TF, TH, 0.02).A_contingent_first(62, 65)
    assert a1_rev == pytest.approx(0.202757, **A6)
    assert a1 + a1_rev == pytest.approx(J.A_joint(65, 62), rel=1e-9)
    assert J.A_contingent_second(65, 62) == pytest.approx(0.172793, **A6)
    assert J.reversionary_annuity_due(65, 62) == pytest.approx(6.288457, **A6)
    assert J.joint_survivor_annuity_due(65, 62, 0.6) == pytest.approx(18.136678, **A6)
    assert J.joint_survivor_annuity_due(65, 62, 1.0) == pytest.approx(J.annuity_due_last(65, 62))
    assert round(20_000 * J.annuity_due_single(TH, 65) / J.joint_survivor_annuity_due(65, 62, 0.6), 2) == 15_839.29


# --- Chapter 7 -------------------------------------------------------------

L = Loadings(f=0.03, g1=0.002, g2=0.002, alpha=0.03, r=0.01)


def test_tariffs():
    A = B.endowment(40, 20)
    assert round(net_level_premium(100_000 * A, B, 40, 20), 2) == 4_268.48
    assert round(commercial_level_premium(B, 40, 100_000, A, 20, 20, L), 2) == 4_843.34
    assert round(commercial_single_premium(B, 40, 100_000, A, 20, L), 2) == 77_751.45
    assert round(commercial_level_premium(B, 40, 100_000, A, 20, 10, L), 2) == 8_603.69
    assert round(commercial_level_premium(B, 40, 100_000, B.term(40, 20), 20, 20, L), 2) == 968.94


def test_return_of_premium():
    p0 = deferred_annuity_premium(B, 40, 25, 10_000)
    p1 = deferred_annuity_premium(B, 40, 25, 10_000, return_of_premium=True)
    assert round(p0, 2) == 3_850.84 and round(p1, 2) == 4_286.37
    assert (p1 - p0) * B.annuity_due(40, 25) == pytest.approx(p1 * B.increasing_term(40, 25))


# --- Chapter 8 -------------------------------------------------------------

@pytest.mark.parametrize("t", range(0, 21))
def test_prospective_equals_retrospective(t):
    for c in (E, T):
        assert c.prospective_reserve(t) == pytest.approx(c.retrospective_reserve(t), abs=1e-6)


def test_reserve_values_and_recursion():
    assert round(E.prospective_reserve(10), 2) == 44_635.21
    assert round(T.prospective_reserve(10), 2) == 2_039.13
    a = B.annuity_due(40, 20)
    assert E.prospective_reserve(10) == pytest.approx(100_000 * (1 - B.annuity_due(50, 10) / a))
    for t in range(20):
        for c in (E, T):
            lhs = (c.prospective_reserve(t) + c.premium[t]) * 1.02
            rhs = TH.q(40 + t) * c.death[t] + TH.p(40 + t) * c.prospective_reserve(t + 1)
            assert lhs == pytest.approx(rhs, abs=1e-6)
            s, r = c.premium_split(t)
            assert s + r == pytest.approx(c.premium[t], abs=1e-8)


def test_zillmer_and_pgg():
    assert round(E.zillmer_reserve(1, 3_000), 2) == 1_250.83
    assert E.zillmer_reserve(0, 3_000) == pytest.approx(-3_000, abs=1e-6)
    E10 = Contract.level_premium(B, 40, 20, death=100_000, survival=100_000, m=10)
    assert round(E10.premium[0], 2) == 7_582.50
    assert E10.management_reserve(0, 100_000, 0.003, 0.002, 10) == pytest.approx(0, abs=1e-6)
    assert round(E10.management_reserve(10, 100_000, 0.003, 0.002, 10), 2) == 1_777.57


# --- Chapter 9 -------------------------------------------------------------

def test_alterations():
    assert round(E.surrender_value(10, 0.05), 2) == 42_403.45
    assert E.reduced_benefit_factor(10) == pytest.approx(0.540556, **A6)
    assert E.reduced_benefit_factor(10, 0.002) == pytest.approx(0.529164, **A6)
    assert T.reduced_benefit_factor(10) == pytest.approx(0.295295, **A6)
    assert round(policy_loan_balance(0.8 * E.surrender_value(10, 0.05), 0.04, 3), 2) == 38_158.49
    lhs = E.prospective_reserve(10) + E.pvfp(10)
    new_s = (lhs - 200_000 * B.term(50, 10)) / B.nEx(10, 50)
    assert round(new_s, 2) == 90_874.01


# --- Chapter 10 ------------------------------------------------------------

def test_toolbox():
    bi = borrower_insurance(B, 40, 200_000, 0.05, 20)
    assert round(bi["single_premium"], 2) == 9_270.26
    assert round(bi["level_premium"], 2) == 577.47
    assert bi["rate_on_initial_capital"] == pytest.approx(0.002887, abs=5e-7)
    assert bi["rate_on_outstanding_capital"] == pytest.approx(0.004477, abs=5e-7)
    approx = linear_interpolate(0.02, B.annuity_due(65), 0.03, B.with_rate(0.03).annuity_due(65), 0.025)
    assert approx == pytest.approx(13.764756, **A6)
    assert B.with_rate(0.025).annuity_due(65) == pytest.approx(13.742208, **A6)
    j = secant_solve(lambda r: 16_048.52 * annuity_due_certain(20, r) - 200_000, 0.05, 0.06)
    assert j == pytest.approx(0.056585, abs=1e-6)
