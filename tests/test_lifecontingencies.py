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

TH = LifeTable.builtin("tgh05")
TF = LifeTable.builtin("tgf05")
B = standard_basis("tgh05", 0.02)
BM = B.with_timing("mid")
J = JointLife(TH, TF, 0.02)
E = Contract.level_premium(B, 40, 20, death=100_000, survival=100_000)
T = Contract.level_premium(B, 40, 20, death=100_000)
A6 = dict(abs=5e-7)


# --- Chapter 2 -------------------------------------------------------------

def test_table_extract():
    assert TH.omega == 120 and TF.omega == 120
    assert TH.l(65) == 97_144 and round(TH.d(65)) == 152
    assert TH.q(65) == pytest.approx(0.001565, **A6)
    assert TF.q(65) == pytest.approx(0.001400, **A6)


def test_ch2_probabilities():
    assert TH.tpx(10, 40) == pytest.approx(0.996221, **A6)
    assert TH.tqx(20, 40) == pytest.approx(0.012227, **A6)
    assert TH.deferred_q(5, 10, 40) == pytest.approx(0.005467, **A6)
    assert TH.expected_survivors(25, 40, 10_000) == pytest.approx(9_816.4, abs=0.05)
    assert math.sqrt(10_000 * TH.tpx(25, 40) * TH.tqx(25, 40)) == pytest.approx(13.4, abs=0.05)
    assert TH.tpx(15, 65) == pytest.approx(0.946492, **A6)
    assert TH.deferred_q(15, 10, 65) == pytest.approx(0.168852, **A6)


def test_multiplication_rule():
    assert TH.tpx(25, 40) == pytest.approx(TH.tpx(10, 40) * TH.tpx(15, 50))


def test_force_and_expectation():
    assert TH.mu(65) == pytest.approx(0.001498, **A6)
    assert TH.complete_expectation(65) == pytest.approx(31.43, abs=0.005)
    assert TF.complete_expectation(65) == pytest.approx(35.19, abs=0.005)
    assert TH.complete_expectation(0) == pytest.approx(94.83, abs=0.005)


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
    assert B.D(40) == pytest.approx(44_818.49, abs=0.005)
    assert B.N(65) == pytest.approx(627_775.06, abs=0.005)
    assert B.nEx(25, 40) == pytest.approx(0.598339, **A6)
    assert B.annuity_due(65) == pytest.approx(23.409883, **A6)
    assert B.annuity_arrears(65) == pytest.approx(B.annuity_due(65) - 1)
    assert B.annuity_due(40, 25) == pytest.approx(19.802937, **A6)
    assert B.annuity_due(40, None, 25) == pytest.approx(14.007056, **A6)
    assert B.annuity_due(40) == pytest.approx(B.annuity_due(40, 25) + B.annuity_due(40, None, 25))
    assert B.annuity_due(40) == pytest.approx(33.809992, **A6)
    assert Basis(TF, 0.02).annuity_due(65) == pytest.approx(25.421944, **A6)


def test_annuity_vs_certain():
    assert (1 - 1.02 ** -30.93) / 0.02 == pytest.approx(22.900, abs=0.001)


def test_variable_and_fractional_annuities():
    assert B.increasing_annuity_due(65, 10) == pytest.approx(48.261017, **A6)
    direct = sum((k + 1) * B.nEx(k, 65) for k in range(10))
    assert B.increasing_annuity_due(65, 10) == pytest.approx(direct)
    assert B.geometric_annuity_due(65, 0.01) == pytest.approx(27.202226, **A6)
    assert B.annuity_fractional(65, 12) == pytest.approx(22.951550, **A6)
    assert B.annuity_fractional(40, 12, None, 25) == pytest.approx(13.732817, **A6)
    assert round(12_000 * B.annuity_fractional(40, 12, None, 25), 2) == 164_793.80


# --- Chapter 4 -------------------------------------------------------------

def test_insurances():
    assert B.whole_life(40) == pytest.approx(0.337059, **A6)
    assert B.term(40, 20) == pytest.approx(0.009485, **A6)
    assert B.endowment(40, 20) == pytest.approx(0.674228, **A6)
    assert B.increasing_term(40, 20) == pytest.approx(0.118758, **A6)
    assert B.increasing_term(40) == pytest.approx(18.043157, **A6)
    assert BM.term(40, 20) == pytest.approx(0.009579, **A6)
    assert BM.term(40, 20) / B.term(40, 20) == pytest.approx(1.02 ** 0.5)


@pytest.mark.parametrize("x", [20, 40, 65, 90])
def test_key_relation(x):
    d = 0.02 / 1.02
    assert B.whole_life(x) == pytest.approx(1 - d * B.annuity_due(x))
    assert B.endowment(x, 10) == pytest.approx(1 - d * B.annuity_due(x, 10))


# --- Chapters 5 and 6 --------------------------------------------------------

def test_joint_and_last_survivor():
    assert J.tpxy(10, 65, 62) == pytest.approx(0.958611, **A6)
    assert J.tp_last(10, 65, 62) == pytest.approx(0.999599, **A6)
    assert J.annuity_due_joint(65, 62) == pytest.approx(J.N(65, 62) / J.D(65, 62))
    assert J.annuity_due_joint(65, 62) == pytest.approx(22.030075, **A6)
    assert J.annuity_due_last(65, 62) == pytest.approx(28.178777, **A6)
    assert J.A_joint(65, 62) == pytest.approx(0.568038, **A6)
    assert J.A_last(65, 62) == pytest.approx(0.447475, **A6)
    assert J.A_last(65, 62) == pytest.approx(1 - 0.02 / 1.02 * J.annuity_due_last(65, 62))
    assert J.annuity_due_single(TH, 65) == pytest.approx(B.annuity_due(65))


def test_last_survivor_is_not_markov():
    assert J.tp_last(20, 65, 62) == pytest.approx(0.995481, **A6)
    assert J.tp_last(10, 65, 62) * J.tp_last(10, 75, 72) == pytest.approx(0.997327, **A6)
    assert J.tpxy(20, 65, 62) == pytest.approx(J.tpxy(10, 65, 62) * J.tpxy(10, 75, 72))


def test_contingent_and_reversionary():
    a1 = J.A_contingent_first(65, 62)
    assert a1 == pytest.approx(0.404262, **A6)
    a1_rev = JointLife(TF, TH, 0.02).A_contingent_first(62, 65)
    assert a1_rev == pytest.approx(0.163776, **A6)
    assert a1 + a1_rev == pytest.approx(J.A_joint(65, 62), rel=1e-9)
    assert J.A_contingent_second(65, 62) == pytest.approx(0.136721, **A6)
    assert J.reversionary_annuity_due(65, 62) == pytest.approx(4.768894, **A6)
    assert J.joint_survivor_annuity_due(65, 62, 0.6) == pytest.approx(26.271219, **A6)
    assert J.joint_survivor_annuity_due(65, 62, 1.0) == pytest.approx(J.annuity_due_last(65, 62))
    assert round(20_000 * J.annuity_due_single(TH, 65) / J.joint_survivor_annuity_due(65, 62, 0.6), 2) == 17_821.70


# --- Chapter 7 -------------------------------------------------------------

L = Loadings(f=0.03, g1=0.002, g2=0.002, alpha=0.03, r=0.01)


def test_tariffs():
    A = B.endowment(40, 20)
    assert round(net_level_premium(100_000 * A, B, 40, 20), 2) == 4_058.10
    assert round(commercial_level_premium(B, 40, 100_000, A, 20, 20, L), 2) == 4_617.78
    assert round(commercial_single_premium(B, 40, 100_000, A, 20, L), 2) == 76_721.53
    assert round(commercial_level_premium(B, 40, 100_000, A, 20, 10, L), 2) == 8_385.39
    assert round(commercial_level_premium(B, 40, 100_000, B.term(40, 20), 20, 20, L), 2) == 451.78


def test_return_of_premium():
    p0 = deferred_annuity_premium(B, 40, 25, 10_000)
    p1 = deferred_annuity_premium(B, 40, 25, 10_000, return_of_premium=True)
    assert round(p0, 2) == 7_073.22 and round(p1, 2) == 7_148.43
    assert (p1 - p0) * B.annuity_due(40, 25) == pytest.approx(p1 * B.increasing_term(40, 25))


# --- Chapter 8 -------------------------------------------------------------

@pytest.mark.parametrize("t", range(0, 21))
def test_prospective_equals_retrospective(t):
    for c in (E, T):
        assert c.prospective_reserve(t) == pytest.approx(c.retrospective_reserve(t), abs=1e-6)


def test_reserve_values_and_recursion():
    assert round(E.prospective_reserve(10), 2) == 45_021.96
    assert round(T.prospective_reserve(10), 2) == 229.03
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
    assert round(E.zillmer_reserve(1, 3_000), 2) == 1_238.49
    assert E.zillmer_reserve(0, 3_000) == pytest.approx(-3_000, abs=1e-6)
    E10 = Contract.level_premium(B, 40, 20, death=100_000, survival=100_000, m=10)
    assert round(E10.premium[0], 2) == 7_369.07
    assert E10.management_reserve(0, 100_000, 0.003, 0.002, 10) == pytest.approx(0, abs=1e-6)
    assert round(E10.management_reserve(10, 100_000, 0.003, 0.002, 10), 2) == 1_826.85


# --- Chapter 9 -------------------------------------------------------------

def test_alterations():
    assert round(E.surrender_value(10, 0.05), 2) == 42_770.87
    assert E.reduced_benefit_factor(10) == pytest.approx(0.548448, **A6)
    assert E.reduced_benefit_factor(10, 0.002) == pytest.approx(0.536509, **A6)
    assert T.reduced_benefit_factor(10) == pytest.approx(0.305176, **A6)
    assert round(policy_loan_balance(0.8 * E.surrender_value(10, 0.05), 0.04, 3), 2) == 38_489.13
    lhs = E.prospective_reserve(10) + E.pvfp(10)
    new_s = (lhs - 200_000 * B.term(50, 10)) / B.nEx(10, 50)
    assert round(new_s, 2) == 99_077.33


# --- Chapter 10 ------------------------------------------------------------

def test_toolbox():
    bi = borrower_insurance(B, 40, 200_000, 0.05, 20)
    assert round(bi["single_premium"], 2) == 958.91
    assert round(bi["level_premium"], 2) == 57.72
    assert bi["rate_on_initial_capital"] == pytest.approx(0.000289, abs=5e-7)
    assert bi["rate_on_outstanding_capital"] == pytest.approx(0.000453, abs=5e-7)
    approx = linear_interpolate(0.02, B.annuity_due(65), 0.03, B.with_rate(0.03).annuity_due(65), 0.025)
    assert approx == pytest.approx(21.915639, **A6)
    assert B.with_rate(0.025).annuity_due(65) == pytest.approx(21.833149, **A6)
    j = secant_solve(lambda r: 16_048.52 * annuity_due_certain(20, r) - 200_000, 0.05, 0.06)
    assert j == pytest.approx(0.056585, abs=1e-6)
