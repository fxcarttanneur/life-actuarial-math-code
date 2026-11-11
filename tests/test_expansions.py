"""Tests for the figures added when Chapters 2-10 were expanded."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finmath import amortization_schedule  # noqa: E402
from lifecon import Basis, standard_basis  # noqa: E402
from mortality import LifeTable, makeham_tpx  # noqa: E402
from multilife import JointLife  # noqa: E402
from pricing import Loadings, commercial_level_premium, net_level_premium  # noqa: E402
from reserves import Contract  # noqa: E402
from toolbox import borrower_insurance  # noqa: E402

TH = LifeTable.builtin("tgh05")
TF = LifeTable.builtin("tgf05")
B = standard_basis("tgh05", 0.02)
BF = Basis(TF, 0.02)
J = JointLife(TH, TF, 0.02)
Jr = JointLife(TF, TH, 0.02)
E = Contract.level_premium(B, 40, 20, death=100_000, survival=100_000)
T = Contract.level_premium(B, 40, 20, death=100_000)
A6 = dict(abs=5e-7)
L = Loadings(f=0.03, g1=0.002, g2=0.002, alpha=0.03, r=0.01)


# Chapter 2
def test_fractional_survival():
    assert TH.tpx_frac(0.5, 65) == pytest.approx(0.999218, **A6)
    assert TH.tpx_frac(0.5, 65, "cf") == pytest.approx(0.999217, **A6)
    assert TH.tpx_frac(2.5, 65) == pytest.approx(0.995836, **A6)
    assert TH.tpx_frac(3, 65) == pytest.approx(TH.tpx(3, 65))
    assert TH.mu_udd(65, 0.5) == pytest.approx(0.001566, **A6)


def test_toy_table_and_summaries():
    toy = LifeTable.from_qx(60, [0.01, 0.012, 0.015, 0.02, 0.03, 1.0], 1000)
    assert toy.tpx(2, 61) == pytest.approx(0.973180, **A6)
    assert toy.curtate_expectation(60) == pytest.approx(4.7916, abs=5e-5)
    half = TH.l(65) / 2
    assert TH.l(97) > half >= TH.l(98)
    assert max(range(20, 120), key=TH.d) == 97
    assert max(range(20, 120), key=TF.d) == 101
    assert 1 - makeham_tpx(1, 60, 0.00022, 2.7e-6, 1.124) == pytest.approx(0.003398, **A6)


# Chapter 3
def test_annuity_reference_table():
    assert B.annuity_due(80) == pytest.approx(14.974510, **A6)
    assert BF.annuity_due(40) == pytest.approx(34.859026, **A6)
    assert B.with_rate(0.03).annuity_due(70) == pytest.approx(18.399461, **A6)


def test_fractional_accuracy_and_plan():
    assert B.annuity_fractional_udd(65, 12) == pytest.approx(22.949016, **A6)
    assert B.annuity_fractional_udd(65, 4) == pytest.approx(23.032491, **A6)
    assert B.annuity_fractional(40, 12, 25) == pytest.approx(19.618842, **A6)
    benefit = 12_000 * B.annuity_fractional(40, 12, None, 25)
    assert round(benefit / B.annuity_due(40, 25), 2) == 8_321.69
    assert round(benefit / B.annuity_fractional(40, 12, 25) / 12, 2) == 699.98


def test_guaranteed_period():
    certain10 = (1 - 1.02 ** -10) / (1 - 1 / 1.02)
    assert certain10 + B.annuity_due(65, None, 10) == pytest.approx(23.494061, **A6)
    certain5 = (1 - 1.02 ** -5) / (1 - 1 / 1.02)
    assert certain5 + B.annuity_due(65, None, 5) == pytest.approx(23.425973, **A6)


# Chapter 4
def test_premiums_by_age_and_moments():
    assert round(100_000 * B.term(50, 20) / B.annuity_due(50, 20), 2) == 113.77
    assert round(100_000 * B.whole_life(60) / B.annuity_due(60), 2) == 1_904.99
    m, s = B.pv_moments_whole_life(40)
    assert s == pytest.approx(0.084503, **A6)
    assert 21 * B.term(40, 20) - B.increasing_term(40, 20) == pytest.approx(0.080425, **A6)
    assert B.term(40, None, 10) == pytest.approx(B.nEx(10, 40) * B.whole_life(50))


# Chapter 5
def test_couples():
    assert J.annuity_due_last(65, 55) == pytest.approx(30.442700, **A6)
    assert J.annuity_due_joint(65, 62, 10) == pytest.approx(9.023661, **A6)
    v = 1 / 1.02
    endow = sum(v ** (k + 1) * (J.tpxy(k, 65, 62) - J.tpxy(k + 1, 65, 62)) for k in range(10)) \
        + v ** 10 * J.tpxy(10, 65, 62)
    assert endow == pytest.approx(1 - 0.02 / 1.02 * J.annuity_due_joint(65, 62, 10))
    px = sum(J.tpxy(k, 65, 62) * TH.q(65 + k) * (1 - TF.q(62 + k) / 2) for k in range(60))
    py = sum(J.tpxy(k, 65, 62) * TF.q(62 + k) * (1 - TH.q(65 + k) / 2) for k in range(60))
    assert px == pytest.approx(0.705579, **A6)
    assert px + py == pytest.approx(1.0, abs=1e-9)


# Chapter 6
def test_reversion_extensions():
    assert Jr.reversionary_annuity_due(62, 65) == pytest.approx(1.379808, **A6)
    total = J.reversionary_annuity_due(65, 62) + Jr.reversionary_annuity_due(62, 65) \
        + J.annuity_due_joint(65, 62)
    assert total == pytest.approx(J.annuity_due_last(65, 62))
    assert J.A_contingent_second(65, 62) + Jr.A_contingent_second(62, 65) \
        == pytest.approx(J.A_last(65, 62), rel=1e-9)
    assert J.joint_survivor_annuity_due(65, 65, 0.6) == pytest.approx(25.737637, **A6)
    assert round(300_000 / J.joint_survivor_annuity_due(65, 62, 0.6), 2) == 11_419.34


# Chapter 7
def test_tariff_extensions():
    Awl = B.whole_life(40)
    n = TH.omega + 1 - 40
    assert round(commercial_level_premium(B, 40, 100_000, Awl, n, 20, L), 2) == 2_718.11
    assert round(commercial_level_premium(B, 40, 100_000, Awl, n, n, L), 2) == 1_335.69
    b3 = B.with_rate(0.03)
    assert round(commercial_level_premium(b3, 40, 100_000, b3.endowment(40, 20), 20, 20, L), 2) == 4_195.94
    lapse = sum(1.02 ** -k * TH.tpx(k, 40) * 0.95 ** k for k in range(20))
    assert lapse == pytest.approx(11.023179, **A6)


# Chapter 8
def test_reserve_extensions():
    W = Contract.level_premium(B, 40, TH.omega + 1 - 40, death=100_000)
    assert round(W.prospective_reserve(20), 2) == 23_489.89
    b1 = B.with_rate(0.01)
    v1 = 100_000 * b1.endowment(50, 10) - E.premium[0] * b1.annuity_due(50, 10)
    assert round(v1, 2) == 51_859.50
    risks = [E.premium_split(t)[1] for t in range(20)]
    assert risks.index(max(risks)) == 7


# Chapter 9
def test_alteration_extensions():
    assert E.reduced_benefit_factor(1) == pytest.approx(0.059843, **A6)
    assert T.reduced_benefit_factor(15) == pytest.approx(0.448749, **A6)
    V = E.prospective_reserve(10)
    assert round((V - 100_000 * B.term(50, 10)) / B.nEx(10, 50), 2) == 54_428.22
    assert round((V + E.pvfp(10)) / B.whole_life(50), 2) == 201_037.25
    assert round(V / B.whole_life(50), 2) == 110_258.58


# Chapter 10
def test_borrower_extensions():
    assert round(borrower_insurance(B, 50, 200_000, 0.05, 20)["single_premium"], 2) == 1_988.44
    assert round(borrower_insurance(BF, 40, 200_000, 0.05, 20)["level_premium"], 2) == 95.67
    bal = list(amortization_schedule(200_000, 0.05, 20)["opening_balance"])
    contrib = [bal[k] * B.C(40 + k) / B.D(40) for k in range(20)]
    assert contrib.index(max(contrib)) == 7
    assert round(max(contrib), 2) == 61.11
