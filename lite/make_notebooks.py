"""Build the JupyterLite content folder for the companion code.

(a) deletes and recreates lite/content/,
(b) copies the repository's src/, data/ and tests/ into it unchanged
    (mortality.py finds data/ through Path(__file__).parents[1], so the layout is kept),
(c) writes one notebook per chapter group, plus a start page and a test runner.

Run:  python lite/make_notebooks.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

LITE = Path(__file__).resolve().parent
REPO = LITE.parent
CONTENT = LITE / "content"

KERNEL = {"name": "python", "display_name": "Python (Pyodide)", "language": "python"}

SETUP = """import sys
if "src" not in sys.path: sys.path.insert(0, "src")
import numpy as np
import pandas as pd"""

BOOK = "*Practical Introduction to Life Actuarial Mathematics*"


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
    return new_code_cell(text.strip("\n"))


def notebook(cells: list) -> nbformat.NotebookNode:
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = KERNEL
    nb.metadata["language_info"] = {"name": "python"}
    return nb


# ---------------------------------------------------------------------------
# 00 Start here
# ---------------------------------------------------------------------------
NB00 = [
    md(f"""
# Start here

Companion notebooks for {BOOK} by F.-X. Cart-Tanneur.

Python runs **inside your browser** (Pyodide/WebAssembly): nothing to install, no account.
The first cell of a notebook takes a few seconds while Python starts.
Run a cell with **Shift + Enter**, or with the ▶ button in the toolbar on a phone or tablet.
Your edits are saved in your browser only.

| Notebook | Book chapters | Module |
|---|---|---|
| [01 Financial Mathematics](01_Financial_Mathematics.ipynb) | 1 | `finmath.py` |
| [02 Mortality](02_Mortality.ipynb) | 2 | `mortality.py` |
| [03 Annuities and Insurances](03_Annuities_and_Insurances.ipynb) | 3–4 | `lifecon.py` |
| [04 Multiple Lives](04_Multiple_Lives.ipynb) | 5–6 | `multilife.py` |
| [05 Pricing](05_Pricing.ipynb) | 7 | `pricing.py` |
| [06 Reserves and Alterations](06_Reserves_and_Alterations.ipynb) | 8–9 | `reserves.py` |
| [07 Toolbox](07_Toolbox.ipynb) | 10 | `toolbox.py` |
| [08 Run Tests](08_Run_Tests.ipynb) | all | the full test suite |

Tap a notebook name to open it. On a phone, **Run → Run All Cells** runs a whole notebook at once.

Every notebook starts with the same setup cell, which makes the book's modules in `src/` importable.
"""),
    code(SETUP),
    md("## A first look at the life tables (TGH05 men, TGF05 women, 2005 generation)"),
    code("""
from mortality import LifeTable
tgh = LifeTable.builtin("tgh05")
tgf = LifeTable.builtin("tgf05")
preview = pd.DataFrame({
    "age": [0, 20, 40, 60, 65, 80, 90, 100],
})
preview["lx men"] = [tgh.l(x) for x in preview.age]
preview["qx men"] = [tgh.q(x) for x in preview.age]
preview["lx women"] = [tgf.l(x) for x in preview.age]
preview["qx women"] = [tgf.q(x) for x in preview.age]
preview.round(6)
"""),
]

# ---------------------------------------------------------------------------
# 01 Financial mathematics
# ---------------------------------------------------------------------------
NB01 = [
    md("# Chapter 1 — Financial Mathematics in Life Insurance\n\nModule: `src/finmath.py`."),
    code(SETUP),
    code("""
import matplotlib.pyplot as plt
from finmath import (discount_factor, periodic_rate, commercial_coefficient, annuity_arrears,
                     annuity_due, perpetuity_arrears, perpetuity_due, deferred_annuity_arrears,
                     annuity_fractional, level_payment, amortization_schedule,
                     accumulated_annuity_due, solve_rate)
"""),
    md("## 1.1 Discounting: the capitalization bond (Section 1.1.6)"),
    code("""
print("single premium for $100,000 in 8 years at 3.5%:", round(100_000 * discount_factor(0.035) ** 8, 2))
print("monthly equivalent rate i_12:", round(periodic_rate(0.035, 12), 6))
"""),
    md("## 1.2 Annuities certain"),
    code("""
print("a_20 at 5%: ", round(annuity_arrears(20, 0.05), 6))
print("ä_20 at 5%: ", round(annuity_due(20, 0.05), 6))
print("perpetuities:", perpetuity_arrears(0.05), perpetuity_due(0.05))
print("5|a_10 at 5%:", round(deferred_annuity_arrears(5, 10, 0.05), 6))
"""),
    md("### Simple versus compound interest"),
    code("""
t = np.linspace(0, 40, 200)
plt.plot(t, 1 + 0.035 * t, label="simple")
plt.plot(t, 1.035 ** t, label="compound")
plt.xlabel("years"); plt.ylabel("value of $1"); plt.legend(); plt.title("Growth at 3.5%")
plt.show()
"""),
    md("## 1.3 Fractional annuities and the coefficient $g_k$"),
    code("""
rows = []
for k in (1, 2, 4, 12):
    rows.append({"k": k, "g_k": commercial_coefficient(0.05, k),
                 "exact": annuity_fractional(20, 0.05, k),
                 "approx": annuity_fractional(20, 0.05, k, approx=True)})
pd.DataFrame(rows).round(6)
"""),
    md("## 1.4 The $200,000 loan and its amortization schedule"),
    code("""
R = level_payment(200_000, 0.05, 20)
print("level payment:", round(R, 2))
schedule = amortization_schedule(200_000, 0.05, 20)
schedule.round(2)
"""),
    code("""
plt.bar(schedule["year"], schedule["interest"], label="interest")
plt.bar(schedule["year"], schedule["principal_repaid"], bottom=schedule["interest"], label="principal")
plt.xlabel("year"); plt.ylabel("$"); plt.legend(); plt.title("Split of each payment")
plt.show()
"""),
    md("## Exercises 1 and 3"),
    code("""
print("level premium for the bond:", round(100_000 / accumulated_annuity_due(8, 0.035), 2))
print("lender's yield with payments due:", round(solve_rate(200_000, R, 20, due=True), 6))
"""),
    md("## Try it\n\nChange the loan amount, rate or term and re-run the cell."),
    code("""
principal, rate, years = 200_000, 0.05, 20
print("payment:", round(level_payment(principal, rate, years), 2))
amortization_schedule(principal, rate, years).round(2).head()
"""),
]

# ---------------------------------------------------------------------------
# 02 Mortality
# ---------------------------------------------------------------------------
NB02 = [
    md("# Chapter 2 — Survival Probabilities and Mortality Models\n\nModule: `src/mortality.py`."),
    code(SETUP),
    code("""
import matplotlib.pyplot as plt
from mortality import LifeTable, makeham_tpx, gompertz_tpx, complete_expectation_numeric
tm = LifeTable.builtin("tgh05")
tf = LifeTable.builtin("tgf05")
"""),
    md("## 2.2 Reading the table"),
    code("""
print("l65, q65, d65:", tm.l(65), round(tm.q(65), 6), tm.d(65))
print("10p40:", round(tm.tpx(10, 40), 6), " 20q40:", round(tm.tqx(20, 40), 6))
print("5|10 q40:", round(tm.deferred_q(5, 10, 40), 6))
"""),
    md("## Survival curves and mortality rates"),
    code("""
ages = np.arange(0, 121)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ages, [tm.l(x) / tm.l(0) for x in ages], label="TGH05")
ax[0].plot(ages, [tf.l(x) / tf.l(0) for x in ages], label="TGF05")
ax[0].set_title("l(x) / l(0)"); ax[0].set_xlabel("age"); ax[0].legend()
ax[1].semilogy(ages[:-1], [tm.q(x) for x in ages[:-1]], label="TGH05")
ax[1].semilogy(ages[:-1], [tf.q(x) for x in ages[:-1]], label="TGF05")
ax[1].set_title("q(x), log scale"); ax[1].set_xlabel("age"); ax[1].legend()
plt.show()
"""),
    md("## 2.2.6 Fractional ages and 2.3 the force of mortality"),
    code("""
print("0.5p65 UDD:", round(tm.tpx_frac(0.5, 65, "udd"), 6), " CF:", round(tm.tpx_frac(0.5, 65, "cf"), 6))
print("mu65:", round(tm.mu(65), 6))
print("complete expectation at 65: men", round(tm.complete_expectation(65), 2),
      " women", round(tf.complete_expectation(65), 2))
"""),
    md("## 2.4 Analytical laws"),
    code("""
A, B, c = 0.00022, 2.7e-6, 1.124
print("Makeham 10p60:", round(makeham_tpx(10, 60, A, B, c), 6))
print("Gompertz 10p60:", round(gompertz_tpx(10, 60, B, c), 6))
print("Makeham e60:", round(complete_expectation_numeric(lambda t: makeham_tpx(t, 60, A, B, c), 80), 2))
"""),
    md("## 2.2.5 Building a table from rates"),
    code("""
small = LifeTable.from_qx(60, [0.01, 0.012, 0.015, 0.02, 0.03, 1.0], 1000)
small.to_frame().round(2)
"""),
    md("## Try it\n\nPick an age and a sex and compare survival to 90."),
    code("""
age = 65
for name, t in (("men", tm), ("women", tf)):
    print(name, "probability of reaching 90 from", age, ":", round(t.tpx(90 - age, age), 4))
"""),
]

# ---------------------------------------------------------------------------
# 03 Annuities and insurances
# ---------------------------------------------------------------------------
NB03 = [
    md("# Chapters 3–4 — Life Annuities and Death Benefits\n\nModule: `src/lifecon.py`. Basis: TGH05 at 2%."),
    code(SETUP),
    code("""
import matplotlib.pyplot as plt
from lifecon import standard_basis, Basis
from mortality import LifeTable
b = standard_basis("tgh05", 0.02)
"""),
    md("## 3.2 Commutation functions"),
    code("""
pd.DataFrame({"x": [40, 65, 66],
              "D": [b.D(x) for x in (40, 65, 66)],
              "N": [b.N(x) for x in (40, 65, 66)],
              "S": [b.S(x) for x in (40, 65, 66)]}).round(2)
"""),
    md("## 3.1–3.6 Pure endowments and life annuities"),
    code("""
print("25E40:", round(b.nEx(25, 40), 6))
print("ä65, a65:", round(b.annuity_due(65), 6), round(b.annuity_arrears(65), 6))
print("ä40:25 (temporary):", round(b.annuity_due(40, 25), 6))
print("25|ä40 (deferred):", round(b.annuity_due(40, m=25), 6))
print("(Iä)65:10:", round(b.increasing_annuity_due(65, 10), 6))
print("ä65 revalued at 1%:", round(b.geometric_annuity_due(65, 0.01), 6))
print("monthly pension from 65 bought at 40, $12,000:", round(b.annuity_fractional(40, 12, m=25) * 12_000, 2))
"""),
    md("## 3.4.3 Annuity values by age, sex and rate"),
    code("""
tf = Basis(LifeTable.builtin("tgf05"), 0.02)
ages = [40, 50, 60, 65, 70, 80]
pd.DataFrame({"age": ages,
              "TGH05 2%": [b.annuity_due(x) for x in ages],
              "TGF05 2%": [tf.annuity_due(x) for x in ages],
              "TGH05 1%": [b.with_rate(0.01).annuity_due(x) for x in ages],
              "TGH05 3%": [b.with_rate(0.03).annuity_due(x) for x in ages]}).round(6)
"""),
    md("## Chapter 4 — Death benefits"),
    code("""
print("C40, M40, R40:", round(b.C(40), 2), round(b.M(40), 2), round(b.R(40), 2))
print("A40:", round(b.whole_life(40), 6))
print("A1 40:20 (term):", round(b.term(40, 20), 6))
print("A40:20 (endowment):", round(b.endowment(40, 20), 6))
print("(IA)1 40:20:", round(b.increasing_term(40, 20), 6))
print("mid-year term:", round(b.with_timing("mid").term(40, 20), 6))
print("A40 = 1 - d ä40:", round(1 - 0.02 / 1.02 * b.annuity_due(40), 6))
print("mean and sd of Z:", tuple(round(v, 6) for v in b.pv_moments_whole_life(40)))
"""),
    md("### Interest-rate sensitivity of $A_{40}$"),
    code("""
rates = np.linspace(0.005, 0.05, 30)
plt.plot(rates * 100, [b.with_rate(i).whole_life(40) for i in rates], label="whole life A40")
plt.plot(rates * 100, [b.with_rate(i).endowment(40, 20) for i in rates], label="endowment A40:20")
plt.xlabel("technical rate (%)"); plt.legend(); plt.title("Single premium per $1")
plt.show()
"""),
    md("## Try it\n\nChange the age, the term or the rate."),
    code("""
age, term, rate = 40, 20, 0.02
bb = standard_basis("tgh05", rate)
print("net level premium of a $100,000 term cover:", round(100_000 * bb.term(age, term) / bb.annuity_due(age, term), 2))
"""),
]

# ---------------------------------------------------------------------------
# 04 Multiple lives
# ---------------------------------------------------------------------------
NB04 = [
    md("# Chapters 5–6 — Multiple Lives and Survivorship Benefits\n\nModule: `src/multilife.py`. "
       "Husband aged 65 on TGH05, wife aged 62 on TGF05, 2%."),
    code(SETUP),
    code("""
import matplotlib.pyplot as plt
from mortality import LifeTable
from multilife import JointLife
TGH, TGF = LifeTable.builtin("tgh05"), LifeTable.builtin("tgf05")
J = JointLife(TGH, TGF, 0.02)     # (x) = husband, (y) = wife
Jr = JointLife(TGF, TGH, 0.02)    # roles reversed
"""),
    md("## Chapter 5 — Joint-life and last-survivor statuses"),
    code("""
print("10p(65:62):", round(J.tpxy(10, 65, 62), 6), " last survivor:", round(J.tp_last(10, 65, 62), 6))
print("ä joint:", round(J.annuity_due_joint(65, 62), 6))
print("ä last survivor:", round(J.annuity_due_last(65, 62), 6))
print("A first death, second death:", round(J.A_joint(65, 62), 6), round(J.A_last(65, 62), 6))
print("not Markov:", round(J.tp_last(20, 65, 62), 6), "vs", round(J.tp_last(10, 65, 62) * J.tp_last(10, 75, 72), 6))
"""),
    code("""
t = np.arange(0, 50)
plt.plot(t, [J.tpxy(k, 65, 62) for k in t], label="both alive")
plt.plot(t, [J.tp_last(k, 65, 62) for k in t], label="at least one alive")
plt.plot(t, [TGH.tpx(k, 65) for k in t], "--", label="husband")
plt.plot(t, [TGF.tpx(k, 62) for k in t], "--", label="wife")
plt.xlabel("years"); plt.legend(); plt.title("Survival of the couple")
plt.show()
"""),
    md("## Chapter 6 — Joint and survivor annuities and contingent insurance"),
    code("""
print("joint and survivor, 60%:", round(J.joint_survivor_annuity_due(65, 62, 0.6), 6))
print("reversionary, widow:", round(J.reversionary_annuity_due(65, 62), 6))
print("reversionary, widower:", round(Jr.reversionary_annuity_due(62, 65), 6))
print("A1 (husband first):", round(J.A_contingent_first(65, 62), 6))
print("A2 (husband second):", round(J.A_contingent_second(65, 62), 6))
"""),
    md("## Section 6.4 — Advising a couple with $300,000"),
    code("""
pd.DataFrame([{"survivor's percentage": h,
               "annuity factor": J.joint_survivor_annuity_due(65, 62, h),
               "pension": 300_000 / J.joint_survivor_annuity_due(65, 62, h)}
              for h in (0.0, 0.5, 0.6, 1.0)]).round(2)
"""),
    md("## Try it\n\nChange the wife's age or the survivor's percentage."),
    code("""
wife_age, h = 55, 0.6
print("cost over a single-life pension:", round(J.joint_survivor_annuity_due(65, wife_age, h) / J.annuity_due_single(TGH, 65) - 1, 4))
"""),
]

# ---------------------------------------------------------------------------
# 05 Pricing
# ---------------------------------------------------------------------------
NB05 = [
    md("# Chapter 7 — Commercial Tariffs and Premium Loadings\n\nModule: `src/pricing.py`. "
       "Example: $100,000 20-year endowment at 40, TGH05 at 2%."),
    code(SETUP),
    code("""
import matplotlib.pyplot as plt
from lifecon import standard_basis
from pricing import (Loadings, net_level_premium, commercial_level_premium,
                     commercial_single_premium, deferred_annuity_premium)
b = standard_basis("tgh05", 0.02)
L = Loadings(f=0.03, g1=0.002, g2=0.002, alpha=0.03, r=0.01)
A = b.endowment(40, 20)
"""),
    md("## 7.1–7.2 Net and commercial premiums"),
    code("""
print("net level premium:", round(net_level_premium(100_000 * A, b, 40, 20), 2))
print("commercial level premium:", round(commercial_level_premium(b, 40, 100_000, A, 20, 20, L), 2))
print("commercial single premium:", round(commercial_single_premium(b, 40, 100_000, A, 20, L), 2))
print("premiums limited to 10 years:", round(commercial_level_premium(b, 40, 100_000, A, 20, 10, L), 2))
"""),
    md("## 7.2.6 Sensitivity to the technical rate"),
    code("""
rows = []
for i in (0.01, 0.02, 0.03):
    bi = standard_basis("tgh05", i)
    Ai = bi.endowment(40, 20)
    net = net_level_premium(100_000 * Ai, bi, 40, 20)
    com = commercial_level_premium(bi, 40, 100_000, Ai, 20, 20, L)
    rows.append({"rate": i, "net": net, "commercial": com, "loading": com - net})
pd.DataFrame(rows).round(2)
"""),
    md("## 7.3 Deferred annuity with return of premiums"),
    code("""
print("without refund:", round(deferred_annuity_premium(b, 40, 25, 10_000), 2))
print("with refund:   ", round(deferred_annuity_premium(b, 40, 25, 10_000, return_of_premium=True), 2))
"""),
    md("## Try it\n\nChange the loadings and see how the commercial premium moves."),
    code("""
my = Loadings(f=0.04, g1=0.003, g2=0.003, alpha=0.05, r=0.01)
print("commercial premium:", round(commercial_level_premium(b, 40, 100_000, A, 20, 20, my), 2))
"""),
]

# ---------------------------------------------------------------------------
# 06 Reserves and alterations
# ---------------------------------------------------------------------------
NB06 = [
    md("# Chapters 8–9 — Reserves and Policy Alterations\n\nModule: `src/reserves.py`. "
       "Example: $100,000 20-year endowment and term insurance at 40, TGH05 at 2%."),
    code(SETUP),
    code("""
import matplotlib.pyplot as plt
from lifecon import standard_basis
from reserves import Contract, policy_loan_balance
b = standard_basis("tgh05", 0.02)
E = Contract.level_premium(b, 40, 20, death=100_000, survival=100_000)   # endowment
T = Contract.level_premium(b, 40, 20, death=100_000)                     # term
print("premiums:", round(E.premium[0], 2), round(T.premium[0], 2))
"""),
    md("## 8.2 Prospective and retrospective reserves"),
    code("""
print("10V prospective:", round(E.prospective_reserve(10), 2), " retrospective:", round(E.retrospective_reserve(10), 2))
t = np.arange(0, 21)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(t, [E.prospective_reserve(k) for k in t]); ax[0].set_title("Endowment reserve")
ax[1].plot(t, [T.prospective_reserve(k) for k in t]); ax[1].set_title("Term insurance reserve")
for a in ax: a.set_xlabel("duration")
plt.show()
"""),
    md("## 8.3 The premium split (savings and risk)"),
    code("""
rows = []
for k in range(20):
    s, r = E.premium_split(k)
    rows.append({"year": k + 1, "opening V": E.prospective_reserve(k), "savings": s, "risk": r,
                 "closing V": E.prospective_reserve(k + 1)})
pd.DataFrame(rows).round(2)
"""),
    md("## 8.4 Zillmerized and management reserves"),
    code("""
print("Zillmer reserve at t=1:", round(E.zillmer_reserve(1, acquisition=3_000), 2))
E10 = Contract.level_premium(b, 40, 20, 100_000, 100_000, m=10)
print("management reserve at t=10:", round(E10.management_reserve(10, 100_000, 0.003, 0.002, m=10), 2))
"""),
    md("## Chapter 9 — Alterations at duration 10"),
    code("""
V = E.prospective_reserve(10)
print("surrender value (5%):", round(E.surrender_value(10, 0.05), 2))
print("paid-up factor:", round(E.reduced_benefit_factor(10), 6), " with g2:", round(E.reduced_benefit_factor(10, g2=0.002), 6))
print("loan balance after 3 years:", round(policy_loan_balance(0.8 * E.surrender_value(10, 0.05), 0.04, 3), 2))
print("extended term, pure endowment:", round((V - 100_000 * b.term(50, 10)) / b.nEx(10, 50), 2))
print("conversion to whole life:", round((V + E.pvfp(10)) / b.whole_life(50), 2))
"""),
    md("## Try it\n\nChoose a duration and compare surrender with a paid-up policy."),
    code("""
t = 5
print("surrender value:", round(E.surrender_value(t, 0.05), 2))
print("paid-up sum insured:", round(100_000 * E.reduced_benefit_factor(t), 2))
"""),
]

# ---------------------------------------------------------------------------
# 07 Toolbox
# ---------------------------------------------------------------------------
NB07 = [
    md("# Chapter 10 — Problem-Solving Methods and Borrower's Insurance\n\nModule: `src/toolbox.py`."),
    code(SETUP),
    code("""
import matplotlib.pyplot as plt
from lifecon import standard_basis, Basis
from mortality import LifeTable
from toolbox import borrower_insurance, linear_interpolate, secant_solve
from finmath import annuity_due
b = standard_basis("tgh05", 0.02)
"""),
    md("## 10.1 Rate interpolation and the secant method"),
    code("""
approx = linear_interpolate(0.02, b.annuity_due(65), 0.03, b.with_rate(0.03).annuity_due(65), 0.025)
exact = b.with_rate(0.025).annuity_due(65)
print("interpolated:", round(approx, 6), " exact:", round(exact, 6))
print("yield:", round(secant_solve(lambda j: 16_048.52 * annuity_due(20, j) - 200_000, 0.05, 0.06), 6))
"""),
    md("## 10.2 Borrower's insurance on the $200,000 loan"),
    code("""
rows = []
for label, table, age in (("man 30", "tgh05", 30), ("man 40", "tgh05", 40),
                          ("man 50", "tgh05", 50), ("woman 40", "tgf05", 40)):
    res = borrower_insurance(Basis(LifeTable.builtin(table), 0.02), age, 200_000, 0.05, 20)
    rows.append({"borrower": label, **res})
pd.DataFrame(rows).round(6)
"""),
    md("## Try it\n\nChange the loan or the borrower's age."),
    code("""
borrower_insurance(b, 45, 300_000, 0.04, 25)
"""),
]

# ---------------------------------------------------------------------------
# 08 Run tests
# ---------------------------------------------------------------------------
NB08 = [
    md("# Run the test suite\n\nRuns every automated test of the companion code in your browser. "
       "The first cell downloads `pytest`."),
    code("%pip install pytest"),
    code("""
import numpy, pandas
import pytest
pytest.main(["-q", "-p", "no:cacheprovider", "tests"])
"""),
]

NOTEBOOKS = {
    "00_Start_Here": NB00,
    "01_Financial_Mathematics": NB01,
    "02_Mortality": NB02,
    "03_Annuities_and_Insurances": NB03,
    "04_Multiple_Lives": NB04,
    "05_Pricing": NB05,
    "06_Reserves_and_Alterations": NB06,
    "07_Toolbox": NB07,
    "08_Run_Tests": NB08,
}


def main() -> None:
    if CONTENT.exists():
        shutil.rmtree(CONTENT)
    CONTENT.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
    for folder in ("src", "data", "tests"):
        shutil.copytree(REPO / folder, CONTENT / folder, ignore=ignore)
    for name, cells in NOTEBOOKS.items():
        nbformat.write(notebook(cells), CONTENT / f"{name}.ipynb")
    print(f"Wrote {len(NOTEBOOKS)} notebooks to {CONTENT}")


if __name__ == "__main__":
    main()
