"""Financial mathematics toolkit for Chapter 1.

Practical Introduction to Life Actuarial Mathematics: From Foundations to Solvency.

Notation follows the book:
    i    annual effective rate of interest
    v    annual discount factor, v = (1 + i)^(-1)
    i_k  periodic rate equivalent to i over 1/k of a year, (1 + i_k)^k = 1 + i
    a_n  annuity certain in arrears
    ä_n  annuity certain due
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "discount_factor",
    "accumulation_factor",
    "periodic_rate",
    "commercial_coefficient",
    "annuity_arrears",
    "annuity_due",
    "accumulated_annuity_due",
    "perpetuity_arrears",
    "perpetuity_due",
    "deferred_annuity_arrears",
    "annuity_fractional",
    "level_payment",
    "amortization_schedule",
    "solve_rate",
]


def _check_rate(i: float) -> None:
    if i <= -1.0:
        raise ValueError(f"interest rate must exceed -100%, got {i}")


def _check_term(n: int) -> None:
    if n < 0:
        raise ValueError(f"term must be non-negative, got {n}")


def discount_factor(i: float) -> float:
    """Annual discount factor v = 1 / (1 + i)."""
    _check_rate(i)
    return 1.0 / (1.0 + i)


def accumulation_factor(i: float, t: float) -> float:
    """Compound accumulation factor (1 + i)^t."""
    _check_rate(i)
    return (1.0 + i) ** t


def periodic_rate(i: float, k: int) -> float:
    """Periodic rate i_k equivalent to the annual effective rate i: (1+i)^(1/k) - 1."""
    _check_rate(i)
    if k < 1:
        raise ValueError(f"k must be a positive integer, got {k}")
    return (1.0 + i) ** (1.0 / k) - 1.0


def commercial_coefficient(i: float, k: int) -> float:
    """Adjustment coefficient g_k = i / (k * i_k); exceeds 1 for i > 0 and k > 1."""
    if i == 0.0:
        return 1.0
    return i / (k * periodic_rate(i, k))


def annuity_arrears(n: int, i: float) -> float:
    """Present value a_n of 1 paid at the end of each year for n years: (1 - v^n) / i."""
    _check_rate(i)
    _check_term(n)
    if i == 0.0:
        return float(n)
    return (1.0 - discount_factor(i) ** n) / i


def annuity_due(n: int, i: float) -> float:
    """Present value ä_n of 1 paid at the start of each year for n years: (1 + i) a_n."""
    return (1.0 + i) * annuity_arrears(n, i)


def accumulated_annuity_due(n: int, i: float) -> float:
    """Accumulated value s̈_n at time n of 1 paid at the start of each year: (1+i)^n ä_n."""
    return accumulation_factor(i, n) * annuity_due(n, i)


def perpetuity_arrears(i: float) -> float:
    """Perpetuity in arrears a_inf = 1 / i (i > 0)."""
    if i <= 0.0:
        raise ValueError("a perpetuity only has a finite value for i > 0")
    return 1.0 / i


def perpetuity_due(i: float) -> float:
    """Perpetuity due ä_inf = 1 + 1 / i (i > 0)."""
    return 1.0 + perpetuity_arrears(i)


def deferred_annuity_arrears(m: int, n: int, i: float) -> float:
    """Deferred annuity m|a_n = v^m a_n."""
    _check_term(m)
    return discount_factor(i) ** m * annuity_arrears(n, i)


def annuity_fractional(
    n: int, i: float, k: int, due: bool = False, approx: bool = False
) -> float:
    """Annuity certain paying 1/k per period, k times a year, for n years (total 1 per year).

    Exact (approx=False):
        arrears  a_n^(k) = (1 - v^n) / (k * i_k)
        due      ä_n^(k) = (1 + i_k) * a_n^(k)
    First-order linear approximation (approx=True):
        arrears  a_n^(k) ≈ a_n + (k - 1)/(2k) (1 - v^n)
        due      ä_n^(k) ≈ ä_n - (k - 1)/(2k) (1 - v^n)
    """
    _check_rate(i)
    _check_term(n)
    if k < 1:
        raise ValueError(f"k must be a positive integer, got {k}")
    one_minus_vn = 1.0 - discount_factor(i) ** n
    if approx:
        corr = (k - 1) / (2 * k) * one_minus_vn
        return annuity_due(n, i) - corr if due else annuity_arrears(n, i) + corr
    if i == 0.0:
        return float(n)
    ik = periodic_rate(i, k)
    exact_arrears = one_minus_vn / (k * ik)
    return (1.0 + ik) * exact_arrears if due else exact_arrears


def level_payment(principal: float, rate: float, n_periods: int, due: bool = False) -> float:
    """Level instalment repaying `principal` over n_periods at the per-period `rate`."""
    factor = annuity_due(n_periods, rate) if due else annuity_arrears(n_periods, rate)
    return principal / factor


def amortization_schedule(principal: float, rate: float, n_years: int) -> pd.DataFrame:
    """Amortization table of a level-annuity loan.

    Payments are made at the end of each year. Columns:
        year, opening_balance, payment, interest (I_t = i * B_{t-1}),
        principal_repaid (K_t = payment - I_t), closing_balance.
    The final closing balance is forced to exactly zero to absorb float noise.
    """
    if n_years < 1:
        raise ValueError("n_years must be at least 1")
    payment = level_payment(principal, rate, n_years)
    rows = []
    balance = principal
    for t in range(1, n_years + 1):
        interest = rate * balance
        repaid = payment - interest
        closing = balance - repaid
        if t == n_years:
            closing = 0.0
        rows.append(
            {
                "year": t,
                "opening_balance": balance,
                "payment": payment,
                "interest": interest,
                "principal_repaid": repaid,
                "closing_balance": closing,
            }
        )
        balance = closing
    return pd.DataFrame(rows)


def solve_rate(price: float, payment: float, n: int, due: bool = False,
               lo: float = 1e-9, hi: float = 1.0, tol: float = 1e-12) -> float:
    """Yield j such that price = payment * a_n(j) (or ä_n(j) if due), by bisection."""
    f = annuity_due if due else annuity_arrears

    def g(j: float) -> float:
        return payment * f(n, j) - price

    if g(lo) * g(hi) > 0:
        raise ValueError("rate not bracketed in [lo, hi]")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(lo) * g(mid) <= 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


if __name__ == "__main__":  # pragma: no cover
    print(f"C0 = {100_000 * discount_factor(0.035) ** 8:,.2f}")
    print(f"Payment = {level_payment(200_000, 0.05, 20):,.2f}")
    print(amortization_schedule(200_000, 0.05, 20).round(2).to_string(index=False))
