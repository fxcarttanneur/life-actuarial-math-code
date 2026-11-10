"""Chapter 10 — practical toolbox: rate interpolation, root finding, borrower's insurance."""

from __future__ import annotations

from typing import Callable

from finmath import amortization_schedule
from lifecon import Basis


def linear_interpolate(x0: float, y0: float, x1: float, y1: float, x: float) -> float:
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def secant_solve(f: Callable[[float], float], a: float, b: float, tol: float = 1e-12,
                 max_iter: int = 100) -> float:
    """Root of f by the secant method started from a and b."""
    fa, fb = f(a), f(b)
    for _ in range(max_iter):
        if fb == fa:
            break
        c = b - fb * (b - a) / (fb - fa)
        if abs(c - b) < tol:
            return c
        a, fa, b, fb = b, fb, c, f(c)
    return b


def borrower_insurance(basis: Basis, x: int, principal: float, rate: float, n: int) -> dict:
    """Death cover on a level-annuity loan: on death in year k+1 the insurer repays the
    outstanding balance at the start of that year (B_k).

    Returns the single premium, the level annual premium over the loan term, and the
    equivalent annual rates on the initial capital and on the outstanding capital.
    """
    sched = amortization_schedule(principal, rate, n)
    balances = list(sched["opening_balance"])
    single = sum(balances[k] * basis.C(x + k) for k in range(n)) / basis.D(x)
    a_n = basis.annuity_due(x, n)
    level = single / a_n
    on_outstanding = single / (sum(balances[k] * basis.D(x + k) for k in range(n)) / basis.D(x))
    return {
        "single_premium": single,
        "level_premium": level,
        "rate_on_initial_capital": level / principal,
        "rate_on_outstanding_capital": on_outstanding,
    }
