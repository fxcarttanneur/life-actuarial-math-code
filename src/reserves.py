"""Chapters 8 and 9 — mathematical reserves and policy alterations.

A ``Contract`` on a life aged x at issue, term n years, described by yearly vectors:
    death[k]     benefit if death occurs in policy year k+1 (k = 0..n-1)
    premium[k]   premium paid at the start of policy year k+1 if alive
    survival     benefit paid at time n if alive
Death benefits follow the timing convention of the Basis (end or mid-year).
"""

from __future__ import annotations

from typing import Sequence

from lifecon import Basis


def _vec(value: float | Sequence[float], n: int) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * n
    out = [float(v) for v in value]
    if len(out) != n:
        raise ValueError(f"expected {n} values, got {len(out)}")
    return out


class Contract:
    def __init__(self, basis: Basis, x: int, n: int,
                 death: float | Sequence[float] = 0.0, survival: float = 0.0,
                 premium: float | Sequence[float] = 0.0) -> None:
        self.b, self.x, self.n = basis, x, n
        self.death = _vec(death, n)
        self.survival = float(survival)
        self.premium = _vec(premium, n)

    @classmethod
    def level_premium(cls, basis: Basis, x: int, n: int, death: float = 0.0,
                      survival: float = 0.0, m: int | None = None) -> "Contract":
        """Contract whose level net premium, payable for m <= n years, satisfies equivalence."""
        m = n if m is None else m
        c = cls(basis, x, n, death, survival, 0.0)
        p = c.pvfb(0) / basis.annuity_due(x, m)
        c.premium = [p if k < m else 0.0 for k in range(n)]
        return c

    # ----- present values at duration t (life aged x+t alive) -------------
    def pvfb(self, t: int) -> float:
        b, x = self.b, self.x
        deaths = sum(self.death[k] * b.C(x + k) for k in range(t, self.n))
        return (deaths + self.survival * b.D(x + self.n)) / b.D(x + t)

    def pvfp(self, t: int) -> float:
        b, x = self.b, self.x
        return sum(self.premium[k] * b.D(x + k) for k in range(t, self.n)) / b.D(x + t)

    def prospective_reserve(self, t: int) -> float:
        """tV = PV(future benefits) - PV(future premiums); tV at t = n is the survival benefit."""
        if t >= self.n:
            return self.survival if t == self.n else 0.0
        return self.pvfb(t) - self.pvfp(t)

    def retrospective_reserve(self, t: int) -> float:
        """tV = [ sum_{k<t} P_k D_{x+k} - sum_{k<t} S_k C_{x+k} ] / D_{x+t}."""
        b, x = self.b, self.x
        acc = sum(self.premium[k] * b.D(x + k) - self.death[k] * b.C(x + k) for k in range(t))
        return acc / b.D(x + t)

    def premium_split(self, t: int) -> tuple[float, float]:
        """(savings premium, risk premium) for policy year t+1:
        P^s = v tV_{t+1} - tV,  P^r = (C_{x+t}/D_{x+t}) S_t - v q_{x+t} V_{t+1}."""
        b, x = self.b, self.x
        v_next = self.prospective_reserve(t + 1)
        saving = b.v * v_next - self.prospective_reserve(t)
        risk = b.C(x + t) / b.D(x + t) * self.death[t] - b.v * b.table.q(x + t) * v_next
        return saving, risk

    def zillmer_reserve(self, t: int, acquisition: float, m: int | None = None) -> float:
        """Zillmerised reserve tV'' = tV - acquisition * ä_{x+t:m-t} / ä_{x:m}."""
        m = self.n if m is None else m
        if t >= m:
            return self.prospective_reserve(t)
        b = self.b
        return (self.prospective_reserve(t)
                - acquisition * b.annuity_due(self.x + t, m - t) / b.annuity_due(self.x, m))

    def management_reserve(self, t: int, capital: float, g1: float, g2: float,
                           m: int | None = None) -> float:
        """Global management reserve G_t: PV of future management costs minus PV of the
        level management loading still to be collected. The loading gamma is fixed at issue:
        gamma ä_{x:m} = g1 ä_{x:m} + g2 (ä_{x:n} - ä_{x:m})."""
        m = self.n if m is None else m
        b, x, n = self.b, self.x, self.n

        def costs(age: int, rem_m: int, rem_n: int) -> float:
            a_m = b.annuity_due(age, rem_m) if rem_m > 0 else 0.0
            return g1 * a_m + g2 * (b.annuity_due(age, rem_n) - a_m)

        gamma = costs(x, m, n) / b.annuity_due(x, m)
        rem_m = max(m - t, 0)
        loading = gamma * (b.annuity_due(x + t, rem_m) if rem_m > 0 else 0.0)
        return capital * (costs(x + t, rem_m, n - t) - loading)

    # ----- Chapter 9: alterations ------------------------------------------
    def surrender_value(self, t: int, penalty: float = 0.05) -> float:
        """Surrender value: reserve less a proportional penalty."""
        return max(self.prospective_reserve(t), 0.0) * (1 - penalty)

    def reduced_benefit_factor(self, t: int, g2: float = 0.0, value: float | None = None) -> float:
        """Paid-up (reduction) factor: fraction of the original benefits kept when premiums
        stop at duration t, funded by `value` (default: the reserve):
            k = value / (PVFB_t + g2 * C * ä_{x+t:n-t}),
        where C, the reference capital for management costs, is the largest benefit."""
        value = self.prospective_reserve(t) if value is None else value
        ref = max(max(self.death), self.survival)
        cost = self.pvfb(t) + g2 * ref * self.b.annuity_due(self.x + t, self.n - t)
        return value / cost


def policy_loan_balance(advance: float, rate: float, years: float) -> float:
    """Balance of a policy advance with interest capitalised: A (1 + rate)^years."""
    return advance * (1 + rate) ** years
