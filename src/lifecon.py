"""Chapters 3 and 4 — commutation functions, life annuities and life insurances.

A ``Basis`` combines a LifeTable with a technical rate i and a death-timing
convention:
    death_timing="end"  claims paid at the end of the year of death (international)
    death_timing="mid"  claims paid in the middle of the year (k + 1/2 convention)

Commutation functions (Chapter 3 and 4 notation):
    D_x = v^x l_x,          N_x = sum_{k>=0} D_{x+k},   S_x = sum_{k>=0} N_{x+k}
    C_x = v^{x+1} d_x  (end)  or  v^{x+1/2} d_x  (mid)
    M_x = sum_{k>=0} C_{x+k},   R_x = sum_{k>=0} M_{x+k}
"""

from __future__ import annotations

from functools import lru_cache

from mortality import LifeTable


class Basis:
    def __init__(self, table: LifeTable, i: float, death_timing: str = "end") -> None:
        if death_timing not in ("end", "mid"):
            raise ValueError("death_timing must be 'end' or 'mid'")
        self.table = table
        self.i = i
        self.v = 1.0 / (1.0 + i)
        self.death_timing = death_timing
        self._lag = 1.0 if death_timing == "end" else 0.5
        self._build()

    def with_rate(self, i: float) -> "Basis":
        return Basis(self.table, i, self.death_timing)

    def with_timing(self, death_timing: str) -> "Basis":
        return Basis(self.table, self.i, death_timing)

    # ----- commutation columns -------------------------------------------
    def _build(self) -> None:
        t = self.table
        ages = range(t.x0, t.omega + 1)
        self._D = {x: self.v ** x * t.l(x) for x in ages}
        self._C = {x: self.v ** (x + self._lag) * t.d(x) for x in ages}
        self._N, self._M, self._S, self._R = {}, {}, {}, {}
        n = m = s = r = 0.0
        for x in reversed(ages):
            n += self._D[x]
            m += self._C[x]
            self._N[x], self._M[x] = n, m
        for x in reversed(ages):
            s += self._N[x]
            r += self._M[x]
            self._S[x], self._R[x] = s, r

    def _col(self, col: dict, x: int) -> float:
        if x > self.table.omega:
            return 0.0
        return col[x]

    def D(self, x: int) -> float: return self._col(self._D, x)
    def N(self, x: int) -> float: return self._col(self._N, x)
    def S(self, x: int) -> float: return self._col(self._S, x)
    def C(self, x: int) -> float: return self._col(self._C, x)
    def M(self, x: int) -> float: return self._col(self._M, x)
    def R(self, x: int) -> float: return self._col(self._R, x)

    def _end(self, x: int, n: int | None) -> int:
        return self.table.omega + 1 if n is None else x + n

    # ----- Chapter 3: pure endowment and annuities ------------------------
    def nEx(self, n: int, x: int) -> float:
        """Pure endowment nE_x = v^n np_x = D_{x+n} / D_x."""
        return self.D(x + n) / self.D(x)

    def annuity_due(self, x: int, n: int | None = None, m: int = 0) -> float:
        """m| ä_{x:n}: 1 paid at ages x+m, ..., x+m+n-1 while alive (n=None: whole life)."""
        start = x + m
        stop = self._end(start, n)
        return (self.N(start) - self.N(stop)) / self.D(x)

    def annuity_arrears(self, x: int, n: int | None = None, m: int = 0) -> float:
        """m| a_{x:n}: 1 paid at ages x+m+1, ..., x+m+n while alive."""
        start = x + m + 1
        stop = self._end(start, n)
        return (self.N(start) - self.N(stop)) / self.D(x)

    def annuity_fractional(self, x: int, k: int, n: int | None = None, m: int = 0,
                           due: bool = True) -> float:
        """k-thly life annuity, standard approximation
        m|ä^(k)_{x:n} ≈ m|ä_{x:n} - (k-1)/(2k) (mE_x - m+nE_x),
        and a^(k) = ä^(k) - (1/k)(mE_x - m+nE_x)."""
        e_start = self.nEx(m, x)
        e_stop = 0.0 if n is None else self.nEx(m + n, x)
        ad = self.annuity_due(x, n, m) - (k - 1) / (2 * k) * (e_start - e_stop)
        return ad if due else ad - (e_start - e_stop) / k

    def annuity_fractional_udd(self, x: int, k: int, n: int | None = None, m: int = 0,
                               due: bool = True) -> float:
        """k-thly life annuity computed payment by payment, with survival at fractional
        ages interpolated under UDD (benchmark for the standard approximation)."""
        t = self.table
        years = (t.omega + 1 - x - m) if n is None else n
        total = 0.0
        offset = 0 if due else 1
        for j in range(offset, years * k + offset):
            s = m + j / k
            total += self.v ** s * t.tpx_frac(s, x)
        return total / k

    def pv_moments_whole_life(self, x: int) -> tuple[float, float]:
        """(mean, standard deviation) of the present value of a unit whole life insurance:
        E = A_x, Var = 2A_x - A_x^2 where 2A_x is computed at rate (1+i)^2 - 1."""
        a1 = self.whole_life(x)
        a2 = self.with_rate((1 + self.i) ** 2 - 1).whole_life(x)
        if self.death_timing == "mid":
            raise ValueError("moments implemented for end-of-year benefits")
        return a1, (a2 - a1 ** 2) ** 0.5

    def increasing_annuity_due(self, x: int, n: int | None = None) -> float:
        """(Iä)_{x:n}: pays 1, 2, ..., n at ages x, ..., x+n-1. Whole life: S_x / D_x."""
        if n is None:
            return self.S(x) / self.D(x)
        return (self.S(x) - self.S(x + n) - n * self.N(x + n)) / self.D(x)

    def geometric_annuity_due(self, x: int, r: float, n: int | None = None) -> float:
        """Payments (1+r)^k at age x+k: equals ä_{x:n} at the rate i' = (1+i)/(1+r) - 1."""
        return self.with_rate((1 + self.i) / (1 + r) - 1).annuity_due(x, n)

    # ----- Chapter 4: insurances -----------------------------------------
    def term(self, x: int, n: int | None = None, m: int = 0) -> float:
        """m| A^1_{x:n} (n=None: whole life A_x deferred m years)."""
        start = x + m
        stop = self._end(start, n)
        return (self.M(start) - self.M(stop)) / self.D(x)

    def whole_life(self, x: int) -> float:
        return self.term(x)

    def endowment(self, x: int, n: int) -> float:
        """Endowment A_{x:n} = A^1_{x:n} + nE_x."""
        return self.term(x, n) + self.nEx(n, x)

    def increasing_term(self, x: int, n: int | None = None) -> float:
        """(IA)^1_{x:n}: death benefit k+1 in year k+1. Whole life: R_x / D_x."""
        if n is None:
            return self.R(x) / self.D(x)
        return (self.R(x) - self.R(x + n) - n * self.M(x + n)) / self.D(x)


@lru_cache(maxsize=None)
def standard_basis(table: str = "tgh05", i: float = 0.02, death_timing: str = "end") -> Basis:
    return Basis(LifeTable.builtin(table), i, death_timing)
