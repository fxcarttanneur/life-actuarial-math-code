"""Chapter 2 — survival probabilities, life tables and analytical mortality laws.

All tables are indexed by integer ages x = x0, x0+1, ..., omega with l_x > 0, and
l_x = 0 for every age beyond omega.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class LifeTable:
    """Discrete life table built from the number of survivors l_x."""

    def __init__(self, ages: Sequence[int], lx: Sequence[float], name: str = "") -> None:
        ages = [int(a) for a in ages]
        if any(b - a != 1 for a, b in zip(ages, ages[1:])):
            raise ValueError("ages must be consecutive integers")
        lx = np.asarray(lx, dtype=float)
        if np.any(np.diff(lx) > 0):
            raise ValueError("l_x must be non-increasing")
        keep = lx > 0
        self.x0 = ages[0]
        self._lx = lx[keep]
        self.omega = self.x0 + len(self._lx) - 1
        self.name = name

    # ----- constructors ---------------------------------------------------
    @classmethod
    def from_csv(cls, path: str | Path, name: str = "") -> "LifeTable":
        df = pd.read_csv(path)
        return cls(df["age"], df["lx"], name or Path(path).stem)

    @classmethod
    def builtin(cls, name: str) -> "LifeTable":
        """Load the fictitious tables 'tm2026' (male) or 'tf2026' (female) from data/."""
        return cls.from_csv(DATA_DIR / f"{name.lower()}.csv", name.upper())

    @classmethod
    def from_qx(cls, x0: int, qx: Sequence[float], radix: float = 100_000.0,
                name: str = "") -> "LifeTable":
        lx = [radix]
        for q in qx:
            lx.append(lx[-1] * (1.0 - q))
        return cls(range(x0, x0 + len(lx)), lx, name)

    @classmethod
    def from_survival(cls, s: Callable[[float], float], x0: int, omega: int,
                      radix: float = 100_000.0, name: str = "") -> "LifeTable":
        """Discretise a survival function s(x) = P(X > x) on integer ages."""
        base = s(x0)
        ages = range(x0, omega + 1)
        return cls(ages, [radix * s(a) / base for a in ages], name)

    # ----- basic functions -----------------------------------------------
    def l(self, x: int) -> float:
        if x < self.x0:
            raise ValueError(f"age {x} below table start {self.x0}")
        if x > self.omega:
            return 0.0
        return float(self._lx[x - self.x0])

    def d(self, x: int) -> float:
        return self.l(x) - self.l(x + 1)

    def q(self, x: int) -> float:
        return self.d(x) / self.l(x) if self.l(x) > 0 else 1.0

    def p(self, x: int) -> float:
        return 1.0 - self.q(x)

    def tpx(self, t: int, x: int) -> float:
        return self.l(x + t) / self.l(x)

    def tqx(self, t: int, x: int) -> float:
        return 1.0 - self.tpx(t, x)

    def tpx_frac(self, t: float, x: int, method: str = "udd") -> float:
        """Survival over a non-integer duration t from integer age x.

        method="udd": uniform distribution of deaths within each year of age
                      (l is linear between integer ages);
        method="cf":  constant force of mortality within each year of age
                      (l is exponential between integer ages).
        """
        j = int(math.floor(t))
        s = t - j
        lj, lj1 = self.l(x + j), self.l(x + j + 1)
        if s == 0:
            return lj / self.l(x)
        if method == "udd":
            lt = (1 - s) * lj + s * lj1
        elif method == "cf":
            lt = lj * (lj1 / lj) ** s if lj > 0 else 0.0
        else:
            raise ValueError("method must be 'udd' or 'cf'")
        return lt / self.l(x)

    def mu_udd(self, x: int, s: float) -> float:
        """Force of mortality at age x+s (0 <= s < 1) under UDD: q_x / (1 - s q_x)."""
        q = self.q(x)
        return q / (1 - s * q)

    def deferred_q(self, u: int, t: int, x: int) -> float:
        """u|t q_x: probability of dying between ages x+u and x+u+t."""
        return (self.l(x + u) - self.l(x + u + t)) / self.l(x)

    def expected_survivors(self, t: int, x: int, n0: float) -> float:
        """Expected survivors at age x+t out of n0 lives aged x: n0 * tp_x."""
        return n0 * self.tpx(t, x)

    def mu(self, x: int) -> float:
        """Force of mortality at integer age x, estimated as -ln(p_{x-1} p_x) / 2."""
        if x - 1 < self.x0 or x >= self.omega:
            raise ValueError("mu needs ages x-1 and x+1 inside the table")
        return -0.5 * (math.log(self.p(x - 1)) + math.log(self.p(x)))

    def curtate_expectation(self, x: int) -> float:
        """e_x = sum_{k>=1} kp_x."""
        return sum(self.l(k) for k in range(x + 1, self.omega + 1)) / self.l(x)

    def complete_expectation(self, x: int) -> float:
        """Complete expectation under uniform deaths within each year: e_x + 1/2."""
        return self.curtate_expectation(x) + 0.5

    def to_frame(self) -> pd.DataFrame:
        ages = list(range(self.x0, self.omega + 1))
        return pd.DataFrame({
            "age": ages,
            "lx": [self.l(a) for a in ages],
            "dx": [self.d(a) for a in ages],
            "qx": [self.q(a) for a in ages],
            "px": [self.p(a) for a in ages],
        })


# ----- analytical laws ------------------------------------------------------

def gompertz_mu(x: float, B: float, c: float) -> float:
    return B * c ** x


def makeham_mu(x: float, A: float, B: float, c: float) -> float:
    return A + B * c ** x


def makeham_tpx(t: float, x: float, A: float, B: float, c: float) -> float:
    """tp_x = exp(-A t - B c^x (c^t - 1) / ln c)."""
    return math.exp(-A * t - B * c ** x * (c ** t - 1.0) / math.log(c))


def gompertz_tpx(t: float, x: float, B: float, c: float) -> float:
    return makeham_tpx(t, x, 0.0, B, c)


def de_moivre_tpx(t: float, x: float, omega: float) -> float:
    """Uniform deaths up to omega: tp_x = (omega - x - t) / (omega - x)."""
    if x + t >= omega:
        return 0.0
    return (omega - x - t) / (omega - x)


def de_moivre_mu(x: float, omega: float) -> float:
    return 1.0 / (omega - x)


def complete_expectation_numeric(tpx: Callable[[float], float], horizon: float,
                                 steps: int = 20_000) -> float:
    """Integral of tp_x dt from 0 to horizon (composite Simpson rule)."""
    if steps % 2:
        steps += 1
    h = horizon / steps
    total = tpx(0.0) + tpx(horizon)
    for j in range(1, steps):
        total += (4 if j % 2 else 2) * tpx(j * h)
    return total * h / 3.0


def integrate_mu(mu: Callable[[float], float], x: float, t: float, steps: int = 2_000) -> float:
    """tp_x = exp(-integral_0^t mu(x+s) ds), by Simpson's rule."""
    if steps % 2:
        steps += 1
    h = t / steps
    total = mu(x) + mu(x + t)
    for j in range(1, steps):
        total += (4 if j % 2 else 2) * mu(x + j * h)
    return math.exp(-total * h / 3.0)
