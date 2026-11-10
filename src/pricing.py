"""Chapter 7 — equivalence principle, expense loadings and return-of-premium guarantees."""

from __future__ import annotations

from dataclasses import dataclass

from lifecon import Basis


@dataclass(frozen=True)
class Loadings:
    """Expense loadings (classical tariff structure).

    f      acquisition cost, fraction of the capital, incurred once at issue
    g1     management cost per year, fraction of the capital, during the premium period
    g2     management cost per year, fraction of the capital, after the premium period
    alpha  collection cost, fraction of each commercial premium
    r      settlement cost, fraction of each benefit paid
    """
    f: float = 0.0
    g1: float = 0.0
    g2: float = 0.0
    alpha: float = 0.0
    r: float = 0.0


def net_level_premium(net_single_premium: float, basis: Basis, x: int, m: int) -> float:
    """Level annual net premium payable in advance for m years: Π = SP / ä_{x:m}."""
    return net_single_premium / basis.annuity_due(x, m)


def commercial_level_premium(basis: Basis, x: int, capital: float, benefit_value: float,
                             n: int, m: int, load: Loadings) -> float:
    """Commercial (office) level premium Π' payable for m years on a contract of term n.

    Equivalence: Π' ä_{x:m} = C [ (1+r) A + f + g1 ä_{x:m} + g2 (ä_{x:n} - ä_{x:m}) ]
                               + alpha Π' ä_{x:m}
    where A is the net single premium per unit of capital (benefit_value).
    """
    a_m = basis.annuity_due(x, m)
    a_n = basis.annuity_due(x, n)
    cost = capital * ((1 + load.r) * benefit_value + load.f + load.g1 * a_m
                      + load.g2 * (a_n - a_m))
    return cost / ((1 - load.alpha) * a_m)


def commercial_single_premium(basis: Basis, x: int, capital: float, benefit_value: float,
                              n: int, load: Loadings) -> float:
    """Single office premium: Π'' = C [(1+r)A + f + g2 ä_{x:n}] / (1 - alpha)."""
    cost = capital * ((1 + load.r) * benefit_value + load.f
                      + load.g2 * basis.annuity_due(x, n))
    return cost / (1 - load.alpha)


def deferred_annuity_premium(basis: Basis, x: int, m: int, benefit: float,
                             return_of_premium: bool = False, alpha: float = 0.0) -> float:
    """Level premium for a life annuity due of `benefit` per year starting at age x+m,
    premiums payable for m years.

    Without counter-insurance:  Π ä_{x:m} (1 - alpha) = B m|ä_x
    With return of premiums on death before x+m (contre-assurance):
        Π (1 - alpha) ä_{x:m} = B m|ä_x + Π (IA)^1_{x:m}
    """
    a_m = basis.annuity_due(x, m)
    benefit_value = benefit * basis.annuity_due(x, None, m)
    denom = (1 - alpha) * a_m
    if return_of_premium:
        denom -= basis.increasing_term(x, m)
    return benefit_value / denom
