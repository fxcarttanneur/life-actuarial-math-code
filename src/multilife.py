"""Chapters 5 and 6 — two-life statuses, survivorship benefits and reversionary annuities.

Lives are assumed independent. Death benefits are paid at the end of the year of death.
(x) is valued on table_x, (y) on table_y, both at the same technical rate i.
"""

from __future__ import annotations

from mortality import LifeTable


class JointLife:
    def __init__(self, table_x: LifeTable, table_y: LifeTable, i: float) -> None:
        self.tx, self.ty, self.i = table_x, table_y, i
        self.v = 1.0 / (1.0 + i)

    # ----- probabilities --------------------------------------------------
    def tpxy(self, t: int, x: int, y: int) -> float:
        """Joint-life survival: both alive at time t."""
        return self.tx.tpx(t, x) * self.ty.tpx(t, y)

    def tp_last(self, t: int, x: int, y: int) -> float:
        """Last-survivor survival: at least one alive at time t."""
        return self.tx.tpx(t, x) + self.ty.tpx(t, y) - self.tpxy(t, x, y)

    def _horizon(self, x: int, y: int) -> int:
        return max(self.tx.omega - x, self.ty.omega - y) + 1

    # ----- commutation functions for the joint status ----------------------
    def D(self, x: int, y: int) -> float:
        """D_xy = v^{(x+y)/2} l_x l_y."""
        return self.v ** ((x + y) / 2) * self.tx.l(x) * self.ty.l(y)

    def N(self, x: int, y: int) -> float:
        total, k = 0.0, 0
        while self.tx.l(x + k) > 0 and self.ty.l(y + k) > 0:
            total += self.D(x + k, y + k)
            k += 1
        return total

    # ----- annuities ------------------------------------------------------
    def annuity_due_joint(self, x: int, y: int, n: int | None = None) -> float:
        """ä_xy (or ä_{xy:n}): 1 paid while both are alive."""
        h = self._horizon(x, y) if n is None else n
        return sum(self.v ** k * self.tpxy(k, x, y) for k in range(h))

    def annuity_due_single(self, table: LifeTable, x: int, n: int | None = None) -> float:
        h = table.omega - x + 1 if n is None else n
        return sum(self.v ** k * table.tpx(k, x) for k in range(h))

    def annuity_due_last(self, x: int, y: int, n: int | None = None) -> float:
        """ä_{\\bar{xy}} = ä_x + ä_y - ä_xy."""
        return (self.annuity_due_single(self.tx, x, n) + self.annuity_due_single(self.ty, y, n)
                - self.annuity_due_joint(x, y, n))

    def reversionary_annuity_due(self, x: int, y: int) -> float:
        """ä_{x|y} = ä_y - ä_xy: paid to (y) after the death of (x)."""
        return self.annuity_due_single(self.ty, y) - self.annuity_due_joint(x, y)

    def joint_survivor_annuity_due(self, x: int, y: int, h: float) -> float:
        """Joint and survivor annuity: 1 while (x) lives, then the survivor's percentage h
        to (y) if (y) survives (x): ä_x + h (ä_y - ä_xy)."""
        return self.annuity_due_single(self.tx, x) + h * self.reversionary_annuity_due(x, y)

    # Former name, kept so that existing scripts keep working.
    reversible_annuity_due = joint_survivor_annuity_due

    # ----- insurances -----------------------------------------------------
    def A_joint(self, x: int, y: int) -> float:
        """A_xy: 1 paid at the end of the year of the first death."""
        return sum(self.v ** (k + 1) * (self.tpxy(k, x, y) - self.tpxy(k + 1, x, y))
                   for k in range(self._horizon(x, y)))

    def A_single(self, table: LifeTable, x: int) -> float:
        return sum(self.v ** (k + 1) * table.deferred_q(k, 1, x)
                   for k in range(table.omega - x + 1))

    def A_last(self, x: int, y: int) -> float:
        """A_{\\bar{xy}} = A_x + A_y - A_xy: paid on the second death."""
        return self.A_single(self.tx, x) + self.A_single(self.ty, y) - self.A_joint(x, y)

    def A_contingent_first(self, x: int, y: int) -> float:
        """A^1_xy: paid at the end of the year in which (x) dies, if (y) is still alive at
        the moment of death. Deaths uniform within the year of age:
        P(x dies in year k+1 before y) = kp_xy q_{x+k} (1 - q_{y+k}/2)."""
        total = 0.0
        for k in range(self._horizon(x, y)):
            pxy = self.tpxy(k, x, y)
            if pxy == 0:
                break
            total += self.v ** (k + 1) * pxy * self.tx.q(x + k) * (1 - self.ty.q(y + k) / 2)
        return total

    def A_contingent_second(self, x: int, y: int) -> float:
        """A^2_xy: paid on the death of (x) if (y) has died before: A_x - A^1_xy."""
        return self.A_single(self.tx, x) - self.A_contingent_first(x, y)
