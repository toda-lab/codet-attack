"""Minimal concrete BaseConstraints implementation used by the tests."""

from collections.abc import Callable

import numpy as np
from constrained_attacks.constraints.relation_constraint import BaseRelationConstraint
from pyscipopt import Model, Variable
from typing_extensions import override
from z3 import ArithRef, Solver

from smt_attack.box import BoundingBox
from smt_attack.constraints import BaseConstraints
from smt_attack.point import Point


class DummyConstraints(BaseConstraints):
    """Constraints defined only by a bounding box, categorical indices and an optional extra SMT hook."""

    def __init__(
        self,
        victim_x: Point,
        box: BoundingBox,
        categorical: list[int] | None = None,
        immutable: list[int] | None = None,
        smt_hook: Callable[[Solver, list[ArithRef]], None] | None = None,
    ) -> None:
        """Initialize the dummy constraints.

        Args:
            victim_x: Original point being attacked.
            box: Region where adversarial points can exist.
            categorical: Indices of discrete features.
            immutable: Indices of immutable features.
            smt_hook: Function adding extra constraints to the Z3 solver.

        """
        super().__init__(victim_x)
        self.box = box
        self.categorical = categorical if categorical is not None else []
        self.immutable = immutable if immutable is not None else []
        self.smt_hook = smt_hook

    @override
    def is_valid(self, x: Point) -> bool:
        """Check if x is inside the box.

        Args:
            x: Point to check.

        Returns:
            True if x is inside the box.

        """
        return bool(np.all(x >= self.box.lower)) and bool(np.all(x <= self.box.upper))

    @override
    def get_bound_box(self) -> BoundingBox:
        """Return the box given at construction.

        Returns:
            Bounding box of the search region.

        """
        return self.box

    @override
    def get_categorial_index(self) -> list[int]:
        """Return the indices of discrete features.

        Returns:
            List of feature indices.

        """
        return self.categorical

    @override
    def get_immutable_index(self) -> list[int]:
        """Return the indices of immutable features.

        Returns:
            List of feature indices.

        """
        return self.immutable

    @override
    def set_scip_constraints(self, s: Model, x: list[Variable]) -> None:
        """Add no constraint to SCIP.

        Args:
            s: SCIP model.
            x: SCIP variables.

        """

    @override
    def set_smt_constraints(self, s: Solver, x: list[ArithRef]) -> None:
        """Apply the optional hook to the Z3 solver.

        Args:
            s: Z3 solver.
            x: Z3 variables.

        """
        if self.smt_hook is not None:
            self.smt_hook(s, x)

    @override
    def get_moeva_constraints(self) -> list[BaseRelationConstraint]:
        """Return no relational constraint.

        Returns:
            Empty list.

        """
        return []
