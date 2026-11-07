"""Abstract class for constraint classes."""

from abc import ABC, abstractmethod

from constrained_attacks.constraints.relation_constraint import BaseRelationConstraint
from pyscipopt import Model, Variable
from z3 import ArithRef, Solver

from smt_attack.box import BoundingBox
from smt_attack.point import Point


class BaseConstraints(ABC):
    """Abstract class for constraints."""

    def __init__(self, victim_x: Point) -> None:
        """Receive original point information, which is needed to express immutable constraints.

        Args:
            victim_x: Original point being attacked.

        """
        self.victim_x = victim_x

    @abstractmethod
    def is_valid(self, x: Point) -> bool:
        """Check if x satisfies the constraints.

        Args:
            x: Point to check.

        Returns:
            True if the constraint is satisfied, False otherwise.

        """
        ...

    @abstractmethod
    def get_bound_box(self) -> BoundingBox:
        """Return the region where adversarial points for victim_x can exist.

        Corresponds to the border constraint in the paper. For immutable variables, the upper and lower borders are identical.

        Returns:
            Bounding box of the search region.

        """

    @abstractmethod
    def get_categorial_index(self) -> list[int]:
        """Return the indices of discrete features.

        Corresponds to the type constraint in the paper. This information is used for variable definition in the solver.
        For more detailed constraints (e.g. features taking discrete values like 36 or 60), use get_*_constraints().

        Returns:
            List of feature indices.

        """
        ...

    @abstractmethod
    def get_immutable_index(self) -> list[int]:
        """Return the indices of immutable features.

        Returns:
            List of feature indices.

        """

    @abstractmethod
    def set_scip_constraints(self, s: Model, x: list[Variable]) -> None:
        """Add constraints to the SCIP solver instance.

        x represents variables corresponding to features in the SCIP solver. Constraints are described using these variables.
        This covers all constraints that cannot be expressed by type and border constraints.

        Args:
            s: SCIP model to add the constraints to.
            x: SCIP variables corresponding to the features.

        Returns:
            None.

        """

    @abstractmethod
    def set_smt_constraints(self, s: Solver, x: list[ArithRef]) -> None:
        """Add constraints to the SMT solver instance.

        SMT solver version of set_scip_constraints.

        Args:
            s: Z3 solver to add the constraints to.
            x: Z3 variables corresponding to the features.

        Returns:
            None.

        """

    @abstractmethod
    def get_moeva_constraints(self) -> list[BaseRelationConstraint]:
        """Return relational constraints used in Moeva2.

        Returns:
            List of relation constraints.

        """
        ...
