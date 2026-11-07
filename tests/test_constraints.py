"""Tests for constraints.py."""

import numpy as np
import pytest
from typing_extensions import override
from z3 import Real, Solver, sat

from smt_attack.box import BoundingBox
from smt_attack.constraints import BaseConstraints
from smt_attack.point import Point
from tests.dummy_constraints import DummyConstraints


def test_base_is_abstract() -> None:
    """Test that BaseConstraints cannot be instantiated directly.

    Returns:
        None.

    """
    with pytest.raises(TypeError):
        BaseConstraints(np.array([0.0]))  # type: ignore[abstract]


def test_incomplete_subclass_is_abstract() -> None:
    """Test that a subclass missing abstract methods cannot be instantiated.

    Returns:
        None.

    """

    class Incomplete(BaseConstraints):
        @override
        def is_valid(self, x: Point) -> bool:
            return True

    with pytest.raises(TypeError):
        Incomplete(np.array([0.0]))  # type: ignore[abstract]


def test_victim_x_is_stored() -> None:
    """Test that the original point is kept by the constructor.

    Returns:
        None.

    """
    victim = np.array([1.0, 2.0])
    c = DummyConstraints(victim, BoundingBox.global_box(2))
    assert bool(np.all(c.victim_x == victim))


def test_concrete_subclass_interface() -> None:
    """Test that a complete subclass exposes the abstract interface.

    Returns:
        None.

    """
    box = BoundingBox(np.array([0.0, 1.0]), np.array([2.0, 1.0]))
    c = DummyConstraints(np.array([0.0, 1.0]), box, categorical=[0], immutable=[1])

    assert c.get_bound_box() is box
    assert c.get_categorial_index() == [0]
    assert c.get_immutable_index() == [1]
    assert c.get_moeva_constraints() == []
    assert c.is_valid(np.array([1.0, 1.0]))
    assert not c.is_valid(np.array([3.0, 1.0]))


def test_immutable_feature_has_equal_bounds() -> None:
    """Test the convention that an immutable feature has identical lower and upper bounds.

    Returns:
        None.

    """
    victim = np.array([5.0, 7.0])
    box = BoundingBox(np.array([-np.inf, 7.0]), np.array([np.inf, 7.0]))
    c = DummyConstraints(victim, box, immutable=[1])

    bound = c.get_bound_box()
    for i in c.get_immutable_index():
        assert bound.lower[i] == bound.upper[i] == victim[i]


def test_set_smt_constraints_adds_to_solver() -> None:
    """Test that set_smt_constraints can add constraints to a Z3 solver.

    Returns:
        None.

    """
    c = DummyConstraints(
        np.array([0.0]),
        BoundingBox.global_box(1),
        smt_hook=lambda s, x: s.add(x[0] >= 10),
    )
    x = [Real("x_0")]
    s = Solver()
    c.set_smt_constraints(s, x)
    s.add(x[0] <= 5)
    assert s.check() != sat
