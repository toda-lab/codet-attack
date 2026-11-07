"""Tests for box.py."""

import numpy as np

from smt_attack.box import BoundingBox, SortedBoxes


def test_valid() -> None:
    """Test that a well-formed box is valid.

    Returns:
        None.

    """
    b1 = BoundingBox(np.array([-1, -2]), np.array([-1, -2]))
    assert b1.is_valid()


def test_invalid() -> None:
    """Test that a box with lower > upper is invalid.

    Returns:
        None.

    """
    b1 = BoundingBox(np.array([1, 4]), np.array([0, 3]))
    assert not b1.is_valid()


def test_global() -> None:
    """Test that the global box covers the whole space.

    Returns:
        None.

    """
    b1 = BoundingBox.global_box(3)
    assert bool(np.all(b1.lower == np.array([-np.inf, -np.inf, -np.inf])))
    assert bool(np.all(b1.upper == np.array([np.inf, np.inf, np.inf])))


def test_just_one() -> None:
    """Test that a box with equal bounds is valid.

    Returns:
        None.

    """
    b1 = BoundingBox(np.array([1, 2, 3]), np.array([1, 2, 3]))
    assert b1.is_valid()


def test_and() -> None:
    """Test the intersection of two boxes.

    Returns:
        None.

    """
    b1 = BoundingBox(np.array([1, 2]), np.array([3, 2]))
    b2 = BoundingBox(np.array([1, -1]), np.array([6, 4]))
    b1_and_b2 = b1 & b2
    assert bool(np.all(b1_and_b2.lower == np.array([1, 2])))
    assert bool(np.all(b1_and_b2.upper == np.array([3, 2])))


def test_sortedboxes_intersection_and_replace() -> None:
    """Test intersection and replacement in SortedBoxes.

    Returns:
        None.

    """
    b1 = BoundingBox(np.array([0.0, -1.0]), np.array([2.0, 3.0]))
    b2 = BoundingBox(np.array([1.0, 0.0]), np.array([4.0, 2.0]))

    sb = SortedBoxes([b1, b2])
    inter = sb.get_intersection()

    assert bool(np.all(inter.lower == np.array([1.0, 0.0])))
    assert bool(np.all(inter.upper == np.array([2.0, 2.0])))

    new_b = BoundingBox(np.array([1.5, 0.5]), np.array([1.8, 1.5]))
    sb.replace_box(0, new_b)
    inter2 = sb.get_intersection()
    assert bool(np.all(inter2.lower == np.array([1.5, 0.5])))
    assert bool(np.all(inter2.upper == np.array([1.8, 1.5])))
