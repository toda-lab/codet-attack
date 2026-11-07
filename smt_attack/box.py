"""Module for bounding boxes."""

import numpy as np
from sortedcontainers import SortedList

from smt_attack.point import Point


class BoundingBox:
    """Class representing a bounding box in a d-dimensional space."""

    def __init__(self, lower_bound: Point, upper_bound: Point) -> None:
        """Initialize the box with lower and upper bounds for each dimension.

        Args:
            lower_bound: Lower bound of each dimension.
            upper_bound: Upper bound of each dimension.

        """
        if len(lower_bound) != len(upper_bound):
            raise ValueError(
                "lower_bound and upper_bound must have same length:"
                f" got {len(lower_bound)} and {len(upper_bound)}",
            )

        self.d = len(lower_bound)
        self.lower = lower_bound
        self.upper = upper_bound

    @staticmethod
    def global_box(d: int) -> "BoundingBox":
        """Return a bounding box covering the entire d-dimensional space.

        Args:
            d: Number of dimensions.

        Returns:
            Box with infinite bounds in every dimension.

        """
        return BoundingBox(
            lower_bound=np.full((d,), -np.inf),
            upper_bound=np.full((d,), np.inf),
        )

    def is_valid(self) -> bool:
        """Check if the BoundingBox is not empty.

        Returns:
            True if the lower bound does not exceed the upper bound in every dimension.

        """
        return bool(np.all(self.lower <= self.upper))

    def __and__(self, other: "BoundingBox") -> "BoundingBox":
        """Return the intersection of two BoundingBoxes using &.

        Args:
            other: Box to intersect with.

        Returns:
            Intersection of the two boxes (possibly empty).

        """
        left = np.maximum(self.lower, other.lower)
        right = np.minimum(self.upper, other.upper)
        return BoundingBox(left, right)

    def copy(self) -> "BoundingBox":
        """Return a copy of the BoundingBox.

        Returns:
            New box with copied bound arrays.

        """
        return BoundingBox(lower_bound=self.lower.copy(), upper_bound=self.upper.copy())


class SortedBoxes:
    """Data structure to manage the regions corresponding to the leaf nodes of decision trees with a list of thresholds for each dimension.

    See Section 3.4 of 'An Efficient Adversarial Attack for Tree Ensembles'.
    """

    def __init__(self, boxes: list[BoundingBox]) -> None:
        """Construct from BoundingBoxes corresponding to the leaf nodes reached by each tree.

        Args:
            boxes: Boxes of the leaf nodes reached by each tree.

        """
        if len(boxes) == 0:
            raise ValueError("boxes must not be empty")

        self.boxes: list[BoundingBox] = [
            BoundingBox(lower_bound=e.lower, upper_bound=e.upper) for e in boxes
        ]
        self.tree_num = len(self.boxes)
        self.d = len(self.boxes[0].lower)
        self.lower: list[SortedList[np.float64]] = [SortedList() for i in range(self.d)]
        self.upper: list[SortedList[np.float64]] = [SortedList() for i in range(self.d)]

        for box in self.boxes:
            for i in range(self.d):
                self.lower[i].add(box.lower[i])
                self.upper[i].add(box.upper[i])

    def get_intersection(self) -> BoundingBox:
        """Calculate and return the overall intersection.

        Returns:
            Intersection of all managed boxes.

        """
        box = BoundingBox.global_box(self.d)

        for i in range(self.d):
            box.lower[i] = self.lower[i][-1]
            box.upper[i] = self.upper[i][0]

        return box

    def replace_box(self, t: int, box: BoundingBox) -> None:
        """Modify the T-th tree.

        Args:
            t: Index of the tree whose box is replaced.
            box: New box for the t-th tree.

        Returns:
            None.

        """
        old = self.boxes[t]
        for i in range(self.d):
            self.lower[i].remove(old.lower[i])
            self.upper[i].remove(old.upper[i])

        self.boxes[t] = BoundingBox(lower_bound=box.lower, upper_bound=box.upper)

        for i in range(self.d):
            self.lower[i].add(self.boxes[t].lower[i])
            self.upper[i].add(self.boxes[t].upper[i])
