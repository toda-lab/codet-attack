"""Module for leaf tuples."""

from smt_attack.box import BoundingBox
from smt_attack.tree import LeafNode


class LeafTuple:
    """Object representing a leaf tuple."""

    def __init__(self, leaves: list[LeafNode]) -> None:
        """Initialize the leaf tuple.

        Args:
            leaves: Leaf node reached in each tree.

        """
        self.leaves = leaves

    def get_indexes(self) -> tuple[int, ...]:
        """Return a list of leaf tuple indices.

        Can be used when performing equality comparisons of leaf tuples or using them as elements of a set.

        Returns:
            Tuple of node IDs of the leaves.

        """
        return tuple([leaf.node_id for leaf in self.leaves])

    def get_bounding_box(self) -> BoundingBox:
        """Calculate the region corresponding to the leaf tuple.

        Returns:
            Intersection of the boxes of all leaves.

        """
        box = self.leaves[0].box
        for leaf in self.leaves[1:]:
            box &= leaf.box
        return box

    def copy(self) -> "LeafTuple":
        """Return a copy of this leaf tuple.

        Returns:
            New leaf tuple holding the same leaves.

        """
        return LeafTuple(list(self.leaves))
