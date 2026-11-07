"""Classes related to decision trees and decision tree ensemble models."""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from smt_attack.box import BoundingBox
from smt_attack.point import Point

InnerNodeObject: TypeAlias = dict[object, object]
LeafNodeObject: TypeAlias = dict[object, object]


@dataclass
class InnerNode:
    """Class representing a non-leaf node."""

    # Node ID (unique per decision tree)
    node_id: int
    # Go left if x[feature_id] < threshold, otherwise go right
    feature_id: int
    threshold: float
    # Left and right subtrees
    left: "InnerNode | LeafNode"
    right: "InnerNode | LeafNode"


@dataclass
class LeafNode:
    """Class representing a leaf node."""

    # Node ID (unique per decision tree)
    node_id: int
    # Value associated with the leaf node used as output
    value: float
    # Set of inputs leading to that leaf node (becomes a bounding box)
    box: BoundingBox

    @staticmethod
    def from_node_object(node: dict[object, object], box: BoundingBox) -> "LeafNode":
        """Validate a dictionary representing a LeafNode and convert it into an object.

        Args:
            node: Dictionary representing the leaf node.
            box: Region of inputs reaching the leaf.

        Returns:
            Validated leaf node.

        """
        assert "leaf" in node
        node_id_key = "nodeid" if "nodeid" in node else "node_id"
        assert node_id_key in node

        value = node["leaf"]
        assert isinstance(value, (int, float))
        node_id = node[node_id_key]
        assert isinstance(node_id, int)
        return LeafNode(
            node_id=node_id,
            value=float(value),
            box=box,
        )


class Tree:
    """Class representing a single decision tree."""

    def __init__(self, root: InnerNodeObject, d: int) -> None:
        """Construct a decision tree by traversing the dictionary representing the root node.

        Args:
            root (Node): Root node of this tree.
            d: Dimensionality of the features.

        """
        self.d = d
        self.nodes: dict[int, InnerNode | LeafNode] = {}
        self.root = self._precompute_nodes(root)
        self.leaves = [
            node for node in self.nodes.values() if isinstance(node, LeafNode)
        ]

    def _precompute_nodes(self, node: InnerNodeObject) -> InnerNode | LeafNode:
        """Read a dictionary representing a Node and return an object representing the Node.

        Args:
            node: Dictionary representing the root node.

        Returns:
            Root node object.

        """
        # Box to hold the region reachable to that vertex within dfs
        current_box = BoundingBox.global_box(self.d)

        def dfs(node: dict[object, object]) -> InnerNode | LeafNode:
            """Recursively convert a node dictionary and its subtree into node objects.

            Args:
                node: Dictionary representing the node.

            Returns:
                Converted node object.

            """
            if "leaf" in node:
                new_node = LeafNode.from_node_object(node, current_box.copy())
                self.nodes[new_node.node_id] = new_node
                return new_node

            assert "nodeid" in node
            assert "split" in node
            assert "split_condition" in node
            assert "children" in node
            assert "yes" in node
            assert "no" in node

            assert isinstance(node["nodeid"], int)
            assert isinstance(node["split"], str)
            assert isinstance(node["split_condition"], (int, float))
            assert isinstance(node["children"], list)
            assert isinstance(node["yes"], int)
            assert isinstance(node["no"], int)

            # Subtrees left/right based on child nodes whose id matches yes/no specified id
            left_child: dict[object, object] | None = None
            right_child: dict[object, object] | None = None

            for child in node["children"]:
                assert "nodeid" in child
                assert isinstance(child["nodeid"], int)

                if child["nodeid"] == node["yes"]:
                    left_child = child
                if child["nodeid"] == node["no"]:
                    right_child = child
            assert left_child is not None
            assert right_child is not None

            # Convert feature_id written as a string like f13 to a number
            split_feature_id = int(node["split"].removeprefix("f"))
            split_threshold = float(node["split_condition"])
            current_upper_value = current_box.upper[split_feature_id]
            current_lower_value = current_box.lower[split_feature_id]

            current_box.upper[split_feature_id] = min(
                current_box.upper[split_feature_id],
                split_threshold,
            )
            left_node = dfs(left_child)
            current_box.upper[split_feature_id] = current_upper_value

            current_box.lower[split_feature_id] = max(
                current_box.lower[split_feature_id],
                np.nextafter(split_threshold, np.inf),
            )
            right_node = dfs(right_child)
            current_box.lower[split_feature_id] = current_lower_value

            new_node = InnerNode(
                node_id=node["nodeid"],
                threshold=split_threshold,
                feature_id=split_feature_id,
                left=left_node,
                right=right_node,
            )
            self.nodes[new_node.node_id] = new_node
            return new_node

        return dfs(node)

    def predict(self, x: Point) -> LeafNode:
        """Return the leaf node that input data x reaches.

        Args:
            x: Input point.

        Returns:
            Leaf node reached by x.

        """
        node: InnerNode | LeafNode = self.root

        while isinstance(node, InnerNode):
            node = node.left if x[node.feature_id] <= node.threshold else node.right

        return node


class TreeEnsemble:
    """Class representing a decision tree ensemble model."""

    def __init__(
        self,
        trees: list[Tree],
        num_classes: int = 2,
        learning_rate: float = 1.0,
        init_pred: float = 0.0,
    ) -> None:
        """Initialize the ensemble model.

        Args:
            trees (list[Tree]): List of decision tree (Tree) objects composing this model.
            num_classes (int): Number of classes (currently assumes binary classification, defaults to 2).
            learning_rate (float): Learning rate score to multiply to the output of each tree (e.g. for GBDT).
            init_pred (float): Initial prediction value of the whole model (e.g. initial offset for GBDT).

        """
        self.trees = trees
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.init_pred = init_pred

    def predict_raw(self, x: Point) -> float:
        """Calculate the raw predicted value (function margin) for point x.

        Args:
            x: Input point.

        Returns:
            Raw model output.

        """
        return (
            sum(tree.predict(x).value for tree in self.trees) * self.learning_rate
            + self.init_pred
        )

    def predict_one(self, x: Point) -> int:
        """Return the predicted class for point x.

        Args:
            x: Input point.

        Returns:
            Predicted class (0 or 1).

        """
        return int(self.predict(np.array([x]))[0])

    def predict(self, xs: Iterable[Point]) -> npt.NDArray[np.int64]:
        """Return a numpy array [y0, ..., y_n] where y[i] is the predicted class for x[i].

        Args:
            xs: Input points.

        Returns:
            Predicted classes.

        """
        preds = np.array([self.predict_raw(x) for x in xs])
        return (preds > 0).astype(int)

    def predict_proba(self, xs: Iterable[Point]) -> npt.NDArray[np.float64]:
        """Return a numpy array where y[i][j] represents the probability that the class of x_i is j.

        Args:
            xs: Input points.

        Returns:
            Class probabilities of shape (n, 2).

        """
        preds = np.array([self.predict_raw(x) for x in xs])
        probas = 1 / (1 + np.exp(-preds))
        return np.vstack([1 - probas, probas]).T

    def predict_from_leaf_values(self, leaf_values: list[float]) -> int:
        """Convert the sum of the output values of each tree into a predicted value.

        Args:
            leaf_values: Output value of the reached leaf in each tree.

        Returns:
            Predicted class (0 or 1).

        """
        raw = sum(leaf_values) * self.learning_rate + self.init_pred
        return 1 if raw > 0 else 0


def parse_xgboost_json(json_path: str, d: int) -> TreeEnsemble:
    """Read the contents of the given json file and build a model.

    Args:
        json_path: Path to the XGBoost JSON dump.
        d: Dimensionality of the features.

    Returns:
        Model built from the JSON file.

    """
    with open(json_path) as f:
        trees_data = json.load(f)
        return TreeEnsemble(
            [Tree(root_node, d) for root_node in trees_data],
            num_classes=2,
            learning_rate=1.0,
            init_pred=0.0,
        )
