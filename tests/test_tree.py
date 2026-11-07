"""Tests for tree.py."""

import numpy as np

from smt_attack.tree import LeafNode, parse_xgboost_json


def test_tree() -> None:
    """Test prediction of a small tree.

    Returns:
        None.

    """
    model = parse_xgboost_json("tests/sample_xgboost.json", 2)
    assert len(model.trees) == 1

    tree = model.trees[0]
    pred_leaf = tree.predict(np.array([2, 6]))
    assert pred_leaf.node_id == 4
    assert pred_leaf.value == -3


def test_leaf_boxes() -> None:
    """Test the boxes of the leaves.

    Returns:
        None.

    """
    model = parse_xgboost_json("tests/sample_xgboost.json", 2)
    tree = model.trees[0]

    assert isinstance(tree.nodes[3], LeafNode)
    assert bool(np.all(tree.nodes[3].box.lower == np.array([-np.inf, -np.inf])))
    assert bool(np.all(tree.nodes[3].box.upper == np.array([3, 4])))

    assert isinstance(tree.nodes[4], LeafNode)
    assert bool(np.all(
        tree.nodes[4].box.lower == np.array([-np.inf, np.nextafter(4, np.inf)]),
    ))
    assert bool(np.all(tree.nodes[4].box.upper == np.array([3, np.inf])))

    assert isinstance(tree.nodes[5], LeafNode)
    assert not tree.nodes[5].box.is_valid()
    assert bool(np.all(
        tree.nodes[5].box.lower == np.array([np.nextafter(3, np.inf), -np.inf]),
    ))
    assert bool(np.all(tree.nodes[5].box.upper == np.array([2, np.inf])))

    assert isinstance(tree.nodes[6], LeafNode)
    assert bool(np.all(
        tree.nodes[6].box.lower == np.array([np.nextafter(3, np.inf), -np.inf]),
    ))
    assert bool(np.all(tree.nodes[6].box.upper == np.array([np.inf, np.inf])))


def test_leaf_list() -> None:
    """Test the list of leaves.

    Returns:
        None.

    """
    model = parse_xgboost_json("tests/sample_xgboost.json", 2)
    tree = model.trees[0]
    assert len(tree.leaves) == 4
