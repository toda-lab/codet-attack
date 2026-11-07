"""Search using SCIP solver."""

import logging as _logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from pyscipopt import Model

from smt_attack.box import BoundingBox, SortedBoxes
from smt_attack.constraints import BaseConstraints
from smt_attack.leaf_tuple import LeafTuple
from smt_attack.point import Point
from smt_attack.tree import LeafNode, TreeEnsemble

FindScip: TypeAlias = Callable[[Point, BoundingBox], Point | None]

logger = _logging.getLogger(__name__)


@dataclass
class SearchStatus:
    """Object summarizing the state of the search."""

    C: LeafTuple
    r: float
    x: Point


@dataclass
class NextStatus:
    """Indicates the region selected as the destination."""

    leaf: LeafNode
    tree_index: int
    r: float
    x: Point


def find_scip(
    victim_x: Point,
    box: BoundingBox,
    constraint: BaseConstraints,
    time_limit: float,
) -> Point | None:
    """Check if there is a point satisfying constraints in the box using SCIP.

    Args:
        victim_x: Original point being attacked.
        box: Region to search in.
        constraint: Constraints that the point must satisfy.
        time_limit: Time limit of the SCIP solver in seconds.

    Returns:
        Point closest to victim_x in the box, or None if none was found.

    """
    scip_model = Model("find_scip")
    scip_model.hideOutput()
    integer_indexes = set(constraint.get_categorial_index())
    d = victim_x.shape[0]
    x = []
    for i in range(d):
        vtype = "I" if i in integer_indexes else "C"
        lower = float(box.lower[i]) if box.lower[i] != -np.inf else None
        upper = float(box.upper[i]) if box.upper[i] != np.inf else None
        var = scip_model.addVar(vtype=vtype, name=f"x_{i}", lb=lower, ub=upper)
        x.append(var)

    constraint.set_scip_constraints(scip_model, x)

    t = scip_model.addVar(vtype="C", name="t")
    objective_val = sum((x[i] - victim_x[i]) * (x[i] - victim_x[i]) for i in range(d))
    scip_model.addCons(objective_val <= t)
    scip_model.setObjective(t, "minimize")
    scip_model.setParam("limits/time", time_limit)

    try:
        scip_model.optimize()
    except Exception as err:  # noqa: BLE001
        logger.debug(f"scip_err: {err}")
        return None

    status = scip_model.getStatus()
    if status == "optimal":
        try:
            point = []
            for i in range(d):
                val = scip_model.getVal(x[i])
                # Clamp numerical errors back into the box
                val = min(val, box.upper[i])
                val = max(val, box.lower[i])
                point.append(val)
            return np.array(point)
        except Exception:
            return None

    return None


def filter_by_t_bound(
    c: LeafTuple,
    model: TreeEnsemble,
    victim_x: Point,
    use_t_bound: bool,
) -> list[int]:
    """Trees where the boundary of the tree intersects with the point when projected onto the LeafTuple.

    Args:
        c: Current leaf tuple.
        model: Tree ensemble model under attack.
        victim_x: Original point being attacked.
        use_t_bound: Whether to filter trees; if False all trees are returned.

    Returns:
        Indices of the trees to explore.

    """
    if not use_t_bound:
        return list(range(len(model.trees)))

    c_box = c.get_bounding_box()
    proj = np.clip(victim_x, c_box.lower, c_box.upper)

    t_bound: list[int] = []

    for t_idx, leaf_node in enumerate(c.leaves):
        on_edge = False
        box = leaf_node.box
        for j in range(victim_x.shape[0]):
            if (box.lower[j] != -np.inf and proj[j] == box.lower[j]) or (box.upper[j] != np.inf and proj[j] == box.upper[j]):
                on_edge = True
                break
        if on_edge:
            t_bound.append(t_idx)

    return t_bound


def attack_loop(  # noqa: PLR0913, PLR0917
    victim_x: Point,
    dist: Callable[[Point, Point], float],
    status: SearchStatus,
    model: TreeEnsemble,
    victim_y: int,
    box_adv: BoundingBox,
    sorted_box: SortedBoxes,
    constraints: BaseConstraints,
    visited_leaf_tuples: set[tuple[int, ...]],
    time_limit: float,
    *,
    use_t_bound: bool,
) -> NextStatus | None:
    """Apply SCIP to the neighboring region.

    Args:
        victim_x: Original point being attacked.
        dist: Distance function between two points.
        status: Current search status.
        model: Tree ensemble model under attack.
        victim_y: Predicted class of victim_x.
        box_adv: Region where adversarial points can exist.
        sorted_box: Boxes of the current leaf tuple.
        constraints: Constraints that the adversarial point must satisfy.
        visited_leaf_tuples: Leaf tuples already explored (updated in place).
        time_limit: Time limit of each SCIP call in seconds.
        use_t_bound: Whether to restrict trees using the t-bound.

    Returns:
        Best neighboring state closer than the current one, or None if there is none.

    """
    next_status: NextStatus | None = None
    current_leaf_tuple = status.C.copy()
    for index_tree in filter_by_t_bound(
        current_leaf_tuple,
        model,
        victim_x,
        use_t_bound,
    ):
        tree = model.trees[index_tree]
        original_box = sorted_box.boxes[index_tree]
        for new_leaf in tree.leaves:
            current_leaf_tuple.leaves[index_tree] = new_leaf
            if current_leaf_tuple.get_indexes() in visited_leaf_tuples:
                continue
            sorted_box.replace_box(index_tree, new_leaf.box)
            new_box = sorted_box.get_intersection()
            if not new_box.is_valid():
                continue
            if (
                model.predict_from_leaf_values(
                    [leaf.value for leaf in current_leaf_tuple.leaves],
                )
                == victim_y
            ):
                continue
            new_box_advarsarial = new_box & box_adv
            if not new_box_advarsarial.is_valid():
                continue
            # Prune: even the unconstrained closest point is no better than the current best
            if min(
                status.r,
                next_status.r if next_status is not None else float("inf"),
            ) <= dist(
                victim_x,
                np.clip(victim_x, new_box_advarsarial.lower, new_box_advarsarial.upper),
            ):
                continue
            new_x = find_scip(victim_x, new_box_advarsarial, constraints, time_limit)
            visited_leaf_tuples.add(current_leaf_tuple.get_indexes())
            if new_x is None:
                continue
            new_r = dist(victim_x, new_x)
            if next_status is None or new_r < next_status.r:
                next_status = NextStatus(
                    leaf=new_leaf,
                    tree_index=index_tree,
                    r=dist(victim_x, new_x),
                    x=new_x,
                )
        current_leaf_tuple.leaves[index_tree] = status.C.leaves[index_tree]
        sorted_box.replace_box(index_tree, original_box)

    return next_status


def scip_attack(  # noqa: PLR0913, PLR0917
    model: TreeEnsemble,
    victim_x: Point,
    initial_point: Point,
    constraints: BaseConstraints,
    time_limit: float,
    dist: Callable[[Point, Point], float] = lambda x, y: float(np.linalg.norm(x - y)),
    max_iteration: int = 2000,
    *,
    use_t_bound: bool,
) -> Point:
    """Search by SCIP.

    Args:
        model: Tree ensemble model under attack.
        victim_x: Original point being attacked.
        initial_point: Starting adversarial point, e.g. found by SMT.
        constraints: Constraints that the adversarial point must satisfy.
        time_limit: Time limit of each SCIP call in seconds.
        dist: Distance function between two points.
        max_iteration: Maximum number of search iterations.
        use_t_bound: Whether to restrict trees using the t-bound.

    Returns:
        Best adversarial point found.

    """
    victim_y = model.predict_one(victim_x)
    box_adv = constraints.get_bound_box()
    leaf_tuple = LeafTuple([tree.predict(initial_point) for tree in model.trees])
    first_x0 = find_scip(
        victim_x,
        leaf_tuple.get_bounding_box(),
        constraints,
        time_limit,
    )
    # Fall back to the initial point if SCIP fails (e.g. timeout)
    x0 = first_x0 if first_x0 is not None else initial_point

    status = SearchStatus(
        C=leaf_tuple,
        r=dist(victim_x, x0),
        x=x0,
    )
    sorted_box = SortedBoxes(boxes=[leaf.box for leaf in leaf_tuple.leaves])

    visited_leaf_tuples: set[tuple[int, ...]] = set()
    visited_leaf_tuples.add(leaf_tuple.get_indexes())
    current_iteration = 0
    # Corresponds to Algorithm 1 of Zhang et al., "An Efficient Adversarial Attack for Tree Ensembles"
    has_better_neighborhood = True
    while has_better_neighborhood and current_iteration < max_iteration:
        next_status = attack_loop(
            victim_x,
            dist,
            status,
            model,
            victim_y,
            box_adv,
            sorted_box,
            constraints,
            visited_leaf_tuples,
            time_limit,
            use_t_bound=use_t_bound,
        )
        if next_status is not None and next_status.r < status.r:
            has_better_neighborhood = True
            status.r = next_status.r
            status.C.leaves[next_status.tree_index] = next_status.leaf
            status.x = next_status.x
            visited_leaf_tuples.add(status.C.get_indexes())
            sorted_box.replace_box(next_status.tree_index, next_status.leaf.box)
            logger.debug(
                f"continue / visited: {len(visited_leaf_tuples)}/ new_dist: {dist(victim_x, status.x)}",
            )
        else:
            has_better_neighborhood = False
        current_iteration += 1
    logger.debug(f"{dist(victim_x, x0):.2f} -> {dist(victim_x, status.x):.2f}")
    return status.x
