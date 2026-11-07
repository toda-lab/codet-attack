"""Attack using SMT solver."""

import logging as _logging
import time

import numpy as np
from z3 import (
    And,
    ArithRef,
    Bool,
    If,
    Implies,
    Int,
    ModelRef,
    Real,
    Solver,
    Sum,
    sat,
)

from smt_attack.constraints import BaseConstraints
from smt_attack.point import Point
from smt_attack.tree import TreeEnsemble

logger = _logging.getLogger(__name__)


def model_to_point(x: list[ArithRef], m: ModelRef) -> Point:
    """Convert the solution of the SMT solver to a numpy array.

    Args:
        x: Z3 variables of the point.
        m: Z3 model holding the solution.

    Returns:
        Solution as a numpy array.

    """
    ans = []
    for e in x:
        try:
            v = m[e].as_long()
        except Exception:  # noqa: BLE001
            v = m[e].as_fraction()
        ans.append(float(v))
    return np.array(ans)


def attack_smt(  # noqa: C901, PLR0912, PLR0915
    victim_x: Point,
    model: TreeEnsemble,
    constraints: BaseConstraints,
) -> Point | None:
    """Create an instance of the solver with added constraints.

    Args:
        victim_x: Original point being attacked.
        model: Tree ensemble model under attack.
        constraints: Constraints that the adversarial point must satisfy.

    Returns:
        Adversarial point, or None if the solver found none.

    """
    s = Solver()
    bound_adv = constraints.get_bound_box()
    d = victim_x.shape[0]
    victim_y = model.predict_one(victim_x)
    categorical_index = set(constraints.get_categorial_index())

    x: list[ArithRef] = []
    for i in range(d):
        lower, upper = bound_adv.lower[i], bound_adv.upper[i]
        x.append(Int(f"x_{i}") if i in categorical_index else Real(f"x_{i}"))
        if lower == upper:
            s.add(x[i] == lower)
        else:
            if lower != -np.inf:
                s.add(float(lower) <= x[i])
            if upper != np.inf:
                s.add(x[i] <= float(upper))

    tree_outputs = []
    for tree_index, tree in enumerate(model.trees):
        z_vars = []
        leaf_vals = []

        for leaf in tree.leaves:
            if not (leaf.box & bound_adv).is_valid():
                continue
            z = Bool(f"z_{tree_index}_{leaf.node_id}")
            z_vars.append(z)
            leaf_vals.append(leaf.value)
            conds = []
            for xi in range(d):
                lower, upper = leaf.box.lower[xi], leaf.box.upper[xi]
                if lower != -np.inf:
                    conds.append(float(lower) <= x[xi])
                if upper != np.inf:
                    conds.append(x[xi] <= float(upper))
            # z == 1 => the modified point reaches that leaf
            assert len(conds) > 0
            s.add(Implies(z, And(*conds)))
        s.add(Sum(z_vars) == 1)
        t_out = Real(f"t_out_{tree_index}")
        s.add(
            t_out
            == Sum(
                [If(z, val, 0.0) for (z, val) in zip(z_vars, leaf_vals, strict=True)],
            ),
        )
        tree_outputs.append(t_out)

    ensemble_output = Real("ensemble_output")
    s.add(ensemble_output == Sum(tree_outputs))

    if victim_y == 1:
        s.add(ensemble_output <= 0)
    else:
        s.add(ensemble_output >= 0)

    constraints.set_smt_constraints(s, x)

    upper = 1
    for _ in range(20):
        upper *= 2
        logger.debug(f"{upper = }")
        timer = time.time()
        response = s.check(And(*[abs(victim_x[i] - x[i]) <= upper for i in range(d)]))
        logger.debug(f"smt: {time.time() - timer:.3f}s")
        logger.debug(f"{response = }")
        if response == sat:
            return model_to_point(x, s.model())
        continue
    return None
