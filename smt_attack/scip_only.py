"""Exact adversarial attack using SCIP in one shot."""

import numpy as np
from pyscipopt import Model

from smt_attack.constraints import BaseConstraints
from smt_attack.point import Point
from smt_attack.tree import TreeEnsemble


def solve_exact(
    model: TreeEnsemble,
    victim_x: Point,
    constraints: BaseConstraints,
    time_limit: int = 600,
) -> Point | None:
    """Perform a constrained adversarial attack in one shot using SCIP.

    Args:
        model: Tree ensemble model under attack.
        victim_x: Original point being attacked.
        constraints: Constraints that the adversarial point must satisfy.
        time_limit: Time limit of the SCIP solver in seconds.

    Returns:
        Optimal adversarial point, or None if no solution was found.

    """
    d = victim_x.shape[0]
    victim_y0 = model.predict_one(victim_x)

    scip = Model("exact_milp")
    scip.hideOutput()

    box_const = constraints.get_bound_box()
    categorical_indexes = constraints.get_categorial_index()

    scip_vars = []
    for i in range(d):
        l_val = box_const.lower[i]
        r_val = box_const.upper[i]
        # SCIP expects None for unbounded sides
        lb = float(l_val) if l_val != -np.inf else None
        ub = float(r_val) if r_val != np.inf else None
        vtype = "C"
        if i in categorical_indexes:
            vtype = "I"
        var = scip.addVar(vtype=vtype, name=f"x_{i}", lb=lb, ub=ub)
        scip_vars.append(var)

    constraints.set_scip_constraints(scip, scip_vars)

    tree_outputs = []
    for tree_index, tree in enumerate(model.trees):
        z_vars = []
        leaf_vals = []

        for leaf in tree.leaves:
            if not (leaf.box & box_const).is_valid():
                continue
            # z == 1 iff the modified point falls into this leaf
            z = scip.addVar(vtype="B", name=f"z_{tree_index}_{leaf.node_id}")
            z_vars.append(z)
            leaf_vals.append(leaf.value)
            for xi in range(d):
                lower, upper = leaf.box.lower[xi], leaf.box.upper[xi]
                if lower != -np.inf:
                    scip.addConsIndicator(lower <= scip_vars[xi], binvar=z)
                if upper != np.inf:
                    scip.addConsIndicator(scip_vars[xi] <= upper, binvar=z)

        scip.addCons(sum(z_vars) == 1)
        t_out = scip.addVar(vtype="C", name=f"t_out_{tree_index}", lb=-np.inf)
        scip.addCons(
            t_out == sum(z * val for z, val in zip(z_vars, leaf_vals, strict=True)),
        )
        tree_outputs.append(t_out)

    ensemble_out = scip.addVar(vtype="C", name="ensemble_out", lb=-np.inf)
    scip.addCons(ensemble_out == sum(tree_outputs))

    if victim_y0 == 0:
        scip.addCons(ensemble_out >= 0)
    else:
        scip.addCons(ensemble_out <= 0)

    t = scip.addVar(vtype="C", name="obj_t", lb=0)
    obj_expr = sum((scip_vars[i] - victim_x[i]) * (scip_vars[i] - victim_x[i]) for i in range(d))
    scip.addCons(obj_expr <= t)
    scip.setObjective(t, "minimize")
    scip.setParam("limits/time", time_limit)

    scip.optimize()

    status = scip.getStatus()
    if status != "optimal":
        return None
    try:
        res_x = np.zeros(d)
        for i in range(d):
            res_x[i] = scip.getVal(scip_vars[i])
        return res_x
    except Exception:
        return None
