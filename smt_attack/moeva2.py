"""Attack using Moeva2."""

import contextlib
import io
import logging as _logging
from collections.abc import Callable

import numpy as np
from constrained_attacks.attacks import Moeva2
from constrained_attacks.constraints.constraints import Constraints
from pymoo.config import Config as PymooConfig

from smt_attack.constraints import BaseConstraints
from smt_attack.point import Point
from smt_attack.tree import TreeEnsemble

logger = _logging.getLogger(__name__)

# Needs this to work due to numpy compatibility issues
# pyrefly: ignore [missing-attribute]
np.object = object  # noqa: NPY001
# pyrefly: ignore [missing-attribute]
np.float = float  # noqa: NPY001
# pyrefly: ignore [missing-attribute]
np.int = int  # noqa: NPY001

PymooConfig.show_compile_hint = False


def generate_attacker(
    model: TreeEnsemble,
    victim_x: Point,
    constraints: BaseConstraints,
) -> Moeva2:
    """Generate an instance to use for Moeva2 attack.

    Args:
        model: Tree ensemble model under attack.
        victim_x: Original point being attacked.
        constraints: Constraints that the adversarial point must satisfy.

    Returns:
        Configured Moeva2 attacker.

    """
    d = victim_x.shape[0]
    border_box = constraints.get_bound_box()

    feature_types = np.array(["real"] * d)
    intege_indexes = set(constraints.get_categorial_index())
    for i in range(d):
        if i in intege_indexes:
            feature_types[i] = "int"

    immutable_indexes = set(constraints.get_immutable_index())
    mutable_features = np.array([True] * d)
    for i in range(d):
        if i in immutable_indexes:
            mutable_features[i] = False

    relation_const = constraints.get_moeva_constraints()

    moeva_constraints = Constraints(
        feature_types=feature_types,
        mutable_features=mutable_features,
        lower_bounds=border_box.lower,
        upper_bounds=border_box.upper,
        relation_constraints=relation_const,
    )

    return Moeva2(
        model,
        moeva_constraints,
        norm=2,
        fun_distance_preprocess=lambda x: x,
        n_gen=100,
        save_history="full",
        seed=42,
        n_jobs=1,
        verbose=0,
    )


def moeva(
    model: TreeEnsemble,
    victim_x: Point,
    constraints: BaseConstraints,
    dist: Callable[[Point, Point], float] = lambda x, y: float(np.linalg.norm(x - y)),
) -> Point | None:
    """Perform attack using Moeva2.

    Args:
        model: Tree ensemble model under attack.
        victim_x: Original point being attacked.
        constraints: Constraints that the adversarial point must satisfy.
        dist: Distance function between two points.

    Returns:
        Closest valid adversarial point, or None if the attack failed.

    """
    victim_y = model.predict_one(victim_x)

    x_clean = np.array([victim_x])
    y_clean = np.array([victim_y])

    moeva2 = generate_attacker(model, victim_x, constraints)

    with contextlib.redirect_stdout(io.StringIO()):
        out = moeva2.generate(x_clean, y_clean)
    candidates = out[0][0]

    candidates = out[0][0]
    preds = model.predict(candidates)
    success_mask = preds != victim_y

    if not success_mask.any():
        logger.debug("moeva2 failed: no adversarial point")
        return None

    successful_candidates = candidates[success_mask]
    valid_candidates = [c for c in successful_candidates if constraints.is_valid(c)]

    if len(valid_candidates) == 0:
        logger.debug("moeva2 failed: no adversarial valid point")
        return None

    best_candidate = valid_candidates[0]
    best_dist = dist(best_candidate, victim_x)

    for candidate in valid_candidates:
        new_dist = dist(candidate, victim_x)
        if new_dist < best_dist:
            best_dist, best_candidate = new_dist, candidate

    return best_candidate
