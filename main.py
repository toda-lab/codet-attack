from smt_attack.scip_only import solve_exact
import contextlib
import logging as _logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np

from constraints.ctu import CtuCostraints
from constraints.lcld import LcldCostraints
from constraints.url import UrlCostraints
from constraints.wids import WidsCostraints
from smt_attack.constraints import BaseConstraints
from smt_attack.moeva2 import moeva
from smt_attack.point import Point
from smt_attack.scip_search import scip_attack
from smt_attack.smt_search import attack_smt
from smt_attack.tree import TreeEnsemble, parse_xgboost_json

logger = _logging.getLogger(__name__)

Attack: TypeAlias = Callable[[TreeEnsemble, Point, BaseConstraints], None]

@dataclass
class Case:
    model_path: str # Path to the model
    test_data_path: str # Path to the points to attack
    test_range: range # Indices of the points to attack
    # Factory for the constraints object, e.g. LcldCostraints from constraints.lcld
    constraints: Callable[[Point], BaseConstraints]
    # Directory to write results to; defaults to the current directory
    dump_dir: str | None = None




cases = [
    Case(
        "dataset/ctu/xgboost_100.json",
        "dataset/ctu/x_test.npy",
        range(1),
        CtuCostraints,
        "output",
    ),
    Case(
        "dataset/lcld/xgboost_100.json",
        "dataset/lcld/x_test.npy",
        range(1),
        LcldCostraints,
        "output",
    ),
    Case(
        "dataset/url/xgboost_100.json",
        "dataset/url/x_test.npy",
        range(1),
        UrlCostraints,
        "output",
    ),
    Case(
        "dataset/wids/xgboost_100.json",
        "dataset/wids/x_test.npy",
        range(1),
        WidsCostraints,
        "output",
    ),
]


def attack_proposal(model: TreeEnsemble,
    victim_x: Point,
    constraints: BaseConstraints) -> None:
    """Run the proposed method.

    Args:
        model: Tree ensemble model under attack.
        victim_x: Original point being attacked.
        constraints: Constraints that the adversarial point must satisfy.

    Returns:
        None.

    """
    print('# use_t_bound: False')
    attack_proposal_inner(model, victim_x, constraints, use_t_bound=False)

def attack_proposal_t_bound(model: TreeEnsemble,
    victim_x: Point,
    constraints: BaseConstraints) -> None:
    """Run the proposed method.

    Args:
        model: Tree ensemble model under attack.
        victim_x: Original point being attacked.
        constraints: Constraints that the adversarial point must satisfy.

    Returns:
        None.

    """
    print('# use_t_bound: True')
    attack_proposal_inner(model, victim_x, constraints, use_t_bound=True)


def attack_proposal_inner(
    model: TreeEnsemble,
    victim_x: Point,
    constraints: BaseConstraints,
    use_t_bound: bool,
) -> None:
    """Run the proposed method.

    Args:
        model: Tree ensemble model under attack.
        victim_x: Original point being attacked.
        constraints: Constraints that the adversarial point must satisfy.
        use_t_bound: Whether to restrict the search to trees whose boundary touches the projected point.

    Returns:
        None.

    """
    timer = time.time()
    # Attack with the SMT solver
    smt_result = attack_smt(victim_x, model, constraints)
    # Check for failure
    if smt_result is None:
        print("Final-SMT failed(smt)", flush=True)
        return
    if not constraints.is_valid(smt_result):
        print("Final-SMT failed(smt-const-violate)", flush=True)
        return
    # Search with SCIP
    scip_result = scip_attack(model, victim_x, smt_result, constraints, 1, use_t_bound=use_t_bound)
    # Check for failure

    if scip_result is None or (not constraints.is_valid(scip_result)):
        print(
            f"Final-SMT {float(np.linalg.norm(victim_x - smt_result)):.2f}"
            f" {time.time() - timer:.2f}",
            flush=True,
        )
        return
    print(
        f"Final-SMT {float(np.linalg.norm(victim_x - scip_result)):.2f}"
        f" {time.time() - timer:.2f}",
        flush=True,
    )

def attack_scip_one(model: TreeEnsemble, victim_x: Point, constraints: BaseConstraints) -> None:
    """Run the one-shot SCIP method.

    Args:
        model: Tree ensemble model under attack.
        victim_x: Original point being attacked.
        constraints: Constraints that the adversarial point must satisfy.

    Returns:
        None.

    """
    print("# SCIP_exact", flush=True)
    time_limit = 60
    timer = time.time()
    result = solve_exact(model, victim_x, constraints, time_limit=time_limit)
    if result is not None:
        print(
            f"Final-SCIP_exact {float(np.linalg.norm(victim_x - result)):.2f}"
            f" {time.time() - timer:.2f}",
            flush=True,
        )
        return

    print("Final-SCIP_exact failed(smt)", flush=True)



def attack_moeva2(
    model: TreeEnsemble,
    victim_x: Point,
    constraints: BaseConstraints,
) -> None:
    """Run Moeva2.

    Args:
        model: Tree ensemble model under attack.
        victim_x: Original point being attacked.
        constraints: Constraints that the adversarial point must satisfy.

    Returns:
        None.

    """
    print("# Moeva2")
    timer = time.time()
    result = moeva(model, victim_x, constraints)
    if result is None:
        print("Final-Moeva failed", flush=True)
    else:
        print(
            f"Final-Moeva {float(np.linalg.norm(victim_x - result)):.2f}"
            f" {time.time() - timer:.2f}",
            flush=True,
        )


attacks = [attack_proposal, attack_proposal_t_bound, attack_moeva2]



def run_case(case: Case) -> None:
    """Run all attacks on every test point of a case and write the results to a file.

    Args:
        case: Experiment setting to run.

    Returns:
        None.

    """
    tm = int(time.time())
    if not Path(case.model_path).exists():
        raise FileNotFoundError(f"{case.model_path} (model)  notfound")
    if not Path().exists():
        raise FileNotFoundError(f"{case.test_range} (test_data)  notfound")

    with open(case.test_data_path, "rb") as f:
        x_test: Point = np.load(f)
    model = parse_xgboost_json(case.model_path, x_test.shape[1])

    dump_to = Path() if case.dump_dir is None else Path(case.dump_dir)

    if dump_to.exists() and dump_to.is_file():
        raise FileExistsError(f"{dump_to} (output_dir specified) is file")

    dump_to.mkdir(exist_ok=True, parents=True)
    output_path = dump_to / f"{tm}.txt"
    with output_path.open("w") as f, contextlib.redirect_stdout(f):
        print(f"# setting: {case}", flush=True)
        print("# case adv_dist exec_time(sec)", flush=True)
        for point_idx in case.test_range:
            print(f"# index: {point_idx}")
            victim_x = x_test[point_idx]
            constraints = case.constraints(victim_x)
            for attack in attacks:
                attack(model, victim_x, constraints)


def main() -> None:
    """Run all configured cases.

    Returns:
        None.

    """
    _logging.basicConfig(level=_logging.INFO, format="%(message)s", stream=sys.stderr)

    for case in cases:
        run_case(case)


if __name__ == "__main__":
    main()
