"""Constraints for the URL dataset."""

import logging as _logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from constrained_attacks.constraints.relation_constraint import (
    BaseRelationConstraint,
    Constant,
    Feature,
)
from pyscipopt import Model, Variable
from typing_extensions import override
from z3 import ArithRef, Or, Solver, Sum

from smt_attack.box import BoundingBox
from smt_attack.constraints import BaseConstraints
from smt_attack.point import Point

if TYPE_CHECKING:
    from collections.abc import Callable

logger = _logging.getLogger(__name__)

EPS = 1e-4


@dataclass(frozen=True)
class Item:
    idx: int
    min: float | None
    max: float | None
    mutable: bool
    type: Literal["real", "cat", "int", "date"]


@dataclass(frozen=True)
class UrlMeta:
    """Metadata extracted from the URL dataset"""

    length_url: Item = Item(0, 12.0, 1641.0, mutable=True, type="int")
    length_hostname: Item = Item(1, 4.0, 214.0, mutable=True, type="int")
    ip: Item = Item(2, 0.0, 1.0, mutable=True, type="int")
    nb_dots: Item = Item(3, 1.0, 24.0, mutable=True, type="int")
    nb_hyphens: Item = Item(4, 0.0, 43.0, mutable=True, type="int")
    nb_at: Item = Item(5, 0.0, 4.0, mutable=True, type="int")
    nb_qm: Item = Item(6, 0.0, 3.0, mutable=True, type="int")
    nb_and: Item = Item(7, 0.0, 19.0, mutable=True, type="int")
    nb_or: Item = Item(8, 0.0, 0.0, mutable=True, type="int")
    nb_eq: Item = Item(9, 0.0, 19.0, mutable=True, type="int")
    nb_underscore: Item = Item(10, 0.0, 18.0, mutable=True, type="int")
    nb_tilde: Item = Item(11, 0.0, 1.0, mutable=True, type="int")
    nb_percent: Item = Item(12, 0.0, 96.0, mutable=True, type="int")
    nb_slash: Item = Item(13, 2.0, 33.0, mutable=True, type="int")
    nb_star: Item = Item(14, 0.0, 1.0, mutable=True, type="int")
    nb_colon: Item = Item(15, 1.0, 7.0, mutable=True, type="int")
    nb_comma: Item = Item(16, 0.0, 4.0, mutable=True, type="int")
    nb_semicolumn: Item = Item(17, 0.0, 20.0, mutable=True, type="int")
    nb_dollar: Item = Item(18, 0.0, 6.0, mutable=True, type="int")
    nb_space: Item = Item(19, 0.0, 18.0, mutable=True, type="int")
    nb_www: Item = Item(20, 0.0, 2.0, mutable=True, type="int")
    nb_com: Item = Item(21, 0.0, 6.0, mutable=True, type="int")
    nb_dslash: Item = Item(22, 0.0, 1.0, mutable=True, type="int")
    http_in_path: Item = Item(23, 0.0, 4.0, mutable=True, type="int")
    https_token: Item = Item(24, 0.0, 1.0, mutable=True, type="int")
    ratio_digits_url: Item = Item(
        25,
        0.0,
        0.7238805970000001,
        mutable=True,
        type="real",
    )
    ratio_digits_host: Item = Item(26, 0.0, 0.8, mutable=True, type="real")
    punycode: Item = Item(27, 0.0, 1.0, mutable=True, type="int")
    port: Item = Item(28, 0.0, 1.0, mutable=True, type="int")
    tld_in_path: Item = Item(29, 0.0, 1.0, mutable=True, type="int")
    tld_in_subdomain: Item = Item(30, 0.0, 1.0, mutable=True, type="int")
    abnormal_subdomain: Item = Item(31, 0.0, 1.0, mutable=True, type="int")
    nb_subdomains: Item = Item(32, 1.0, 3.0, mutable=True, type="int")
    prefix_suffix: Item = Item(33, 0.0, 1.0, mutable=True, type="int")
    random_domain: Item = Item(34, 0.0, 1.0, mutable=True, type="int")
    shortening_service: Item = Item(35, 0.0, 1.0, mutable=True, type="int")
    path_extension: Item = Item(36, 0.0, 1.0, mutable=True, type="int")
    nb_redirection: Item = Item(37, 0.0, 6.0, mutable=True, type="int")
    nb_external_redirection: Item = Item(38, 0.0, 1.0, mutable=True, type="int")
    length_words_raw: Item = Item(39, 1.0, 106.0, mutable=True, type="int")
    char_repeat: Item = Item(40, 0.0, 146.0, mutable=True, type="int")
    shortest_words_raw: Item = Item(41, 1.0, 31.0, mutable=True, type="int")
    shortest_word_host: Item = Item(42, 1.0, 39.0, mutable=True, type="int")
    shortest_word_path: Item = Item(43, 0.0, 40.0, mutable=True, type="int")
    longest_words_raw: Item = Item(44, 2.0, 829.0, mutable=True, type="int")
    longest_word_host: Item = Item(45, 1.0, 62.0, mutable=True, type="int")
    longest_word_path: Item = Item(46, 0.0, 829.0, mutable=True, type="int")
    avg_words_raw: Item = Item(47, 2.0, 128.25, mutable=True, type="real")
    avg_word_host: Item = Item(48, 1.0, 39.0, mutable=True, type="real")
    avg_word_path: Item = Item(49, 0.0, 250.0, mutable=True, type="real")
    phish_hints: Item = Item(50, 0.0, 10.0, mutable=True, type="int")
    domain_in_brand: Item = Item(51, 0.0, 1.0, mutable=True, type="int")
    brand_in_subdomain: Item = Item(52, 0.0, 1.0, mutable=True, type="int")
    brand_in_path: Item = Item(53, 0.0, 1.0, mutable=True, type="int")
    suspecious_tld: Item = Item(54, 0.0, 1.0, mutable=True, type="int")
    statistical_report: Item = Item(55, 0.0, 2.0, mutable=True, type="int")
    whois_registered_domain: Item = Item(56, 0.0, 1.0, mutable=True, type="int")
    domain_registration_length: Item = Item(57, -1.0, 29829.0, mutable=True, type="int")
    domain_age: Item = Item(58, -12.0, 12874.0, mutable=True, type="int")
    web_traffic: Item = Item(59, 0.0, 10767986.0, mutable=True, type="int")
    dns_record: Item = Item(60, 0.0, 1.0, mutable=True, type="int")
    google_index: Item = Item(61, 0.0, 1.0, mutable=True, type="int")
    page_rank: Item = Item(62, 0.0, 10.0, mutable=True, type="int")


url_meta = UrlMeta()
url_meta_dict = {k: Item(**v) for (k, v) in asdict(url_meta).items()}

categorical_index = [
    item.idx for item in url_meta_dict.values() if item.type in ("int", "cat")
]
immutable_index = [item.idx for item in url_meta_dict.values() if not item.mutable]


# relational constraints
def apply_if_a_supp_zero_than_b_supp_zero(x: Point, i: int, j: int) -> bool:
    """Check the implication: x[i] == 0 implies x[j] > 0.

    Args:
        x: Point to check.
        i: Index of the feature in the premise.
        j: Index of the feature that must be positive.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    return x[i] >= 0 or x[j] > 0


def g1(x: Point) -> bool:
    """Check that x[1] <= x[0] (g1).

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    ok = bool(x[1] <= x[0])
    if not ok:
        logger.debug(f"[g1] x1 <= x0. got {x[1]=}, {x[0]=}")
        return False
    return True


def g2(x: Point) -> bool:
    """Check that x[3] + x[4] + ... + x[17] + 3 * x[19] <= x[0] (g2).

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    intermediate_sum = x[3]
    for i in range(4, 18):
        intermediate_sum += x[i]
    intermediate_sum += 3 * x[19]
    ok = bool(intermediate_sum <= x[0])
    if not ok:
        logger.debug(
            f"[g2]: intermediate_sum <= x[0], got: {intermediate_sum = } {x[0] = }",
        )
        return False
    return True


def g3(x: Point) -> bool:
    """Check that x[21] == 0 implies x[3] > 0 (g3).

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    ok = apply_if_a_supp_zero_than_b_supp_zero(x, 21, 3)
    if not ok:
        logger.debug(
            f"[g3] x[21] == 0 implies x[3] > 0, but got x[21]={x[21]}, x[3]={x[3]}",
        )
        return False
    return True


def g4(x: Point) -> bool:
    """Check that x[23] == 0 implies x[13] > 0 (g4).

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    ok = apply_if_a_supp_zero_than_b_supp_zero(x, 23, 13)
    if not ok:
        logger.debug(
            f"[g4] x[23] == 0 implies x[13] > 0, but got x[23]={x[23]}, x[13]={x[13]}",
        )
        return False
    return True


def g5(x: Point) -> bool:
    """Check that 3 * x[20] + 4 * x[21] + 2 * x[23] <= x[0] (g5).

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    lhs = 3 * x[20] + 4 * x[21] + 2 * x[23]
    ok = lhs <= x[0]
    if not ok:
        logger.debug(
            "[g5] 3*x[20] + 4*x[21] + 2*x[23] <= x[0] is violated:"
            f"lhs={lhs}, x[0]={x[0]}",
        )
        return False
    return True


def g6(x: Point) -> bool:
    """Check that x[19] == 0 implies x[25] > 0 (g6).

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    ok = apply_if_a_supp_zero_than_b_supp_zero(x, 19, 25)
    if not ok:
        logger.debug(
            f"[g6] x[19] == 0 implies x[25] > 0, but got x[19]={x[19]}, x[25]={x[25]}",
        )
        return False
    return True


def g8(x: Point) -> bool:
    """Check that x[2] == 0 implies x[25] > 0 (g8).

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    ok = apply_if_a_supp_zero_than_b_supp_zero(x, 2, 25)
    if not ok:
        logger.debug(
            f"[g8] x[2] == 0 implies x[25] > 0, but got x[2]={x[2]}, x[25]={x[25]}",
        )
        return False
    return True


def g10(x: Point) -> bool:
    """Check that x[28] == 0 implies x[25] > 0 (g10).

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    ok = apply_if_a_supp_zero_than_b_supp_zero(x, 28, 25)
    if not ok:
        logger.debug(
            f"[g10] x[28] == 0 implies x[25] > 0, but got x[28]={x[28]}, x[25]={x[25]}",
        )
        return False
    return True


def g11(x: Point) -> bool:
    """Check that x[31] == 0 implies x[26] > 0 (g11).

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    ok = apply_if_a_supp_zero_than_b_supp_zero(x, 31, 26)
    if not ok:
        logger.debug(
            f"[g11] x[31] == 0 implies x[26] > 0, but got x[31]={x[31]}, x[26]={x[26]}",
        )
        return False
    return True


def g12(x: Point) -> bool:
    """Check that x[38] <= x[37] (g12).

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    ok = x[38] <= x[37]
    if not ok:
        logger.debug(
            f"[g12] x[38] <= x[37] is violated: x[38]={x[38]}, x[37]={x[37]}",
        )
        return False
    return True


def g13(x: Point) -> bool:
    """Check that 3 * x[20] <= x[0] + 1 (g13).

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    ok = bool(3 * x[20] <= x[0] + 1)
    if not ok:
        logger.debug(
            f"[g13] 3*x[20] <= x[0] + 1 is violated: lhs={3 * x[20]}, x[0]={x[0]}",
        )
        return False
    return True


def g14(x: Point) -> bool:
    """Check that 4 * x[21] <= x[0] + 1 (g14).

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    ok = bool(4 * x[21] <= x[0] + 1)
    if not ok:
        logger.debug(
            f"[g14] 4*x[21] <= x[0] + 1 is violated: lhs={4 * x[21]}, x[0]={x[0]}",
        )
        return False
    return True


def g15(x: Point) -> bool:
    """Check that 4 * x[2] <= x[0] + 1 (g15).

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    ok = bool(4 * x[2] <= x[0] + 1)
    if not ok:
        logger.debug(
            f"[g15] 4*x[2] <= x[0] + 1 is violated: lhs={4 * x[2]}, x[0]={x[0]}",
        )
        return False
    return True


def g16(x: Point) -> bool:
    """Check that 2 * x[23] <= x[0] + 1 (g16).

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    ok = bool(2 * x[23] <= x[0] + 1)
    if not ok:
        logger.debug(
            f"[g16] 2*x[23] <= x[0] + 1 is violated: lhs={2 * x[23]}, x[0]={x[0]}",
        )
        return False
    return True


def g_type(x: Point) -> bool:
    """Check that all categorical features of the URL dataset take integer values.

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    for idx in categorical_index:
        round_value = round(x[idx])
        if abs(x[idx] - round_value) > EPS:
            logger.debug(f"[g_type] x[{idx}] must be integer: got {x[idx]}")
            return False
    return True


def g_min_max(x: Point) -> bool:
    """Check that every feature of the URL dataset lies within its [min, max] range.

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    for col_name, col_item in asdict(url_meta).items():
        item = Item(**col_item)
        if item.min is None or item.max is None:
            continue
        if not (item.min <= x[item.idx] <= item.max):
            logger.debug(
                f"[g_min_max] {col_name} value {x[item.idx]} out of bounds "
                f"[{item.min}, {item.max}]",
            )
            return False
    return True


def g_immutable(victim_x: Point, x: Point) -> bool:
    """Check that all immutable features of the URL dataset are unchanged.

    Args:
        victim_x: Original point being attacked.
        x: Modified point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    for col_item in asdict(url_meta).values():
        item = Item(**col_item)
        if item.mutable:
            continue
        if abs(x[item.idx] - victim_x[item.idx]) > EPS:
            return False
    return True


def bound_box(victim_x: Point) -> BoundingBox:
    """Build the bounding box of the URL dataset around the victim point.

    Mutable features are bounded by their [min, max] range and immutable features are fixed to the victim's value.

    Args:
        victim_x: Original point being attacked.

    Returns:
        Bounding box in which adversarial points can exist.

    """
    lower = [-float("inf")] * victim_x.shape[0]
    upper = [float("inf")] * victim_x.shape[0]
    is_set = [False] * victim_x.shape[0]
    for item in url_meta_dict.values():
        if item.mutable:
            assert item.min is not None
            assert item.max is not None
            lower[item.idx] = item.min
            upper[item.idx] = item.max
        else:
            lower[item.idx] = victim_x[item.idx]
            upper[item.idx] = victim_x[item.idx]
        is_set[item.idx] = True

    # Prevent forgetting to set the ranges
    assert all(is_set)
    return BoundingBox(np.array(lower), np.array(upper))


def apply_if_a_supp_zero_than_b_supp_zero_moeva(
    a: Feature,
    b: Feature,
) -> BaseRelationConstraint:
    """Build the Moeva2 relation constraint for: a == 0 implies b > 0.

    Args:
        a: Feature in the premise.
        b: Feature that must be positive.

    Returns:
        Relation constraint expressing the implication.

    """
    return (Constant(0) <= a) | (Constant(0) < b)


class UrlCostraints(BaseConstraints):
    def __init__(self, victim_x: Point) -> None:
        """Initialize the constraints for the given victim point.

        Args:
            victim_x: Original point being attacked.

        """
        super().__init__(victim_x)
        self.relationship_consts: list[Callable[[Point], bool]] = [
            g1,
            g2,
            g3,
            g4,
            g5,
            g6,
            g8,
            g10,
            g11,
            g12,
            g13,
            g14,
            g_type,
            g_min_max,
            lambda x_test: g_immutable(victim_x, x_test),
        ]
        self.bound_box = bound_box(victim_x)

    @override
    def get_categorial_index(self) -> list[int]:
        """Return the indices of discrete features of the URL dataset.

        Returns:
            List of feature indices.

        """
        return categorical_index

    @override
    def get_immutable_index(self) -> list[int]:
        """Return the indices of immutable features of the URL dataset.

        Returns:
            List of feature indices.

        """
        return immutable_index

    @override
    def get_bound_box(self) -> BoundingBox:
        """Return the region where adversarial points can exist for the URL dataset.

        Returns:
            Bounding box of the search region.

        """
        return self.bound_box

    @override
    def is_valid(self, x: Point) -> bool:
        """Check if x satisfies all constraints of the URL dataset.

        Args:
            x: Point to check.

        Returns:
            True if the constraint is satisfied, False otherwise.

        """
        return all(func(x) for func in self.relationship_consts)

    @override
    def set_smt_constraints(self, s: Solver, x: list[ArithRef]) -> None:
        """Add the relational constraints of the URL dataset to the SMT solver.

        Args:
            s: Z3 solver to add the constraints to.
            x: Z3 variables corresponding to the features.

        Returns:
            None.

        """
        s.add(x[1] <= x[0])

        s.add(Sum(x[i] for i in range(3, 18)) + 3 * x[19] <= x[0])

        s.add(Or(x[21] >= 0, x[3] > 0))

        s.add(Or(x[23] >= 0, x[13] > 0))

        s.add(3 * x[20] + 4 * x[21] + 2 * x[23] <= x[0])

        s.add(Or(x[19] >= 0, x[25] > 0))

        s.add(Or(x[2] >= 0, x[25] > 0))

        s.add(Or(x[28] >= 0, x[25] > 0))

        s.add(Or(x[31] >= 0, x[26] > 0))

        s.add(x[38] <= x[37])

        s.add(3 * x[20] <= x[0] + 1)
        s.add(4 * x[21] <= x[0] + 1)
        s.add(4 * x[2] <= x[0] + 1)
        s.add(2 * x[23] <= x[0] + 1)

    @override
    def set_scip_constraints(self, s: Model, x: list[Variable]) -> None:

        """Add the relational constraints of the URL dataset to the SCIP model.

        Args:
            s: SCIP model to add the constraints to.
            x: SCIP variables corresponding to the features.

        Returns:
            None.

        """
        s.addCons(x[1] <= x[0])
        s.addCons(sum(x[i] for i in range(3, 18)) + 3 * x[19] <= x[0])

        s.addConsDisjunction([x[21] >= 0, x[3] >= EPS])

        s.addConsDisjunction([x[23] >= 0, x[13] >= EPS])

        s.addCons(3 * x[20] + 4 * x[21] + 2 * x[23] <= x[0])

        s.addConsDisjunction([x[19] >= 0, x[25] >= EPS])

        s.addConsDisjunction([x[2] >= 0, x[25] >= EPS])

        s.addConsDisjunction([x[28] >= 0, x[25] >= EPS])

        s.addConsDisjunction([x[31] >= 0, x[26] >= EPS])

        s.addCons(x[38] <= x[37])

        s.addCons(3 * x[20] <= x[0] + 1)

        s.addCons(4 * x[21] <= x[0] + 1)

        s.addCons(4 * x[2] <= x[0] + 1)

        s.addCons(2 * x[23] <= x[0] + 1)

    @override
    def get_moeva_constraints(self) -> list[BaseRelationConstraint]:
        """Return the relational constraints of the URL dataset for Moeva2.

        Returns:
            List of relation constraints.

        """
        g1 = Feature(1) <= Feature(0)

        intermediate_sum = Feature(3)
        for i in range(4, 18):
            intermediate_sum = intermediate_sum + Feature(i)
        intermediate_sum = intermediate_sum + (Constant(3) * Feature(19))

        g2 = intermediate_sum <= Feature(0)

        # g3: if x[:, 21] > 0 then x[:,3] > 0
        g3 = apply_if_a_supp_zero_than_b_supp_zero_moeva(Feature(21), Feature(3))

        # g4: if x[:, 23] > 0 then x[:,13] > 0
        g4 = apply_if_a_supp_zero_than_b_supp_zero_moeva(Feature(23), Feature(13))

        intermediate_sum = (
            Constant(3) * Feature(20)
            + Constant(4) * Feature(21)
            + Constant(2) * Feature(23)
        )
        g5 = intermediate_sum <= Feature(0)

        # g6: if x[:, 19] > 0 then x[:,25] > 0
        g6 = apply_if_a_supp_zero_than_b_supp_zero_moeva(Feature(19), Feature(25))

        # g8: if x[:, 2] > 0 then x[:,25] > 0
        g8 = apply_if_a_supp_zero_than_b_supp_zero_moeva(Feature(2), Feature(25))

        # g10: if x[:, 28] > 0 then x[:,25] > 0
        g10 = apply_if_a_supp_zero_than_b_supp_zero_moeva(Feature(28), Feature(25))

        # g11: if x[:, 31] > 0 then x[:,26] > 0
        g11 = apply_if_a_supp_zero_than_b_supp_zero_moeva(Feature(31), Feature(26))

        # g12: x[:,38] <= x[:,37]
        g12 = Feature(38) <= Feature(37)

        g13 = (Constant(3) * Feature(20)) <= (Feature(0) + Constant(1))

        g14 = (Constant(4) * Feature(21)) <= (Feature(0) + Constant(1))

        g15 = (Constant(4) * Feature(2)) <= (Feature(0) + Constant(1))
        g16 = (Constant(2) * Feature(23)) <= (Feature(0) + Constant(1))

        return [g1, g2, g3, g4, g5, g6, g8, g10, g11, g12, g13, g14, g15, g16]
