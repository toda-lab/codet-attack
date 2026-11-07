"""Constraints for the LCLD dataset."""

import logging as _logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from constrained_attacks.constraints.relation_constraint import (
    BaseRelationConstraint,
    Constant,
    Feature,
    SafeDivision,
)
from pyscipopt import Model, Variable
from typing_extensions import override
from z3 import ArithRef, Solver

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
class LcldMeta:
    loan_amnt: Item = Item(0, 1000.0, 40000.0, mutable=True, type="real")
    term: Item = Item(1, 36, 60, mutable=False, type="int")
    int_rate: Item = Item(2, 5.31, 30.99, mutable=False, type="real")
    installment: Item = Item(3, 4.93, 1719.83, mutable=True, type="real")
    sub_grade: Item = Item(4, 0, 34, mutable=False, type="int")
    emp_length: Item = Item(5, 0, 10, mutable=True, type="int")
    home_ownership: Item = Item(6, 0, 5, mutable=False, type="cat")
    annual_inc: Item = Item(7, 33.0, 10999200.0, mutable=True, type="real")
    verification_status: Item = Item(8, 0, 2, mutable=False, type="cat")
    issue_d: Item = Item(9, None, None, mutable=False, type="date")
    purpose: Item = Item(10, 0, 13, mutable=True, type="cat")
    dti: Item = Item(11, -1.0, 999.0, mutable=True, type="real")
    open_acc: Item = Item(12, 1, 90, mutable=True, type="int")
    pub_rec: Item = Item(13, 0.0, 86.0, mutable=False, type="int")
    revol_bal: Item = Item(14, 0.0, 2904836.0, mutable=True, type="real")
    revol_util: Item = Item(15, 0.0, 892.3, mutable=True, type="real")
    total_acc: Item = Item(16, 2, 176, mutable=True, type="int")
    initial_list_status: Item = Item(17, 0, 1, mutable=False, type="cat")
    application_type: Item = Item(18, 0, 1, mutable=True, type="cat")
    mort_acc: Item = Item(19, 0, 51, mutable=True, type="int")
    pub_rec_bankruptcies: Item = Item(20, 0, 12, mutable=False, type="int")
    fico_score: Item = Item(21, 662.0, 847.5, mutable=False, type="real")
    month_of_year: Item = Item(22, 0, 11, mutable=False, type="int")
    ratio_loan_amnt_annual_inc: Item = Item(
        23,
        0.0001714285714285,
        310.6060606060606,
        mutable=True,
        type="real",
    )
    ratio_open_acc_total_acc: Item = Item(
        24,
        0.024390243902439,
        1.0,
        mutable=True,
        type="real",
    )
    month_since_earliest_cr_line: Item = Item(25, 36, 999, mutable=False, type="int")
    ratio_pub_rec_month_since_earliest_cr_line: Item = Item(
        26,
        0.0,
        0.4959349593495935,
        mutable=False,
        type="real",
    )
    ratio_pub_rec_bankruptcies_month_since_earliest_cr_line: Item = Item(
        27,
        0.0,
        0.0909090909090909,
        mutable=False,
        type="real",
    )
    ratio_pub_rec_bankruptcies_pub_rec: Item = Item(
        28,
        -1.0,
        1.0,
        mutable=False,
        type="real",
    )


lcld_meta = LcldMeta()
lcld_meta_dict = {k: Item(**v) for (k, v) in asdict(lcld_meta).items()}

categorical_index = [
    item.idx for item in lcld_meta_dict.values() if item.type in ("int", "cat")
]
immutable_index = [item.idx for item in lcld_meta_dict.values() if not item.mutable]


# relational constraints
def g1(x: Point) -> bool:
    """Check that the installment matches the loan amortization formula (g1).

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    int_rate = x[lcld_meta.int_rate.idx] / 1200
    term = x[lcld_meta.term.idx]
    installment = (
        x[lcld_meta.loan_amnt.idx]
        * (int_rate * ((1 + int_rate) ** term))
        / ((1 + int_rate) ** term - 1)
    )
    actual = x[lcld_meta.installment.idx]
    ok = bool((abs(actual - installment) - 0.01) <= EPS)
    if not ok:
        logger.debug(
            "[g1] installment must match loan formula:"
            f" expected ~{installment:.4f}, got {actual:.4f}",
        )
        return False
    return True


# open_acc <= total_acc
def g2(x: Point) -> bool:
    """Check that open_acc <= total_acc (g2).

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    open_acc = x[lcld_meta.open_acc.idx]
    total_acc = x[lcld_meta.total_acc.idx]
    ok = bool(open_acc <= total_acc)
    if not ok:
        logger.debug(
            "[g2] open_acc <= total_acc:"
            f"got open_acc={open_acc:.4f}, total_acc={total_acc:.4f}",
        )
        return False
    return True


# pub_rec_bankrupcies <= pub_rec
def g3(x: Point) -> bool:
    """Check that pub_rec_bankruptcies <= pub_rec (g3).

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    pub_bankrupt = x[lcld_meta.pub_rec_bankruptcies.idx]
    pub_rec = x[lcld_meta.pub_rec.idx]
    return bool(pub_bankrupt <= pub_rec)


# term == 36 || term == 60
def g4(x: Point) -> bool:
    """Check that term is either 36 or 60 (g4).

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    term = x[lcld_meta.term.idx]
    possible_terms = (36, 60)
    return bool(term in possible_terms)


# ratio_loan_amnt_annual_inc * annual_inc == loan_amnt
def g5(x: Point) -> bool:
    """Check that ratio_loan_amnt_annual_inc equals loan_amnt / annual_inc (g5).

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    actual = x[lcld_meta.ratio_loan_amnt_annual_inc.idx]
    expected = x[lcld_meta.loan_amnt.idx] / x[lcld_meta.annual_inc.idx]
    return bool(abs(actual - expected) < EPS)  # Floating point safe check


# ration_open_acc_total_acc * total_acc == open_acc
def g6(x: Point) -> bool:
    """Check that ratio_open_acc_total_acc equals open_acc / total_acc (g6).

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    actual = x[lcld_meta.ratio_open_acc_total_acc.idx]
    expected = x[lcld_meta.open_acc.idx] / x[lcld_meta.total_acc.idx]
    return bool(abs(actual - expected) < EPS)


# ration_pub_rec_month_since_earliest_cr_line * month_since_earliest_cr_line = pub_rec
def g8(x: Point) -> bool:
    """Check that ratio_pub_rec_month_since_earliest_cr_line equals pub_rec / month_since_earliest_cr_line (g8).

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    actual = x[lcld_meta.ratio_pub_rec_month_since_earliest_cr_line.idx]
    expected = x[lcld_meta.pub_rec.idx] / x[lcld_meta.month_since_earliest_cr_line.idx]
    return bool(abs(actual - expected) < EPS)


# ratio_pub_rec_bankruptcies_month_since_earliest_cr_line * month_since_earliest_cr_line
# == pub_rec_bankruptcies
def g9(x: Point) -> bool:
    """Check that ratio_pub_rec_bankruptcies_month_since_earliest_cr_line equals pub_rec_bankruptcies / month_since_earliest_cr_line (g9).

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    actual = x[lcld_meta.ratio_pub_rec_bankruptcies_month_since_earliest_cr_line.idx]
    expected = (
        x[lcld_meta.pub_rec_bankruptcies.idx]
        / x[lcld_meta.month_since_earliest_cr_line.idx]
    )
    return bool(abs(actual - expected) < EPS)


# ratio_pub_rec_bankruptcies_pub_rec == (pub_rec_bankruptcies / pub_rec || -1)
def g10(x: Point) -> bool:
    """Check that ratio_pub_rec_bankruptcies_pub_rec equals pub_rec_bankruptcies / pub_rec, or -1 if pub_rec is 0 (g10).

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    actual = x[lcld_meta.ratio_pub_rec_bankruptcies_pub_rec.idx]
    expected = (
        x[lcld_meta.pub_rec_bankruptcies.idx] / x[lcld_meta.pub_rec.idx]
        if x[lcld_meta.pub_rec.idx] != 0
        else -1
    )
    return bool(abs(actual - expected) < EPS)


def g_type(x: Point) -> bool:
    """Check that all categorical features of the LCLD dataset take integer values.

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
    """Check that every feature of the LCLD dataset lies within its [min, max] range.

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    for col_name, item in lcld_meta_dict.items():
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
    """Check that all immutable features of the LCLD dataset are unchanged.

    Args:
        victim_x: Original point being attacked.
        x: Modified point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    for item in lcld_meta_dict.values():
        if item.mutable:
            continue
        if abs(x[item.idx] - victim_x[item.idx]) > EPS:
            return False
    return True


def bound_box(victim_x: Point) -> BoundingBox:
    """Build the bounding box of the LCLD dataset around the victim point.

    Mutable features are bounded by their [min, max] range and immutable features are fixed to the victim's value.

    Args:
        victim_x: Original point being attacked.

    Returns:
        Bounding box in which adversarial points can exist.

    """
    lower = [-float("inf")] * victim_x.shape[0]
    upper = [float("inf")] * victim_x.shape[0]
    is_set = [False] * victim_x.shape[0]
    for item in lcld_meta_dict.values():
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


class LcldCostraints(BaseConstraints):
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
            g9,
            g10,
            g_type,
            g_min_max,
            lambda x_test: g_immutable(victim_x, x_test),
        ]
        self.bound_box = bound_box(victim_x)

    @override
    def get_categorial_index(self) -> list[int]:
        """Return the indices of discrete features of the LCLD dataset.

        Returns:
            List of feature indices.

        """
        return categorical_index

    @override
    def get_immutable_index(self) -> list[int]:
        """Return the indices of immutable features of the LCLD dataset.

        Returns:
            List of feature indices.

        """
        return immutable_index

    @override
    def get_bound_box(self) -> BoundingBox:
        """Return the region where adversarial points can exist for the LCLD dataset.

        Returns:
            Bounding box of the search region.

        """
        return self.bound_box

    @override
    def is_valid(self, x: Point) -> bool:
        """Check if x satisfies all constraints of the LCLD dataset.

        Args:
            x: Point to check.

        Returns:
            True if the constraint is satisfied, False otherwise.

        """
        return all(func(x) for func in self.relationship_consts)

    @override
    def set_smt_constraints(self, s: Solver, x: list[ArithRef]) -> None:
        """Add the relational constraints of the LCLD dataset to the SMT solver.

        Args:
            s: Z3 solver to add the constraints to.
            x: Z3 variables corresponding to the features.

        Returns:
            None.

        """
        idx_inst = lcld_meta.installment.idx
        idx_int_rate = lcld_meta.int_rate.idx
        idx_term = lcld_meta.term.idx
        idx_loan = lcld_meta.loan_amnt.idx
        int_rate_val = x[idx_int_rate] / 1200.0
        if self.victim_x[idx_term] == 36 or self.victim_x[idx_term] == 60:
            ir = int_rate_val
            if self.victim_x[idx_term] == 36:
                coef_c = (ir * (1 + ir) ** 36) / ((1 + ir) ** 36 - 1)
            else:
                coef_c = (ir * (1 + ir) ** 60) / ((1 + ir) ** 60 - 1)
            expected = x[idx_loan] * coef_c
            s.add(x[idx_inst] - expected <= 0.00999)
            s.add(expected - x[idx_inst] <= 0.00999)
        else:
            raise ValueError(f"term must be 36 or 60: got {x[idx_term]}")

        # g2
        idx_open = lcld_meta.open_acc.idx
        idx_total = lcld_meta.total_acc.idx
        s.add(x[idx_open] <= x[idx_total])

        # g3 （immutable）
        idx_pub_bank = lcld_meta.pub_rec_bankruptcies.idx
        idx_pub = lcld_meta.pub_rec.idx
        s.add(x[idx_pub_bank] <= x[idx_pub])

        # g4（immutable）

        # g5
        idx_ratio_loan = lcld_meta.ratio_loan_amnt_annual_inc.idx
        idx_inc = lcld_meta.annual_inc.idx
        diff5 = x[idx_ratio_loan] * x[idx_inc] - x[idx_loan]
        tol5 = 0.9e-4 * x[idx_inc]
        s.add(diff5 <= tol5)
        s.add(-diff5 <= tol5)

        # g6
        idx_ratio_open = lcld_meta.ratio_open_acc_total_acc.idx
        diff6 = x[idx_ratio_open] * x[idx_total] - x[idx_open]
        tol6 = 0.9e-4 * x[idx_total]
        s.add(diff6 <= tol6)
        s.add(-diff6 <= tol6)

        # g8 (immutable)
        idx_month = lcld_meta.month_since_earliest_cr_line.idx
        idx_ratio_pub_month = lcld_meta.ratio_pub_rec_month_since_earliest_cr_line.idx
        diff8 = x[idx_ratio_pub_month] * x[idx_month] - x[idx_pub]
        tol8 = 0.9e-4 * x[idx_month]
        s.add(diff8 <= tol8)
        s.add(-diff8 <= tol8)

        # g9 (immutable)
        idx_ratio_pub_bank_month = (
            lcld_meta.ratio_pub_rec_bankruptcies_month_since_earliest_cr_line.idx
        )
        diff9 = x[idx_ratio_pub_bank_month] * x[idx_month] - x[idx_pub_bank]
        tol9 = 0.9e-4 * x[idx_month]
        s.add(diff9 <= tol9)
        s.add(-diff9 <= tol9)

        # g10 (immutable)
        idx_ratio_pub_bank_pub = lcld_meta.ratio_pub_rec_bankruptcies_pub_rec.idx
        diff10 = x[idx_ratio_pub_bank_pub] * x[idx_pub] - x[idx_pub_bank]
        tol10 = 0.9e-4 * x[idx_pub]
        s.add(diff10 <= tol10)
        s.add(-diff10 <= tol10)

    @override
    def set_scip_constraints(self, s: Model, x: list[Variable]) -> None:
        """Add the relational constraints of the LCLD dataset to the SCIP model.

        Args:
            s: SCIP model to add the constraints to.
            x: SCIP variables corresponding to the features.

        Returns:
            None.

        """
        idx_open_acc = lcld_meta.open_acc.idx
        idx_total_acc = lcld_meta.total_acc.idx
        idx_pub_rec_bank_bankruptcies = lcld_meta.pub_rec_bankruptcies.idx
        idx_pub_rec = lcld_meta.pub_rec.idx
        index_loan_amnt_annual_inc = lcld_meta.ratio_loan_amnt_annual_inc.idx
        idx_loan_amnt = lcld_meta.loan_amnt.idx
        idx_annual_inc = lcld_meta.annual_inc.idx
        idx_month_since_earliest_cr_line = lcld_meta.month_since_earliest_cr_line.idx
        idx_ratio_pub_rec_month_since_earliest_cr_line = (
            lcld_meta.ratio_pub_rec_month_since_earliest_cr_line.idx
        )
        idx_ratio_pub_rec_bankruptcies_month_since_earliest_cr_line = (
            lcld_meta.ratio_pub_rec_bankruptcies_month_since_earliest_cr_line.idx
        )
        idx_ratio_pub_rec_bankruptcies_pub_rec = (
            lcld_meta.ratio_pub_rec_bankruptcies_pub_rec.idx
        )
        idx_term = lcld_meta.term.idx
        idx_installment = lcld_meta.installment.idx
        idx_int_rate = lcld_meta.int_rate.idx
        idx_term = lcld_meta.term.idx
        idx_ratio_open_acc_total_acc = lcld_meta.ratio_open_acc_total_acc.idx

        # Add constraints

        # g1
        int_rate_val = self.victim_x[idx_int_rate] / 1200.0
        term_val = self.victim_x[idx_term]
        if term_val != 0 and int_rate_val != 0:
            coef = (int_rate_val * ((1 + int_rate_val) ** term_val)) / (
                (1 + int_rate_val) ** term_val - 1
            )
            expected_inst = x[idx_loan_amnt] * coef
            s.addCons(x[idx_installment] - expected_inst <= EPS)
            s.addCons(expected_inst - x[idx_installment] <= EPS)

        # g2
        s.addCons(x[idx_open_acc] <= x[idx_total_acc])

        # g3
        s.addCons(
            x[idx_pub_rec_bank_bankruptcies] <= x[idx_pub_rec],
        )

        # g4

        # g5
        diff5 = x[index_loan_amnt_annual_inc] * x[idx_annual_inc] - x[idx_loan_amnt]
        tol5 = 1e-5 * x[idx_annual_inc]
        s.addCons(diff5 <= tol5)
        s.addCons(-diff5 <= tol5)

        # g6
        diff6 = x[idx_ratio_open_acc_total_acc] * x[idx_total_acc] - x[idx_open_acc]
        tol6 = 1e-5 * x[idx_total_acc]
        s.addCons(diff6 <= tol6)
        s.addCons(-diff6 <= tol6)

        # g8
        diff8 = (
            x[idx_ratio_pub_rec_month_since_earliest_cr_line]
            * x[idx_month_since_earliest_cr_line]
            - x[idx_pub_rec]
        )
        tol8 = 1e-5 * x[idx_month_since_earliest_cr_line]
        s.addCons(diff8 <= tol8)
        s.addCons(-diff8 <= tol8)

        # g9
        diff9 = (
            x[idx_ratio_pub_rec_bankruptcies_month_since_earliest_cr_line]
            * x[idx_month_since_earliest_cr_line]
            - x[idx_pub_rec_bank_bankruptcies]
        )
        tol9 = 1e-5 * x[idx_month_since_earliest_cr_line]
        s.addCons(diff9 <= tol9)
        s.addCons(-diff9 <= tol9)

        # g10
        diff10 = (
            x[idx_ratio_pub_rec_bankruptcies_pub_rec] * x[idx_pub_rec]
            - x[idx_pub_rec_bank_bankruptcies]
        )
        tol10 = 1e-5 * x[idx_pub_rec]
        s.addCons(diff10 <= tol10)
        s.addCons(-diff10 <= tol10)

    @override
    def get_moeva_constraints(self) -> list[BaseRelationConstraint]:
        """Return the relational constraints of the LCLD dataset for Moeva2.

        Returns:
            List of relation constraints.

        """
        idx_loan = lcld_meta.loan_amnt.idx
        idx_inc = lcld_meta.annual_inc.idx
        idx_open = lcld_meta.open_acc.idx
        idx_total = lcld_meta.total_acc.idx
        idx_inst = lcld_meta.installment.idx
        idx_int_rate = lcld_meta.int_rate.idx
        idx_term = lcld_meta.term.idx
        idx_ratio_loan = lcld_meta.ratio_loan_amnt_annual_inc.idx
        idx_ratio_open = lcld_meta.ratio_open_acc_total_acc.idx
        idx_pub = lcld_meta.pub_rec.idx
        idx_pub_bank = lcld_meta.pub_rec_bankruptcies.idx
        idx_month = lcld_meta.month_since_earliest_cr_line.idx
        idx_ratio_pub_month = lcld_meta.ratio_pub_rec_month_since_earliest_cr_line.idx
        idx_ratio_pub_bank_month = (
            lcld_meta.ratio_pub_rec_bankruptcies_month_since_earliest_cr_line.idx
        )
        idx_ratio_pub_bank_pub = lcld_meta.ratio_pub_rec_bankruptcies_pub_rec.idx

        # g1
        rate = Feature(idx_int_rate) / Constant(1200.0)
        term = Feature(idx_term)
        one = Constant(1.0)
        coef = (rate * ((one + rate) ** term)) / (((one + rate) ** term) - one)
        expected_inst = Feature(idx_loan) * coef
        g1_upper = (Feature(idx_inst) - expected_inst) <= Constant(1e-4)
        g1_lower = (expected_inst - Feature(idx_inst)) <= Constant(1e-4)

        # g2
        g2 = Feature(idx_open) <= Feature(idx_total)

        # g3
        g3 = Feature(idx_pub_bank) <= Feature(idx_pub)

        # g4
        term_diff = (Feature(idx_term) - Constant(36.0)) * (
            Feature(idx_term) - Constant(60.0)
        )
        g4_upper = term_diff <= Constant(1e-4)
        g4_lower = (Constant(0.0) - term_diff) <= Constant(1e-4)

        # g5
        div_g5 = SafeDivision(Feature(idx_loan), Feature(idx_inc), Constant(-1.0))
        g5_upper = (Feature(idx_ratio_loan) - div_g5) <= Constant(1e-4)
        g5_lower = (div_g5 - Feature(idx_ratio_loan)) <= Constant(1e-4)

        # g6
        div_g6 = SafeDivision(Feature(idx_open), Feature(idx_total), Constant(-1.0))
        g6_upper = (Feature(idx_ratio_open) - div_g6) <= Constant(1e-4)
        g6_lower = (div_g6 - Feature(idx_ratio_open)) <= Constant(1e-4)

        # g8 (multiplication to avoid div by zero)
        diff_g8 = Feature(idx_ratio_pub_month) * Feature(idx_month) - Feature(idx_pub)
        tol8 = Constant(1e-4) * Feature(idx_month)
        g8_upper = diff_g8 <= tol8
        g8_lower = (Constant(0.0) - diff_g8) <= tol8

        # g9
        diff_g9 = Feature(idx_ratio_pub_bank_month) * Feature(idx_month) - Feature(
            idx_pub_bank,
        )
        tol9 = Constant(1e-4) * Feature(idx_month)
        g9_upper = diff_g9 <= tol9
        g9_lower = (Constant(0.0) - diff_g9) <= tol9

        # g10
        diff_g10 = Feature(idx_ratio_pub_bank_pub) * Feature(idx_pub) - Feature(
            idx_pub_bank,
        )
        tol10 = Constant(1e-4) * Feature(idx_pub)
        g10_upper = diff_g10 <= tol10
        g10_lower = (Constant(0.0) - diff_g10) <= tol10

        return [
            g1_upper,
            g1_lower,
            g2,
            g3,
            g4_upper,
            g4_lower,
            g5_upper,
            g5_lower,
            g6_upper,
            g6_lower,
            g8_upper,
            g8_lower,
            g9_upper,
            g9_lower,
            g10_upper,
            g10_lower,
        ]
