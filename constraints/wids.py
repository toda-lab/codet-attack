"""Constraints for the WiDS dataset."""

import logging as _logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from constrained_attacks.constraints.relation_constraint import (
    BaseRelationConstraint,
    Feature,
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
class WidsMeta:
    age: Item = Item(0, 16.0, 89.0, mutable=True, type="int")
    bmi: Item = Item(1, 14.8449, 67.815, mutable=True, type="real")
    elective_surgery: Item = Item(2, 0.0, 1.0, mutable=True, type="int")
    ethnicity: Item = Item(3, 1.0, 7.0, mutable=True, type="cat")
    gender: Item = Item(4, 1.0, 3.0, mutable=True, type="cat")
    height: Item = Item(5, 137.2, 195.59, mutable=True, type="real")
    hospital_admit_source: Item = Item(6, 1.0, 16.0, mutable=True, type="cat")
    icu_admit_source: Item = Item(7, 1.0, 6.0, mutable=True, type="cat")
    icu_stay_type: Item = Item(8, 1.0, 3.0, mutable=True, type="cat")
    icu_type: Item = Item(9, 1.0, 8.0, mutable=True, type="cat")
    pre_icu_los_days: Item = Item(10, -0.22431, 67.02361111, mutable=True, type="real")
    readmission_status: Item = Item(11, 0.0, 0.0, mutable=True, type="int")
    weight: Item = Item(12, 38.6, 186.0, mutable=True, type="real")
    apache_2_diagnosis: Item = Item(13, 101.0, 308.0, mutable=True, type="int")
    apache_3j_diagnosis: Item = Item(14, 101.01, 2201.05, mutable=True, type="real")
    apache_post_operative: Item = Item(15, 0.0, 1.0, mutable=True, type="int")
    arf_apache: Item = Item(16, 0.0, 1.0, mutable=True, type="int")
    bun_apache: Item = Item(17, 4.0, 127.0, mutable=True, type="real")
    creatinine_apache: Item = Item(18, 0.3, 11.18, mutable=True, type="real")
    gcs_eyes_apache: Item = Item(19, 1.0, 4.0, mutable=True, type="int")
    gcs_motor_apache: Item = Item(20, 1.0, 6.0, mutable=True, type="int")
    gcs_unable_apache: Item = Item(21, 0.0, 0.0, mutable=True, type="int")
    gcs_verbal_apache: Item = Item(22, 1.0, 5.0, mutable=True, type="int")
    glucose_apache: Item = Item(23, 39.0, 598.7, mutable=True, type="real")
    heart_rate_apache: Item = Item(24, 30.0, 178.0, mutable=True, type="int")
    hematocrit_apache: Item = Item(25, 16.2, 51.4, mutable=True, type="real")
    intubated_apache: Item = Item(26, 0.0, 1.0, mutable=True, type="int")
    map_apache: Item = Item(27, 40.0, 200.0, mutable=True, type="int")
    resprate_apache: Item = Item(28, 4.0, 60.0, mutable=True, type="real")
    sodium_apache: Item = Item(29, 117.0, 158.0, mutable=True, type="real")
    temp_apache: Item = Item(30, 32.1, 39.7, mutable=True, type="real")
    ventilated_apache: Item = Item(31, 0.0, 1.0, mutable=True, type="int")
    wbc_apache: Item = Item(32, 0.9, 45.8, mutable=True, type="real")
    d1_diasbp_max: Item = Item(33, 46.0, 165.0, mutable=True, type="int")
    d1_diasbp_min: Item = Item(34, 13.0, 90.0, mutable=True, type="int")
    d1_diasbp_noninvasive_max: Item = Item(35, 46.0, 165.0, mutable=True, type="int")
    d1_diasbp_noninvasive_min: Item = Item(36, 13.0, 90.0, mutable=True, type="int")
    d1_heartrate_max: Item = Item(37, 58.0, 177.0, mutable=True, type="int")
    d1_heartrate_min: Item = Item(38, 0.0, 143.0, mutable=True, type="int")
    d1_mbp_max: Item = Item(39, 60.0, 184.0, mutable=True, type="int")
    d1_mbp_min: Item = Item(40, 22.0, 112.0, mutable=True, type="int")
    d1_mbp_noninvasive_max: Item = Item(41, 60.0, 181.0, mutable=True, type="int")
    d1_mbp_noninvasive_min: Item = Item(42, 22.0, 112.0, mutable=True, type="int")
    d1_resprate_max: Item = Item(43, 14.0, 92.0, mutable=True, type="int")
    d1_resprate_min: Item = Item(44, 0.0, 72.0, mutable=True, type="int")
    d1_spo2_max: Item = Item(45, 13.0, 100.0, mutable=True, type="int")
    d1_spo2_min: Item = Item(46, 0.0, 100.0, mutable=True, type="int")
    d1_sysbp_max: Item = Item(47, 90.0, 232.0, mutable=True, type="int")
    d1_sysbp_min: Item = Item(48, 41.0, 160.0, mutable=True, type="int")
    d1_sysbp_noninvasive_max: Item = Item(49, 90.0, 232.0, mutable=True, type="int")
    d1_sysbp_noninvasive_min: Item = Item(50, 41.03, 160.0, mutable=True, type="real")
    d1_temp_max: Item = Item(51, 35.1, 39.9, mutable=True, type="real")
    d1_temp_min: Item = Item(52, 31.889, 37.8, mutable=True, type="real")
    h1_diasbp_max: Item = Item(53, 37.0, 143.0, mutable=True, type="int")
    h1_diasbp_min: Item = Item(54, 22.0, 113.0, mutable=True, type="int")
    h1_diasbp_noninvasive_max: Item = Item(55, 37.0, 144.0, mutable=True, type="int")
    h1_diasbp_noninvasive_min: Item = Item(56, 22.0, 114.0, mutable=True, type="int")
    h1_heartrate_max: Item = Item(57, 46.0, 164.0, mutable=True, type="int")
    h1_heartrate_min: Item = Item(58, 36.0, 144.0, mutable=True, type="int")
    h1_mbp_max: Item = Item(59, 49.0, 165.0, mutable=True, type="int")
    h1_mbp_min: Item = Item(60, 32.0, 138.0, mutable=True, type="int")
    h1_mbp_noninvasive_max: Item = Item(61, 49.0, 163.0, mutable=True, type="int")
    h1_mbp_noninvasive_min: Item = Item(62, 32.0, 138.0, mutable=True, type="int")
    h1_resprate_max: Item = Item(63, 10.0, 59.0, mutable=True, type="int")
    h1_resprate_min: Item = Item(64, 0.0, 118.0, mutable=True, type="int")
    h1_spo2_max: Item = Item(65, 10.0, 100.0, mutable=True, type="int")
    h1_spo2_min: Item = Item(66, 0.0, 100.0, mutable=True, type="int")
    h1_sysbp_max: Item = Item(67, 75.0, 223.0, mutable=True, type="int")
    h1_sysbp_min: Item = Item(68, 53.0, 194.0, mutable=True, type="int")
    h1_sysbp_noninvasive_max: Item = Item(69, 75.0, 223.0, mutable=True, type="int")
    h1_sysbp_noninvasive_min: Item = Item(70, 53.0, 195.0, mutable=True, type="int")
    h1_temp_max: Item = Item(71, 33.4, 39.5, mutable=True, type="real")
    h1_temp_min: Item = Item(72, 32.9, 39.3, mutable=True, type="real")
    d1_bun_max: Item = Item(73, 4.0, 126.0, mutable=True, type="real")
    d1_bun_min: Item = Item(74, 3.0, 113.09, mutable=True, type="real")
    d1_calcium_max: Item = Item(75, 6.2, 10.8, mutable=True, type="real")
    d1_calcium_min: Item = Item(76, 5.5, 10.3, mutable=True, type="real")
    d1_creatinine_max: Item = Item(77, 0.34, 11.11, mutable=True, type="real")
    d1_creatinine_min: Item = Item(78, 0.3, 9.9379, mutable=True, type="real")
    d1_glucose_max: Item = Item(79, 73.0, 611.0, mutable=True, type="int")
    d1_glucose_min: Item = Item(80, 33.0, 288.0, mutable=True, type="int")
    d1_hco3_max: Item = Item(81, 12.0, 40.0, mutable=True, type="real")
    d1_hco3_min: Item = Item(82, 7.0, 39.0, mutable=True, type="real")
    d1_hemaglobin_max: Item = Item(83, 6.8, 17.2, mutable=True, type="real")
    d1_hemaglobin_min: Item = Item(84, 5.3, 16.7, mutable=True, type="real")
    d1_hematocrit_max: Item = Item(85, 20.4, 51.5, mutable=True, type="real")
    d1_hematocrit_min: Item = Item(86, 16.1, 50.0, mutable=True, type="real")
    d1_platelets_max: Item = Item(87, 27.0, 585.0, mutable=True, type="int")
    d1_platelets_min: Item = Item(88, 18.55, 557.45, mutable=True, type="real")
    d1_potassium_max: Item = Item(89, 2.8, 7.0, mutable=True, type="real")
    d1_potassium_min: Item = Item(90, 2.4, 5.8, mutable=True, type="real")
    d1_sodium_max: Item = Item(91, 123.0, 158.0, mutable=True, type="real")
    d1_sodium_min: Item = Item(92, 117.0, 153.0, mutable=True, type="real")
    d1_wbc_max: Item = Item(93, 1.2, 46.08, mutable=True, type="real")
    d1_wbc_min: Item = Item(94, 0.9, 40.898, mutable=True, type="real")
    apache_4a_hospital_death_prob: Item = Item(
        95,
        -1.0,
        0.98,
        mutable=True,
        type="real",
    )
    apache_4a_icu_death_prob: Item = Item(96, -1.0, 0.97, mutable=True, type="real")
    aids: Item = Item(97, 0.0, 1.0, mutable=True, type="int")
    cirrhosis: Item = Item(98, 0.0, 1.0, mutable=True, type="int")
    diabetes_mellitus: Item = Item(99, 0.0, 1.0, mutable=True, type="int")
    hepatic_failure: Item = Item(100, 0.0, 1.0, mutable=True, type="int")
    immunosuppression: Item = Item(101, 0.0, 1.0, mutable=True, type="int")
    leukemia: Item = Item(102, 0.0, 1.0, mutable=True, type="int")
    lymphoma: Item = Item(103, 0.0, 1.0, mutable=True, type="int")
    solid_tumor_with_metastasis: Item = Item(104, 0.0, 1.0, mutable=True, type="int")
    apache_3j_bodysystem: Item = Item(105, 1.0, 11.0, mutable=True, type="cat")
    apache_2_bodysystem: Item = Item(106, 1.0, 10.0, mutable=True, type="cat")
    apache_3j_diagnosis_split0: Item = Item(107, 2.0, 112.0, mutable=True, type="cat")


wids_meta = WidsMeta()
wids_meta_dict = {k: Item(**v) for (k, v) in asdict(wids_meta).items()}

categorical_index = [
    item.idx for item in wids_meta_dict.values() if item.type in ("int", "cat")
]
immutable_index = [item.idx for item in wids_meta_dict.values() if not item.mutable]


# relational constraints
max_min_pair = [
    ("d1_diasbp_max", "d1_diasbp_min"),
    ("d1_diasbp_noninvasive_max", "d1_diasbp_noninvasive_min"),
    ("d1_heartrate_max", "d1_heartrate_min"),
    ("d1_mbp_max", "d1_mbp_min"),
    ("d1_mbp_noninvasive_max", "d1_mbp_noninvasive_min"),
    ("d1_resprate_max", "d1_resprate_min"),
    ("d1_spo2_max", "d1_spo2_min"),
    ("d1_sysbp_max", "d1_sysbp_min"),
    ("d1_sysbp_noninvasive_max", "d1_sysbp_noninvasive_min"),
    ("d1_temp_max", "d1_temp_min"),
    ("h1_diasbp_max", "h1_diasbp_min"),
    ("h1_diasbp_noninvasive_max", "h1_diasbp_noninvasive_min"),
    ("h1_heartrate_max", "h1_heartrate_min"),
    ("h1_mbp_max", "h1_mbp_min"),
    ("h1_mbp_noninvasive_max", "h1_mbp_noninvasive_min"),
    ("h1_resprate_max", "h1_resprate_min"),
    ("h1_spo2_max", "h1_spo2_min"),
    ("h1_sysbp_max", "h1_sysbp_min"),
    ("h1_sysbp_noninvasive_max", "h1_sysbp_noninvasive_min"),
    ("h1_temp_max", "h1_temp_min"),
    ("d1_bun_max", "d1_bun_min"),
    ("d1_calcium_max", "d1_calcium_min"),
    ("d1_creatinine_max", "d1_creatinine_min"),
    ("d1_glucose_max", "d1_glucose_min"),
    ("d1_hco3_max", "d1_hco3_min"),
    ("d1_hemaglobin_max", "d1_hemaglobin_min"),
    ("d1_hematocrit_max", "d1_hematocrit_min"),
    ("d1_platelets_max", "d1_platelets_min"),
    ("d1_potassium_max", "d1_potassium_min"),
    ("d1_sodium_max", "d1_sodium_min"),
    ("d1_wbc_max", "d1_wbc_min"),
]


def g_type(x: Point) -> bool:
    """Check that all categorical features of the WiDS dataset take integer values.

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
    """Check that every feature of the WiDS dataset lies within its [min, max] range.

    Args:
        x: Point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    for col_name, item in wids_meta_dict.items():
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
    """Check that all immutable features of the WiDS dataset are unchanged.

    Args:
        victim_x: Original point being attacked.
        x: Modified point to check.

    Returns:
        True if the constraint is satisfied, False otherwise.

    """
    for item in wids_meta_dict.values():
        if item.mutable:
            continue
        if abs(x[item.idx] - victim_x[item.idx]) > EPS:
            return False
    return True


def bound_box(victim_x: Point) -> BoundingBox:
    """Build the bounding box of the WiDS dataset around the victim point.

    Mutable features are bounded by their [min, max] range and immutable features are fixed to the victim's value.

    Args:
        victim_x: Original point being attacked.

    Returns:
        Bounding box in which adversarial points can exist.

    """
    lower = [-float("inf")] * victim_x.shape[0]
    upper = [float("inf")] * victim_x.shape[0]
    is_set = [False] * victim_x.shape[0]
    for item in wids_meta_dict.values():
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


class WidsCostraints(BaseConstraints):
    """Constraints for the WiDS dataset."""

    def __init__(self, victim_x: Point) -> None:
        """Initialize the constraints for the given victim point.

        Args:
            victim_x: Original point being attacked.

        """
        super().__init__(victim_x)
        self.relationship_consts: list[Callable[[Point], bool]] = [
            g_min_max,
            lambda x: g_immutable(victim_x, x),
            g_type,
        ]

        for _max_name, _min_name in max_min_pair:

            def g(
                x: Point,
                max_name: str = _max_name,
                min_name: str = _min_name,
            ) -> bool:
                """Check that the min feature does not exceed the max feature.

                Args:
                    x: Point to check.
                    max_name: Name of the max feature.
                    min_name: Name of the min feature.

                Returns:
                    True if the constraint is satisfied, False otherwise.

                """
                max_value = wids_meta_dict[max_name].idx
                min_value = wids_meta_dict[min_name].idx
                if not (x[min_value] <= x[max_value] + EPS):
                    logger.debug(
                        f"[g1] {min_name} must be lower than {max_name}:"
                        " got {x[min_value]} and {x[max_value]}"
                        " (indices {min_value}, {max_value})",
                    )
                    return False
                return True

            self.relationship_consts.append(g)

        self.bound_box = bound_box(victim_x)

    @override
    def get_categorial_index(self) -> list[int]:
        """Return the indices of discrete features of the WiDS dataset.

        Returns:
            List of feature indices.

        """
        return categorical_index

    @override
    def get_immutable_index(self) -> list[int]:
        """Return the indices of immutable features of the WiDS dataset.

        Returns:
            List of feature indices.

        """
        return immutable_index

    @override
    def get_bound_box(self) -> BoundingBox:
        """Return the region where adversarial points can exist for the WiDS dataset.

        Returns:
            Bounding box of the search region.

        """
        return self.bound_box

    @override
    def is_valid(self, x: Point) -> bool:
        """Check if x satisfies all constraints of the WiDS dataset.

        Args:
            x: Point to check.

        Returns:
            True if the constraint is satisfied, False otherwise.

        """
        return all(func(x) for func in self.relationship_consts)

    @override
    def set_smt_constraints(self, s: Solver, x: list[ArithRef]) -> None:
        """Add the relational constraints of the WiDS dataset to the SMT solver.

        Args:
            s: Z3 solver to add the constraints to.
            x: Z3 variables corresponding to the features.

        Returns:
            None.

        """
        for max_name, min_name in max_min_pair:
            max_index = wids_meta_dict[max_name].idx
            min_index = wids_meta_dict[min_name].idx
            s.add(x[min_index] <= x[max_index])

    @override
    def set_scip_constraints(self, s: Model, x: list[Variable]) -> None:
        """Add the relational constraints of the WiDS dataset to the SCIP model.

        Args:
            s: SCIP model to add the constraints to.
            x: SCIP variables corresponding to the features.

        Returns:
            None.

        """
        for max_name, min_name in max_min_pair:
            max_index = wids_meta_dict[max_name].idx
            min_index = wids_meta_dict[min_name].idx
            s.addCons(x[min_index] <= x[max_index])

    @override
    def get_moeva_constraints(self) -> list[BaseRelationConstraint]:
        """Return the relational constraints of the WiDS dataset for Moeva2.

        Returns:
            List of relation constraints.

        """
        relation_constraints = []
        for max_name, min_name in max_min_pair:
            max_idx = wids_meta_dict[max_name].idx
            min_idx = wids_meta_dict[min_name].idx
            relation_constraints.append(Feature(min_idx) <= Feature(max_idx))

        return relation_constraints
