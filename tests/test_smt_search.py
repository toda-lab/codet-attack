"""Tests for smt_search.py."""

import numpy as np
from z3 import ArithRef, Int, Real, Solver, sat

from smt_attack.box import BoundingBox
from smt_attack.smt_search import attack_smt, model_to_point
from smt_attack.tree import TreeEnsemble, parse_xgboost_json
from tests.dummy_constraints import DummyConstraints


def _load_model() -> TreeEnsemble:
    """Load the sample model.

    Sample tree: f0 <= 3 -> (f1 <= 4 -> -1, else -3), f0 > 3 -> 4 (the leaf with value 2 is unreachable).

    Returns:
        Sample tree ensemble with two features.

    """
    return parse_xgboost_json("tests/sample_xgboost.json", 2)


def test_model_to_point_real_and_int() -> None:
    """Test conversion of a Z3 model with Int and Real variables.

    Returns:
        None.

    """
    x: list[ArithRef] = [Int("x_0"), Real("x_1")]
    s = Solver()
    s.add(x[0] == 3, x[1] == 1.5)
    assert s.check() == sat

    p = model_to_point(x, s.model())
    assert bool(np.all(p == np.array([3.0, 1.5])))


def test_attack_flips_class_0_to_1() -> None:
    """Test that the adversarial point of a class 0 victim is classified as 1.

    Returns:
        None.

    """
    model = _load_model()
    victim = np.array([0.0, 0.0])
    assert model.predict_one(victim) == 0

    c = DummyConstraints(victim, BoundingBox.global_box(2))
    adv = attack_smt(victim, model, c)

    assert adv is not None
    assert model.predict_one(adv) == 1
    assert c.is_valid(adv)


def test_attack_flips_class_1_to_0() -> None:
    """Test that the adversarial point of a class 1 victim is classified as 0.

    Returns:
        None.

    """
    model = _load_model()
    victim = np.array([10.0, 0.0])
    assert model.predict_one(victim) == 1

    c = DummyConstraints(victim, BoundingBox.global_box(2))
    adv = attack_smt(victim, model, c)

    assert adv is not None
    assert model.predict_one(adv) == 0


def test_attack_respects_bound_box() -> None:
    """Test that the adversarial point stays inside the bound box.

    Returns:
        None.

    """
    model = _load_model()
    victim = np.array([0.0, 0.0])
    box = BoundingBox(np.array([-10.0, -5.0]), np.array([20.0, 5.0]))

    c = DummyConstraints(victim, box)
    adv = attack_smt(victim, model, c)

    assert adv is not None
    assert model.predict_one(adv) == 1
    assert bool(np.all(adv >= box.lower))
    assert bool(np.all(adv <= box.upper))


def test_attack_keeps_immutable_feature() -> None:
    """Test that a feature with equal lower and upper bounds stays fixed.

    Returns:
        None.

    """
    model = _load_model()
    victim = np.array([0.0, 2.0])
    box = BoundingBox(np.array([-np.inf, 2.0]), np.array([np.inf, 2.0]))

    c = DummyConstraints(victim, box, immutable=[1])
    adv = attack_smt(victim, model, c)

    assert adv is not None
    assert adv[1] == 2.0
    assert model.predict_one(adv) == 1


def test_attack_returns_none_when_infeasible() -> None:
    """Test that None is returned when f0, the only useful feature, is immutable.

    Returns:
        None.

    """
    model = _load_model()
    victim = np.array([0.0, 0.0])
    box = BoundingBox(np.array([0.0, -np.inf]), np.array([0.0, np.inf]))

    c = DummyConstraints(victim, box, immutable=[0])
    assert attack_smt(victim, model, c) is None


def test_attack_categorical_feature_is_integer() -> None:
    """Test that a categorical feature takes an integer value.

    Returns:
        None.

    """
    model = _load_model()
    victim = np.array([0.0, 0.0])

    c = DummyConstraints(victim, BoundingBox.global_box(2), categorical=[0])
    adv = attack_smt(victim, model, c)

    assert adv is not None
    assert adv[0] == int(adv[0])
    assert model.predict_one(adv) == 1


def test_attack_applies_extra_smt_constraints() -> None:
    """Test that constraints added by set_smt_constraints are respected.

    Returns:
        None.

    """
    model = _load_model()
    victim = np.array([0.0, 0.0])

    c = DummyConstraints(
        victim,
        BoundingBox.global_box(2),
        smt_hook=lambda s, x: s.add(x[0] >= 100),
    )
    adv = attack_smt(victim, model, c)

    assert adv is not None
    assert adv[0] >= 100
    assert model.predict_one(adv) == 1


def test_attack_returns_none_with_contradicting_smt_constraints() -> None:
    """Test that None is returned when extra constraints rule out every adversarial point.

    Returns:
        None.

    """
    model = _load_model()
    victim = np.array([0.0, 0.0])

    c = DummyConstraints(
        victim,
        BoundingBox.global_box(2),
        smt_hook=lambda s, x: s.add(x[0] <= 3),
    )
    assert attack_smt(victim, model, c) is None
