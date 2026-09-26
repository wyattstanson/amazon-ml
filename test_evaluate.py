import math

from ber.evaluate import blocking_report, f05_single, macro_f05


def test_readme_example():
    # From the challenge README: predict 3, 2 of them correct, truth has 2.
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    truth = {"S2-00047", "S3-00812"}
    assert math.isclose(f05_single(pred, truth), 0.7142857, rel_tol=1e-6)


def test_singleton_rules():
    assert f05_single(set(), set()) == 1.0
    assert f05_single({"S2-1"}, set()) == 0.0


def test_empty_prediction_on_nonsingleton_is_zero():
    assert f05_single(set(), {"S2-1"}) == 0.0


def test_no_overlap_is_zero():
    assert f05_single({"S2-2"}, {"S2-1"}) == 0.0


def test_precision_weighted_more_than_recall():
    truth = {"a", "b", "c", "d"}
    half_recall = f05_single({"a", "b"}, truth)           # P=1, R=0.5
    half_precision = f05_single({"a", "b", "c", "d", "x", "y", "z", "w"}, truth)  # P=0.5, R=1
    assert half_recall > half_precision
    assert math.isclose(half_recall, 1.25 * 0.5 / (0.25 + 0.5))


def test_macro_average_and_missing_ids():
    truth = {"S1-a": {"S2-1"}, "S1-b": set(), "S1-c": {"S3-9"}}
    pred = {"S1-a": {"S2-1"}}  # S1-b missing -> empty -> 1.0 ; S1-c missing -> 0.0
    assert math.isclose(macro_f05(pred, truth), (1.0 + 1.0 + 0.0) / 3)


def test_blocking_oracle():
    truth = {"S1-a": {"S2-1", "S2-2"}, "S1-b": set()}
    cands = {"S1-a": {"S2-1", "S2-9"}, "S1-b": {"S2-5"}}
    rep = blocking_report(cands, truth)
    assert rep["pair_recall"] == 0.5
    # Oracle predicts {S2-1} for a (P=1, R=.5) and nothing for b (singleton -> 1.0).
    assert math.isclose(rep["oracle_f05[all]"], ((1.25 * 0.5 / 0.75) + 1.0) / 2)
    assert rep["pairs_per_s1"] == 1.5


def test_vectorised_metric_matches_reference():
    import numpy as np
    from ber.evaluate import f05_from_counts
    rng = np.random.default_rng(0)
    for _ in range(2000):
        n_true = int(rng.integers(0, 6))
        truth = {f"t{i}" for i in range(n_true)}
        pred = {x for x in truth if rng.random() < 0.6} | {f"f{i}" for i in range(int(rng.integers(0, 4)))}
        tp = len(pred & truth)
        vec = f05_from_counts(np.array([n_true]), np.array([len(pred)]), np.array([tp]))[0]
        assert math.isclose(vec, f05_single(pred, truth), abs_tol=1e-12)
