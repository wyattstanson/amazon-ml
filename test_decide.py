import numpy as np

from ber.decide import BETA2, DecisionParams, decide, exclusive_mask


def test_exclusive_mask_keeps_best_owner_per_record():
    q = np.array([0, 1, 1, 2])
    r = np.array([10, 10, 11, 11])
    p = np.array([0.9, 0.8, 0.3, 0.7])
    assert exclusive_mask(q, r, p).tolist() == [True, False, False, True]


def _brute_expected_f(ps, floor, boost):
    """Reference: try every prefix of the sorted candidates."""
    ps = sorted(ps, reverse=True)
    total = sum(ps)
    best_k, best = 0, np.prod([1 - x for x in ps]) * boost
    for k in range(1, len(ps) + 1):
        if ps[k - 1] < floor:
            break
        ef = (1 + BETA2) * sum(ps[:k]) / (BETA2 * total + k)
        if ef > best + 1e-15:
            best_k, best = k, ef
    return best_k


def test_expected_f_matches_brute_force():
    rng = np.random.default_rng(1)
    q, p = [], []
    for g in range(400):
        n = int(rng.integers(1, 8))
        q += [g] * n
        p += list(rng.beta(0.6, 0.6, size=n))
    q, p = np.array(q), np.array(p)
    r = np.arange(len(q))                       # all records distinct -> exclusivity is a no-op
    for floor, boost in [(0.0, 1.0), (0.2, 1.0), (0.1, 1.5)]:
        keep = decide(q, r, p, DecisionParams(exclusive=True, method="expected_f", threshold=floor,
                                              singleton_boost=boost))
        for g in range(400):
            m = q == g
            assert keep[m].sum() == _brute_expected_f(p[m].tolist(), floor, boost)
            if keep[m].any():                   # kept items are the top-k by p
                assert p[m][keep[m]].min() >= p[m][~keep[m]].max(initial=-1)


def test_threshold_method():
    q = np.array([0, 0, 1])
    r = np.array([5, 6, 7])
    p = np.array([0.9, 0.4, 0.6])
    keep = decide(q, r, p, DecisionParams(exclusive=False, method="threshold", threshold=0.5))
    assert keep.tolist() == [True, False, True]
