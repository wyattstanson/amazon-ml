import numpy as np

from ber.blocking import Candidates
from ber.stage2 import competition_features


def test_competition_features_match_bruteforce():
    rng = np.random.default_rng(5)
    n = 400
    q = rng.integers(0, 40, n)
    r = rng.integers(0, 60, n)
    key = np.unique(q * 1000 + r, return_index=True)[1]          # unique (q, r) pairs
    q, r = q[key], r[key]
    p = rng.random(len(q))
    f = competition_features(Candidates(q, r, np.ones(len(q), np.float32), np.zeros(len(q), np.int16)), p)
    for i in range(len(q)):
        same_q = p[q == q[i]]
        others_r = p[(r == r[i]) & (q != q[i])]
        assert np.isclose(f["q_sum_p1"][i], same_q.sum(), atol=1e-5)
        assert f["q_n_above_half"][i] == (same_q > 0.5).sum()
        assert np.isclose(f["p1_frac_qmax"][i], p[i] / same_q.max(), atol=1e-5)
        assert f["p1_rank_q"][i] == (same_q > p[i]).sum()
        best_other = others_r.max() if len(others_r) else 0.0
        assert np.isclose(f["r_max_other"][i], best_other, atol=1e-6)
        assert np.isclose(f["r_sum_p1_other"][i], others_r.sum(), atol=1e-5)
        assert f["p1_rank_r"][i] == (others_r > p[i]).sum()
        lower = np.sort(same_q[same_q < p[i]])
        nxt = lower[-1] if len(lower) else 0.0
        assert np.isclose(f["p1_gap_q_next"][i], p[i] - nxt, atol=1e-6)


def test_mass_features_match_bruteforce():
    from ber.stage2 import mass_features
    rng = np.random.default_rng(11)
    n = 500
    q = rng.integers(0, 30, n)
    r = rng.integers(0, 80, n)
    key = np.unique(q * 1000 + r, return_index=True)[1]
    q, r = q[key], r[key]
    m = len(q)
    p = rng.random(m)
    r_code = rng.integers(0, 4, 80)            # per record: 0 = no code
    r_name = rng.integers(0, 5, 80)
    r_addr = rng.integers(0, 5, 80)
    r_src = rng.integers(2, 4, 80)
    r_empty = rng.random(80) < 0.2
    q_code = rng.integers(0, 4, 30)
    f = mass_features(q, r, p, r_code[r], q_code[q], r_name[r], r_addr[r], r_src[r], r_empty[r])
    for i in range(m):
        sq = q == q[i]
        others = sq & (np.arange(m) != i)
        code = r_code[r[i]]
        if code == 0:
            assert np.isnan(f["code_mass"][i]) and np.isnan(f["other_cluster_mass"][i])
        else:
            assert np.isclose(f["code_mass"][i], p[others & (r_code[r] == code)].sum(), atol=1e-5)
            masses = {c: p[sq & (r_code[r] == c)].sum() for c in set(r_code[r[sq]]) if c != 0}
            rival = max([v for c, v in masses.items() if c != code], default=0.0)
            assert np.isclose(f["other_cluster_mass"][i], rival, atol=1e-5)
            assert f["own_cluster_is_q_code"][i] == float(code == q_code[q[i]])
            xs = p[sq & (r_src[r] != r_src[r[i]]) & (r_code[r] == code)]
            assert np.isclose(f["xsrc_code_p"][i], xs.max() if len(xs) else 0.0, atol=1e-6)
        if r_name[r[i]] != 0:
            assert np.isclose(f["name_mass"][i], p[others & (r_name[r] == r_name[r[i]])].sum(), atol=1e-5)
        same_src = sq & (r_src[r] == r_src[r[i]])
        assert f["p1_rank_q_src"][i] == (p[same_src] > p[i]).sum()
        assert np.isclose(f["q_src_sum_p1"][i], p[same_src].sum(), atol=1e-5)
        assert np.isclose(f["q_sum_p1_empty"][i], p[sq & r_empty[r]].sum(), atol=1e-5)
        assert np.isclose(f["q_sum_p1_nonempty"][i], p[sq & ~r_empty[r]].sum(), atol=1e-5)
        assert f["r_n_claim_01"][i] == ((r == r[i]) & (p > 0.1)).sum()


def test_neighbour_pairs_pick_best_other_candidates():
    from ber.stage2 import neighbour_pairs
    rng = np.random.default_rng(3)
    q = np.sort(rng.integers(0, 25, 300))
    p = rng.random(300)
    empty = rng.random(300) < 0.1
    idx, j1, j2 = neighbour_pairs(q, p, empty)
    expect = np.flatnonzero(empty | ((p > 0.02) & (p < 0.98)))
    assert np.array_equal(idx, expect)
    for i, a, b in zip(idx, j1, j2):
        others = [k for k in np.flatnonzero(q == q[i]) if k != i]
        ranked = sorted(others, key=lambda k: (-p[k], k))
        assert a == (ranked[0] if ranked else -1)
        assert b == (ranked[1] if len(ranked) > 1 else -1)


def test_neighbour_matrix_values():
    from types import SimpleNamespace
    from rapidfuzz import fuzz
    from ber.stage2 import NEIGHBOUR_FEATURES, neighbour_matrix
    from ber.store import StrCol
    names = ["acme tools", "acme tool", "acme tools co", "zeta labs", "zeta lab"]
    addrs = ["1 main st", "1 main street", "", "9 elm rd", "9 elm road"]
    table = SimpleNamespace(name_core=StrCol.from_list(names), addr_norm=StrCol.from_list(addrs),
                            addr_empty=np.array([a == "" for a in addrs]))
    q = np.array([7, 7, 7, 8, 8])
    r = np.array([0, 1, 2, 3, 4])
    p = np.array([0.9, 0.5, 0.3, 0.99, 0.4])
    c = Candidates(q, r, np.ones(5, np.float32), np.zeros(5, np.int16))
    m = neighbour_matrix(table, c, p)
    f = dict(zip(NEIGHBOUR_FEATURES, m.T))
    assert np.isnan(f["nb1_name"][3])                         # p = 0.99: decided by p1 alone
    assert f["nb1_p"][1] == np.float32(0.9) and f["nb2_p"][1] == np.float32(0.3)
    assert np.isclose(f["nb1_name"][1], fuzz.token_set_ratio("acme tool", "acme tools"))
    assert np.isclose(f["nb1_addr"][1], fuzz.token_set_ratio("1 main street", "1 main st"))
    assert np.isnan(f["nb1_addr"][2])                         # empty address: no address similarity
    assert np.isnan(f["nb2_name"][4])                         # only one other candidate
    w = max(fuzz.token_set_ratio("acme tool", "acme tools") * 0.9, fuzz.token_set_ratio("acme tool", "acme tools co") * 0.3)
    assert np.isclose(f["nb_name_max_w"][1], w, rtol=1e-5)
