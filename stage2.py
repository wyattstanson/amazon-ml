"""Stage-2 "competition" features computed from stage-1 probabilities.

A pair's stage-1 probability says how similar two records look in isolation. Whether
they are a match also depends on the alternatives: is there a better candidate for this
S1, and does another S1 explain this S2/S3 record better (in the ground truth every
S2/S3 record belongs to at most one S1)? These features expose exactly that context to
a second model, which learns the trade-off instead of relying on the hard exclusivity
rule alone.

Two further blocks read the same probabilities (EXPERIMENTS.md section 7):
* `mass_features` -- the support counts of features.py weighted by probability, so one
  strong cluster is told apart from two competing ones (a planted sibling entity);
* `neighbour_matrix` -- how similar a candidate is to the S1's other most probable
  candidates (true matches resemble each other, a namesake does not).

All computations are vectorised over the whole candidate set (sorting + segment ops).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .blocking import Candidates

STAGE2_FEATURES: List[str] = [
    "p1", "p1_rank_q", "p1_frac_qmax", "q_sum_p1", "q_n_above_half", "p1_gap_q_next",
    "r_max_other", "p1_gap_r_other", "p1_rank_r", "r_sum_p1_other",
]


def _sorted_groups(major: np.ndarray, p: np.ndarray, tie: np.ndarray):
    """Sort by (major asc, p desc, tie asc). Returns (order, position-in-group, index of the
    group's first element) -- in sorted order.

    (major, descending p) is packed into one int64 key -- the IEEE bit pattern of a
    non-negative float32 is monotone -- and sorted stably; the input arrives ordered by
    `tie` within equal keys, so stability reproduces the tie-break of a 3-key lexsort at a
    fraction of its cost."""
    inv = (1.0 - np.clip(p, 0.0, 1.0)).astype(np.float32).view(np.uint32).astype(np.int64)
    key = (major.astype(np.int64) << 32) | inv
    pre = np.argsort(tie, kind="stable")                       # establish tie order first
    order = pre[np.argsort(key[pre], kind="stable")]
    m = major[order]
    n = len(m)
    first = np.ones(n, dtype=bool)
    first[1:] = m[1:] != m[:-1]
    start = np.maximum.accumulate(np.where(first, np.arange(n), 0))
    return order, np.arange(n) - start, start


def competition_features(c: Candidates, p1: np.ndarray) -> Dict[str, np.ndarray]:
    """Two sorts over the whole candidate graph (one grouped by S1, one by S2/S3 record)."""
    p = p1.astype(np.float64)
    n = len(p)
    nq = int(c.q.max()) + 1
    nr = int(c.r.max()) + 1
    out: Dict[str, np.ndarray] = {"p1": p.astype(np.float32)}
    # ---- S1 side: rank, best, next-best -------------------------------------------------
    order, pos, start = _sorted_groups(c.q, p, c.r)
    ps = p[order]
    same_next = np.zeros(n, dtype=bool)
    same_next[:-1] = start[1:] == start[:-1]
    nxt = np.where(same_next, np.r_[ps[1:], 0.0], 0.0)
    q_best = ps[start]
    tmp = np.empty(n, dtype=np.float32)
    tmp[order] = pos
    out["p1_rank_q"] = tmp.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        tmp[order] = np.where(q_best > 0, ps / q_best, 0.0)
    out["p1_frac_qmax"] = tmp.copy()
    tmp[order] = ps - nxt
    out["p1_gap_q_next"] = tmp.copy()
    out["q_sum_p1"] = np.bincount(c.q, weights=p, minlength=nq)[c.q].astype(np.float32)
    out["q_n_above_half"] = np.bincount(c.q, weights=(p > 0.5).astype(np.float64), minlength=nq)[c.q].astype(np.float32)
    del order, pos, start, ps, nxt, q_best
    # ---- S2/S3 side: the best *competing* S1 for the same record ----------------------
    order, pos, start = _sorted_groups(c.r, p, c.q)
    pr = p[order]
    best = pr[start]
    has_second = np.zeros(n, dtype=bool)
    idx2 = start + 1
    ok = idx2 < n
    has_second[ok] = start[np.minimum(idx2, n - 1)][ok] == start[ok]
    second = np.where(has_second, pr[np.minimum(idx2, n - 1)], 0.0)
    other = np.where(pos == 0, second, best)          # the top claimant's rival is the runner-up
    tmp[order] = other
    out["r_max_other"] = tmp.copy()
    out["p1_gap_r_other"] = (p - out["r_max_other"]).astype(np.float32)
    tmp[order] = pos
    out["p1_rank_r"] = tmp.copy()
    out["r_sum_p1_other"] = (np.bincount(c.r, weights=p, minlength=nr)[c.r] - p).astype(np.float32)
    return out


def matrix(c: Candidates, p1: np.ndarray) -> np.ndarray:
    f = competition_features(c, p1)
    return np.column_stack([f[k] for k in STAGE2_FEATURES]).astype(np.float32)


# --------------------------------------------------------------------------------------
# Probability-weighted cluster support (EXPERIMENTS.md section 7)
# --------------------------------------------------------------------------------------
MASS_FEATURES: List[str] = [
    "code_mass", "name_mass", "addr_mass", "other_cluster_mass", "own_cluster_is_q_code", "xsrc_code_p",
    "p1_rank_q_src", "q_src_sum_p1", "q_sum_p1_empty", "q_sum_p1_nonempty", "r_n_claim_01",
]


def _group_sum(key: np.ndarray, w: np.ndarray):
    _, inv = np.unique(key, return_inverse=True)
    return np.bincount(inv, weights=w)[inv], inv


def mass_features(q: np.ndarray, r: np.ndarray, p: np.ndarray, rcode: np.ndarray, qcode: np.ndarray,
                  rname: np.ndarray, raddr: np.ndarray, src: np.ndarray, r_empty: np.ndarray
                  ) -> Dict[str, np.ndarray]:
    """The support features of features.py weighted by stage-1 probability.

    An S1's likely true matches form one cluster: they share a house code, name or address
    with each other and appear in both sources. A planted sibling forms a second cluster.
    Counting supporters (support_features) treats a supporter at p = 0.01 like one at
    p = 0.99; weighting by p separates "one strong cluster" from "two competing ones".
    Hash arguments are per pair (0 = missing); `src` is the candidate's source (2 or 3).
    """
    q = q.astype(np.int64)
    r = r.astype(np.int64)
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    nq = int(q.max()) + 1 if n else 1
    nr = int(r.max()) + 1 if n else 1
    q32 = q << 32
    out: Dict[str, np.ndarray] = {}

    def keep32(x):
        return x.astype(np.float32)

    has_code = rcode != 0
    code_sum, code_grp = _group_sum(q32 | (rcode & 0xFFFFFFFF), p)
    out["code_mass"] = keep32(np.where(has_code, code_sum - p, np.nan))
    s, _ = _group_sum(q32 | (rname & 0xFFFFFFFF), p)
    out["name_mass"] = keep32(np.where(rname != 0, s - p, np.nan))
    s, _ = _group_sum(q32 | (raddr & 0xFFFFFFFF), p)
    out["addr_mass"] = keep32(np.where(raddr != 0, s - p, np.nan))

    # mass of the strongest code cluster of the same S1 other than this pair's own cluster
    n_grp = int(code_grp.max()) + 1 if n else 0
    grp_mass = np.bincount(code_grp, weights=p, minlength=n_grp)
    grp_q = np.zeros(n_grp, dtype=np.int64)
    grp_q[code_grp] = q
    grp_valid = np.zeros(n_grp, dtype=bool)
    grp_valid[code_grp] = has_code
    m = np.where(grp_valid, grp_mass, -1.0)
    order = np.lexsort((np.arange(n_grp), -m, grp_q))           # per S1: heaviest cluster first
    gq_sorted = grp_q[order]
    first = np.ones(n_grp, dtype=bool)
    first[1:] = gq_sorted[1:] != gq_sorted[:-1]
    pos = np.arange(n_grp) - np.flatnonzero(first)[np.cumsum(first) - 1]
    top1_id = np.full(nq, -1, dtype=np.int64)
    top1_m = np.full(nq, -1.0)
    top2_m = np.full(nq, -1.0)
    top1_id[gq_sorted[pos == 0]] = order[pos == 0]
    top1_m[gq_sorted[pos == 0]] = m[order][pos == 0]
    top2_m[gq_sorted[pos == 1]] = m[order][pos == 1]
    other = np.where(code_grp == top1_id[q], top2_m[q], top1_m[q])
    out["other_cluster_mass"] = keep32(np.where(has_code, np.maximum(other, 0.0), np.nan))
    out["own_cluster_is_q_code"] = keep32(np.where(has_code, (rcode == qcode).astype(np.float64), np.nan))

    # cross-source agreement: best p among the S1's other-source candidates with this code
    code_key = q32 | (rcode & 0xFFFFFFFF)
    xs = np.zeros(n)
    for s_here, s_other in ((2, 3), (3, 2)):
        other_sel = src == s_other
        keys, inv = np.unique(code_key[other_sel], return_inverse=True)
        best = np.zeros(len(keys))
        np.maximum.at(best, inv, p[other_sel])
        here = np.flatnonzero(src == s_here)
        if len(keys) and len(here):
            j = np.minimum(np.searchsorted(keys, code_key[here]), len(keys) - 1)
            xs[here] = np.where(keys[j] == code_key[here], best[j], 0.0)
    out["xsrc_code_p"] = keep32(np.where(has_code, xs, np.nan))

    # rank and mass inside the S1's candidate list, per source
    qs_key = q * 4 + src.astype(np.int64)
    order = np.lexsort((-p, qs_key))
    ks = qs_key[order]
    first = np.ones(n, dtype=bool)
    first[1:] = ks[1:] != ks[:-1]
    rank = np.empty(n)
    rank[order] = np.arange(n) - np.flatnonzero(first)[np.cumsum(first) - 1]
    out["p1_rank_q_src"] = keep32(rank)
    s, _ = _group_sum(qs_key, p)
    out["q_src_sum_p1"] = keep32(s)
    empty = r_empty.astype(bool)
    out["q_sum_p1_empty"] = keep32(np.bincount(q, weights=p * empty, minlength=nq)[q])
    out["q_sum_p1_nonempty"] = keep32(np.bincount(q, weights=p * ~empty, minlength=nq)[q])
    # how many S1s claim this record with some confidence
    out["r_n_claim_01"] = keep32(np.bincount(r, weights=(p > 0.1).astype(np.float64), minlength=nr)[r])
    return out


NEIGHBOUR_FEATURES: List[str] = ["nb1_name", "nb1_p", "nb2_name", "nb2_p", "nb1_addr", "nb_name_max_w"]


def neighbour_pairs(q: np.ndarray, p: np.ndarray, r_empty: np.ndarray, lo: float = 0.02, hi: float = 0.98):
    """For every uncertain pair i (empty-address record, or lo < p < hi): the indices of the
    two most probable *other* candidates of the same S1 (-1 if none). Returns (idx, j1, j2)."""
    q = q.astype(np.int64)
    n = len(p)
    order = np.lexsort((np.arange(n), -p, q))
    qs = q[order]
    first = np.ones(n, dtype=bool)
    first[1:] = qs[1:] != qs[:-1]
    pos = np.arange(n) - np.flatnonzero(first)[np.cumsum(first) - 1]
    nq = int(q.max()) + 1 if n else 1
    top = []
    for k in range(3):
        t = np.full(nq, -1, dtype=np.int64)
        sel = pos == k
        t[qs[sel]] = order[sel]
        top.append(t)
    idx = np.flatnonzero(r_empty.astype(bool) | ((p > lo) & (p < hi)))
    t0, t1, t2 = (t[q[idx]] for t in top)
    j1 = np.where(t0 == idx, t1, t0)
    j2 = np.where((t0 == idx) | (t1 == idx), t2, t1)
    return idx, j1, j2


def neighbour_matrix(table, c: Candidates, p1: np.ndarray) -> np.ndarray:
    """How similar is each candidate to the S1's *other* most probable candidates?

    True matches of one S1 are variants of one canonical record and resemble each other; a
    sibling or namesake resembles the S1 but not the rest of its cluster. Name similarity
    (token-set ratio) to the two most probable other candidates, their probabilities, address
    similarity to the first, and the probability-weighted best name similarity. Computed for
    uncertain pairs only; NaN elsewhere (those pairs are decided by p1 alone)."""
    from rapidfuzz import fuzz
    from rapidfuzz.process import cpdist
    p = np.asarray(p1, dtype=np.float64)
    n = len(p)
    r_empty = np.asarray(table.addr_empty).astype(bool)[c.r]
    idx, j1, j2 = neighbour_pairs(c.q, p, r_empty)
    out = np.full((n, len(NEIGHBOUR_FEATURES)), np.nan, dtype=np.float32)

    def strings(col, rows):
        u, inv = np.unique(rows, return_inverse=True)
        vals = col.take(u)
        return [vals[i] for i in inv]

    r_idx = c.r[idx]
    names_i = strings(table.name_core, r_idx)
    weighted = []
    for slot, j in ((0, j1), (2, j2)):
        ok = j >= 0
        jj = np.where(ok, j, idx)
        sim = cpdist(names_i, strings(table.name_core, c.r[jj]), scorer=fuzz.token_set_ratio, workers=-1,
                     dtype=np.float32)
        out[idx, slot] = np.where(ok, sim, np.nan)
        out[idx, slot + 1] = np.where(ok, p[jj], np.nan)
        weighted.append(np.where(ok, sim * p[jj], 0.0))
        if slot == 0:
            asim = cpdist(strings(table.addr_norm, r_idx), strings(table.addr_norm, c.r[jj]),
                          scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
            both = ok & ~r_empty[idx] & ~r_empty[jj]
            out[idx, 4] = np.where(both, asim, np.nan)
    out[idx, 5] = np.maximum(weighted[0], weighted[1])
    return out


def mass_matrix(table, c: Candidates, p1: np.ndarray) -> np.ndarray:
    """`mass_features` for a whole candidate set, reading the hashes from the record table."""
    from .features import _hashes
    ur, rinv = np.unique(c.r, return_inverse=True)
    uq, qinv = np.unique(c.q, return_inverse=True)
    f = mass_features(
        c.q, c.r, p1,
        rcode=_hashes(table.codes.take(ur), True)[rinv],
        qcode=_hashes(table.codes.take(uq), True)[qinv],
        rname=_hashes(table.name_core.take(ur), False)[rinv],
        raddr=_hashes(table.addr_norm.take(ur), False)[rinv],
        src=np.asarray(table.source)[c.r],
        r_empty=np.asarray(table.addr_empty)[c.r],
    )
    return np.column_stack([f[k] for k in MASS_FEATURES]).astype(np.float32)
