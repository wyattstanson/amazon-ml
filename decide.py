"""Stage 5: turn pair probabilities into per-S1 match lists.

Three composable decisions, each measured on out-of-fold predictions (EXPERIMENTS.md):

1. Exclusivity -- in the ground truth every S2/S3 record belongs to at most one S1, so
   a record is only ever assigned to the S1 that gives it the highest probability.
2. Global threshold -- keep pairs with p >= t, t tuned for macro F0.5.
3. Expected-F0.5 set selection -- per S1, sort candidates by p and pick the prefix size
   k maximising the plug-in expected F0.5,
       E[F](k) ~= (1 + b2) * sum_{i<=k} p_i / (b2 * sum_all p_i + k),
   against k = 0 whose expected score is P(no true match) = prod_i (1 - p_i).
   This makes "predict nothing" a first-class, scored option (the abstain signal)
   instead of the leftover of a threshold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BETA2 = 0.25


@dataclass
class DecisionParams:
    exclusive: bool = True
    method: str = "threshold"          # "threshold" or "expected_f"
    threshold: float = 0.5             # used by "threshold"; floor for "expected_f"
    singleton_boost: float = 1.0       # multiplies P(no match) in "expected_f" (tuned)


def exclusive_mask(q: np.ndarray, r: np.ndarray, p: np.ndarray) -> np.ndarray:
    """True for the single best S1 of every S2/S3 record (ties -> lowest S1 row)."""
    order = np.lexsort((q, -p, r))
    rs = r[order]
    first = np.ones(len(rs), dtype=bool)
    first[1:] = rs[1:] != rs[:-1]
    mask = np.zeros(len(q), dtype=bool)
    mask[order[first]] = True
    return mask


def _expected_f_select(q: np.ndarray, p: np.ndarray, floor: float, singleton_boost: float) -> np.ndarray:
    """Vectorised expected-F0.5 prefix selection per S1. Returns a keep-mask."""
    order = np.lexsort((-p, q))
    qs, ps = q[order], p[order].astype(np.float64)
    n = len(qs)
    if n == 0:
        return np.zeros(0, dtype=bool)
    start = np.ones(n, dtype=bool)
    start[1:] = qs[1:] != qs[:-1]
    grp = np.cumsum(start) - 1
    first_idx = np.flatnonzero(start)
    k = np.arange(n) - first_idx[grp] + 1                     # prefix size at this position
    csum = np.cumsum(ps)
    base = np.concatenate([[0.0], csum])[first_idx][grp]
    prefix = csum - base                                      # sum of top-k p
    total = np.bincount(grp, weights=ps)[grp]
    ef = (1 + BETA2) * prefix / (BETA2 * total + k)
    # P(no match) = prod(1 - p) per group, via log-sum
    log1m = np.log(np.clip(1.0 - ps, 1e-12, 1.0))
    p_none = np.exp(np.bincount(grp, weights=log1m))[grp] * singleton_boost
    ef = np.where(ps >= floor, ef, -1.0)                      # never take items below the floor
    best_ef = np.full(grp.max() + 1, -np.inf)
    np.maximum.at(best_ef, grp, ef)
    # best k per group = first position where ef attains the group max
    is_best = (ef == best_ef[grp]) & (ef > p_none)
    best_k = np.full(grp.max() + 1, 0, dtype=np.int64)
    pos_k = np.where(is_best, k, np.iinfo(np.int64).max)
    kmin = np.full(grp.max() + 1, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(kmin, grp, pos_k)
    best_k = np.where(kmin == np.iinfo(np.int64).max, 0, kmin)
    keep_sorted = k <= best_k[grp]
    keep = np.zeros(n, dtype=bool)
    keep[order] = keep_sorted
    return keep


def decide(q: np.ndarray, r: np.ndarray, p: np.ndarray, params: DecisionParams,
           excl: np.ndarray = None) -> np.ndarray:
    """Boolean mask over candidate pairs: which pairs become predicted matches.
    `excl` optionally passes a precomputed `exclusive_mask(q, r, p)` (tuning loops)."""
    mask = np.ones(len(q), dtype=bool)
    if params.exclusive:
        mask &= exclusive_mask(q, r, p) if excl is None else excl
    if params.method == "threshold":
        return mask & (p >= params.threshold)
    if params.method == "expected_f":
        keep = np.zeros(len(q), dtype=bool)
        idx = np.flatnonzero(mask)
        keep[idx] = _expected_f_select(q[idx], p[idx], params.threshold, params.singleton_boost)
        return keep
    raise ValueError(f"unknown decision method {params.method!r}")
