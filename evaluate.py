"""Local evaluation harness.

Implements the challenge metric exactly as specified:

    F_0.5 = (1.25 * P * R) / (0.25 * P + R)

computed per Source-1 entity and macro-averaged over *all* evaluated S1 entities.
Singletons (no true matches) score 1.0 for an empty prediction and 0.0 otherwise.
A non-singleton with an empty prediction, or with zero true positives, scores 0.0.

Also provides the blocking diagnostics used throughout development:
pair-level blocking recall, and the *oracle* macro F0.5 -- the score a perfect
matcher would get if it could only choose among the blocked candidates. The oracle
is the hard ceiling that blocking imposes on the final score.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

import numpy as np

from .io_utils import id_key

BETA2 = 0.25  # beta = 0.5  ->  beta^2 = 0.25


def f05_single(pred: Set[str], truth: Set[str]) -> float:
    """F0.5 for one Source-1 entity, with the challenge's singleton convention."""
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(truth)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def macro_f05(pred: Mapping[str, Set[str]], truth: Mapping[str, Set[str]],
              ids: Optional[Iterable[str]] = None) -> float:
    """Macro F0.5 over `ids` (default: every S1 id in `truth`).

    An S1 id missing from `pred` is treated as an empty prediction.
    """
    ids = list(truth.keys()) if ids is None else list(ids)
    if not ids:
        return float("nan")
    empty: Set[str] = set()
    return sum(f05_single(pred.get(i, empty), truth[i]) for i in ids) / len(ids)


def f05_breakdown(pred: Mapping[str, Set[str]], truth: Mapping[str, Set[str]],
                  ids: Optional[Iterable[str]] = None,
                  group: Optional[Mapping[str, str]] = None) -> Dict[str, float]:
    """Macro F0.5 plus diagnostic slices: singletons vs non-singletons and per group.

    Returns a flat dict so it prints nicely in the experiment log.
    """
    ids = list(truth.keys()) if ids is None else list(ids)
    empty: Set[str] = set()
    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    tp = n_pred = n_true = 0
    for i in ids:
        t = truth[i]
        p = pred.get(i, empty)
        s = f05_single(p, t)
        keys = ["all", "singleton" if not t else "non_singleton"]
        if group is not None:
            keys.append(f"group={group.get(i, '?')}")
        for k in keys:
            sums[k] += s
            counts[k] += 1
        tp += len(p & t)
        n_pred += len(p)
        n_true += len(t)
    out = {f"f05[{k}]": sums[k] / counts[k] for k in sums}
    out.update({f"n[{k}]": counts[k] for k in counts})
    out["micro_precision"] = tp / n_pred if n_pred else float("nan")
    out["micro_recall"] = tp / n_true if n_true else float("nan")
    out["pred_pairs"] = n_pred
    return out


def blocking_report(cands: Mapping[str, Set[str]], truth: Mapping[str, Set[str]],
                    ids: Optional[Iterable[str]] = None,
                    group: Optional[Mapping[str, str]] = None) -> Dict[str, float]:
    """Blocking-quality diagnostics for a candidate mapping S1 -> set(S2/S3 ids).

    pair_recall   : fraction of true (S1, match) pairs present in the candidates.
    entity_full   : fraction of non-singleton S1s whose *every* true match is a candidate.
    oracle_f05    : macro F0.5 of a perfect matcher restricted to the candidates
                    (predict truth & candidates). This is the blocking-imposed ceiling.
    pairs_per_s1  : mean candidate-list length.
    """
    ids = list(truth.keys()) if ids is None else list(ids)
    empty: Set[str] = set()
    found = total = full = nonsing = n_cand = 0
    oracle: Dict[str, float] = defaultdict(float)
    cnt: Dict[str, int] = defaultdict(int)
    for i in ids:
        t = truth[i]
        c = cands.get(i, empty)
        hit = t & c
        found += len(hit)
        total += len(t)
        n_cand += len(c)
        if t:
            nonsing += 1
            full += len(hit) == len(t)
        s = f05_single(hit, t)
        for k in ["all"] + ([f"group={group.get(i, '?')}"] if group is not None else []):
            oracle[k] += s
            cnt[k] += 1
    out = {
        "pair_recall": found / total if total else float("nan"),
        "entity_full_recall": full / nonsing if nonsing else float("nan"),
        "pairs_per_s1": n_cand / len(ids) if ids else float("nan"),
        "total_pairs": n_cand,
    }
    out.update({f"oracle_f05[{k}]": oracle[k] / cnt[k] for k in oracle})
    return out


def load_truth(path: str) -> Dict[str, Set[str]]:
    """Read train_ground_truth.tsv into {s1_id: set(matched ids)} (stdlib only)."""
    truth: Dict[str, Set[str]] = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if header != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError(f"unexpected ground-truth header {header}")
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            truth[s1] = {x for x in rest.split(",") if x}
    return truth


# --------------------------------------------------------------------------------------
# Vectorised evaluation over RecordTable row indices (used on the full 2.2M-entity set,
# where dict-of-sets over tens of millions of candidate ids would not fit in memory).
# --------------------------------------------------------------------------------------
def id_keys(entity_ids: List[str]) -> np.ndarray:
    return np.fromiter((id_key(e) for e in entity_ids), dtype=np.int64, count=len(entity_ids))


class IdIndex:
    """Maps entity-id strings to RecordTable rows via a sorted int64 key array."""

    def __init__(self, keys: np.ndarray):
        """`keys`: the table's per-row `id_key` column."""
        self.order = np.argsort(keys, kind="stable")
        self.sorted = keys[self.order]
        if len(np.unique(self.sorted)) != len(self.sorted):
            raise ValueError("entity ids are not unique after integer encoding")

    def rows(self, entity_ids: List[str]) -> np.ndarray:
        k = id_keys(entity_ids)
        pos = np.searchsorted(self.sorted, k)
        pos = np.minimum(pos, len(self.sorted) - 1)
        if not np.all(self.sorted[pos] == k):
            raise KeyError("some entity ids are not in the table")
        return self.order[pos]


def truth_pairs(path: str, index: IdIndex) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ground truth as arrays: (s1 rows [all S1 in file], match q rows, match r rows)."""
    s1_ids: List[str] = []
    q_ids: List[str] = []
    r_ids: List[str] = []
    with open(path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            s1_ids.append(s1)
            for x in rest.split(","):
                if x:
                    q_ids.append(s1)
                    r_ids.append(x)
    return index.rows(s1_ids), index.rows(q_ids), index.rows(r_ids)


def _pair_key(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    return q.astype(np.int64) * (1 << 32) + r.astype(np.int64)


def f05_from_counts(n_true: np.ndarray, n_pred: np.ndarray, tp: np.ndarray) -> np.ndarray:
    """Vectorised per-entity F0.5 with the challenge's singleton convention."""
    n_true = n_true.astype(np.float64)
    n_pred = n_pred.astype(np.float64)
    tp = tp.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n_pred > 0, tp / n_pred, 0.0)
        r = np.where(n_true > 0, tp / n_true, 0.0)
        f = np.where(tp > 0, (1 + BETA2) * p * r / (BETA2 * p + r), 0.0)
    return np.where(n_true == 0, (n_pred == 0).astype(np.float64), f)


def pair_eval(eval_q: np.ndarray, gt_q: np.ndarray, gt_r: np.ndarray, pred_q: np.ndarray,
              pred_r: np.ndarray, n_rows: int) -> Dict[str, np.ndarray]:
    """Per-entity counts for the S1 rows `eval_q`. Returns n_true, n_pred, tp, f05 arrays
    aligned with `eval_q`."""
    in_eval = np.zeros(n_rows, dtype=bool)
    in_eval[eval_q] = True
    gsel = in_eval[gt_q]
    psel = in_eval[pred_q]
    gq, gr, pq, pr = gt_q[gsel], gt_r[gsel], pred_q[psel], pred_r[psel]
    pk = np.unique(_pair_key(pq, pr))
    hit = np.isin(_pair_key(gq, gr), pk, assume_unique=False)
    n_true = np.bincount(gq, minlength=n_rows)[eval_q]
    uq = (pk >> 32).astype(np.int64)
    n_pred = np.bincount(uq, minlength=n_rows)[eval_q]
    tp = np.bincount(gq[hit], minlength=n_rows)[eval_q]
    return {"n_true": n_true, "n_pred": n_pred, "tp": tp, "f05": f05_from_counts(n_true, n_pred, tp)}


def blocking_report_arrays(eval_q, gt_q, gt_r, cand_q, cand_r, n_rows, group=None) -> Dict[str, float]:
    """Array version of `blocking_report`: pair recall, oracle F0.5 (per group), size."""
    c = pair_eval(eval_q, gt_q, gt_r, cand_q, cand_r, n_rows)
    oracle = f05_from_counts(c["n_true"], c["tp"], c["tp"])   # perfect matcher: predict found
    ns = c["n_true"] > 0
    out = {
        "pair_recall": c["tp"].sum() / max(1, c["n_true"].sum()),
        "entity_full_recall": float(np.mean(c["tp"][ns] == c["n_true"][ns])) if ns.any() else float("nan"),
        "pairs_per_s1": c["n_pred"].mean(),
        "total_pairs": int(c["n_pred"].sum()),
        "oracle_f05[all]": oracle.mean(),
    }
    if group is not None:
        g = group[eval_q]
        for label in np.unique(g):
            m = g == label
            out[f"oracle_f05[{label}]"] = oracle[m].mean()
            gt_m = c["n_true"][m]
            out[f"pair_recall[{label}]"] = c["tp"][m].sum() / max(1, gt_m.sum())
            out[f"pairs_per_s1[{label}]"] = c["n_pred"][m].mean()
    return out


def load_id_lists(path: str) -> Dict[str, Set[str]]:
    """Read a matching_results.tsv / candidate_pairs.tsv style file."""
    out: Dict[str, Set[str]] = {}
    with open(path, encoding="utf-8") as f:
        f.readline()
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            out[s1] = {x for x in rest.split(",") if x}
    return out
