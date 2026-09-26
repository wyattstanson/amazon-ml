"""Stage 2: candidate generation (blocking).

Both strategies run per country (0 of 7.6M training matches cross countries) and per
target source, on one sparse-matrix engine:

  score(s1, r) = sum_{shared tokens t} w(t)          computed as  Q @ R^T

* Strategy A -- standard key blocking: a handful of exact composite keys
  (house-number+street, first-two-name-words, compact name). Every pair sharing a
  key is a candidate; keys whose block holds more than `max_df` records are skipped.
* Strategy B -- IDF-weighted token meta-blocking: typed tokens (name words, phonetic
  keys, prefixes, address words/codes, street bigrams, and cross-field conjunctions)
  weighted by IDF over Sources 2+3; tokens more frequent than their type's cap are
  dropped as non-discriminating ("block purging"); the top-`k` records per S1 per
  source are kept. Records with an empty address can only ever score on their name,
  so they are ranked in a separate channel against each other (top-`k_empty`) rather
  than against records that also share address evidence.

The output of the chosen strategy is the exact candidate set the matcher scores.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from .store import RecordTable
from .tokens import TokenCSR, hid

log = logging.getLogger(__name__)


@dataclass
class BlockingParams:
    strategy: str = "B"                         # "A" (key blocking) or "B" (meta-blocking)
    token_types: Sequence[str] = ("name", "phon", "pref", "addrw", "addrc", "street", "pa", "ca", "nb", "pb")
    max_df: int = 2000                          # default cap: tokens in more records are dropped
    max_df_by_type: Dict[str, int] = field(default_factory=dict)   # per-type overrides
    k: int = 12                                 # top-k per S1 per target source (strategy B)
    k_empty: int = 4                            # separate top-k among empty-address records
    flops_per_chunk: float = 4e7                # SpGEMM work budget per query chunk


@dataclass
class Candidates:
    """Candidate pairs as parallel arrays of RecordTable row indices."""
    q: np.ndarray        # S1 row
    r: np.ndarray        # S2/S3 row
    score: np.ndarray    # blocking score
    rank: np.ndarray     # rank of r among the S1's candidates from the same source (0 = best)

    def __len__(self):
        return len(self.q)


# --------------------------------------------------------------------------------------
# Strategy A keys
# --------------------------------------------------------------------------------------
def key_tokens(table: RecordTable, rows: np.ndarray, tokens: Dict[str, TokenCSR]) -> TokenCSR:
    """Exact composite keys per record: street bigrams + two name keys."""
    cores = table.name_core.take(rows)
    street = tokens["street"].rows(rows)
    ids: List[int] = []
    indptr = [0]
    for i, core in enumerate(cores):
        words = core.split()
        rec = list(street.ids[street.indptr[i]:street.indptr[i + 1]].tolist())
        if words:
            rec.append(hid("k2:" + " ".join(sorted(words[:2]))))
            rec.append(hid("kc:" + "".join(words)))
        ids.extend(rec)
        indptr.append(len(ids))
    return TokenCSR(np.array(ids, dtype=np.uint32), np.array(indptr, dtype=np.int64))


# --------------------------------------------------------------------------------------
# Sparse engine
# --------------------------------------------------------------------------------------
def _combine(tokens: Dict[str, TokenCSR], rows: np.ndarray,
             types: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate the chosen token types for `rows` -> (ids, type index per id, indptr)."""
    parts = [tokens[t].rows(rows) for t in types]
    lens = sum(np.diff(p.indptr) for p in parts)
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(lens, out=indptr[1:])
    ids = np.empty(indptr[-1], dtype=np.uint32)
    tix = np.empty(indptr[-1], dtype=np.int8)
    fill = indptr[:-1].copy()
    for j, p in enumerate(parts):
        plen = np.diff(p.indptr)
        dst = np.repeat(fill, plen) + (np.arange(len(p.ids)) - np.repeat(p.indptr[:-1], plen))
        ids[dst] = p.ids
        tix[dst] = j
        fill += plen
    return ids, tix, indptr


def sorted_lookup(vocab: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Position of each id in the sorted `vocab`, -1 if absent.

    Keys are sorted before `searchsorted`: binary searches for random-order keys over a
    multi-million vocabulary are cache-miss bound (~1.3 us each, 150 s per country in
    profiling); sorted keys make the same lookups ~50x faster.
    """
    out = np.full(len(ids), -1, dtype=np.int64)
    if len(vocab) == 0 or len(ids) == 0:
        return out
    order = np.argsort(ids, kind="stable")
    s = ids[order]
    pos = np.minimum(np.searchsorted(vocab, s), len(vocab) - 1)
    hit = vocab[pos] == s
    out[order[hit]] = pos[hit]
    return out


def _to_matrix(cols: np.ndarray, vals: np.ndarray, indptr: np.ndarray, n_cols: int) -> sp.csr_matrix:
    """CSR from per-row column ids (-1 = dropped), built directly from the row pointer --
    rows are already grouped, so no COO sort is needed."""
    keep = cols >= 0
    cum = np.concatenate([[0], np.cumsum(keep, dtype=np.int64)])
    new_ptr = cum[np.asarray(indptr)]
    return sp.csr_matrix((vals[keep].astype(np.float32), cols[keep].astype(np.int32), new_ptr),
                         shape=(len(indptr) - 1, n_cols))


def _topk_rows(prod: sp.csr_matrix, k: int, all_pairs: bool):
    """Top-k entries of every row of a score matrix -> (row, col, score, rank).

    Rows with more than k entries use a per-row partial selection (O(n)) instead of a
    global sort of every non-zero; ties at the k-th score keep the lowest column index,
    so the result is deterministic.
    """
    indptr, data, cols = prod.indptr, prod.data, prod.indices
    counts = np.diff(indptr)
    keep = np.ones(len(data), dtype=bool)
    if not all_pairs:
        for i in np.flatnonzero(counts > k).tolist():
            s, e = indptr[i], indptr[i + 1]
            seg = data[s:e]
            kth = np.partition(seg, len(seg) - k)[len(seg) - k]      # k-th largest score
            above = seg > kth
            need = k - int(above.sum())
            tie = np.flatnonzero(seg == kth)
            tie = tie[np.argsort(cols[s:e][tie], kind="stable")][:need]
            m = above
            m[tie] = True
            keep[s:e] = m
    rows = np.repeat(np.arange(prod.shape[0], dtype=np.int64), counts)[keep]
    cols_k, data_k = cols[keep].astype(np.int64), data[keep]
    order = np.lexsort((cols_k, -data_k, rows))
    rows, cols_k, data_k = rows[order], cols_k[order], data_k[order]
    start = np.searchsorted(rows, rows, side="left")
    rank = np.arange(len(rows)) - start
    return rows, cols_k, data_k.astype(np.float32), rank.astype(np.int16)


def _chunks_by_flops(row_flops: np.ndarray, budget: float) -> List[Tuple[int, int]]:
    """Consecutive row ranges whose summed SpGEMM work stays within `budget`."""
    cum = np.cumsum(row_flops, dtype=np.float64)
    bounds, start = [], 0
    while start < len(row_flops):
        base = cum[start - 1] if start else 0.0
        end = int(np.searchsorted(cum, base + budget, side="right"))
        end = max(end, start + 1)
        bounds.append((start, min(end, len(row_flops))))
        start = end
    return bounds


def block_country(table: RecordTable, tokens: Dict[str, TokenCSR], q_rows: np.ndarray,
                  r_rows_by_src: Dict[int, np.ndarray], params: BlockingParams) -> Candidates:
    t0 = time.time()
    all_r = np.concatenate(list(r_rows_by_src.values()))
    if params.strategy == "A":
        q_tok = key_tokens(table, q_rows, tokens)
        r_tok = key_tokens(table, all_r, tokens)
        q_ids, q_ptr = q_tok.ids, q_tok.indptr
        r_ids, r_tix, r_ptr = r_tok.ids, np.zeros(len(r_tok.ids), np.int8), r_tok.indptr
        caps = np.array([params.max_df])
    else:
        q_ids, _, q_ptr = _combine(tokens, q_rows, params.token_types)
        r_ids, r_tix, r_ptr = _combine(tokens, all_r, params.token_types)
        caps = np.array([params.max_df_by_type.get(t, params.max_df) for t in params.token_types])
    # Vocabulary + document frequency over Sources 2+3 of this country; tokens held by
    # more records than their type's cap are dropped as non-discriminating.
    vocab_all, first, r_inv, df_all = np.unique(r_ids, return_index=True, return_inverse=True,
                                                return_counts=True)
    keep_col = df_all <= caps[r_tix[first]]
    new_col = (np.cumsum(keep_col) - 1).astype(np.int32)
    n_col = int(keep_col.sum())
    # per-occurrence arrays span ~300M tokens for the largest country: keep them int32
    r_inv = r_inv.astype(np.int32)
    r_cols = np.where(keep_col[r_inv], new_col[r_inv], np.int32(-1))
    del r_inv, r_tix, first
    df = df_all[keep_col].astype(np.float64)
    n_r = len(all_r)
    idf = np.log(n_r / df).astype(np.float32) if params.strategy == "B" else np.ones(n_col, np.float32)

    # Q matrix carries the weights; R matrices are binary.
    pos = sorted_lookup(vocab_all, q_ids)
    del vocab_all, r_ids
    ok = pos >= 0
    q_cols = np.full(len(q_ids), -1, dtype=np.int32)
    q_cols[ok] = np.where(keep_col[pos[ok]], new_col[pos[ok]], -1)
    del pos, ok
    q_vals = np.where(q_cols >= 0, idf[np.maximum(q_cols, 0)], 0).astype(np.float32)
    Q = _to_matrix(q_cols, q_vals, q_ptr, n_col)
    # SpGEMM work per query row = sum of df of its kept tokens (known before multiplying)
    Qb = Q.copy()
    Qb.data[:] = 1
    row_flops = np.asarray(Qb @ df).ravel()

    out_q, out_r, out_s, out_k = [], [], [], []
    offset = 0
    n_chunks = 0
    empty_flag = table.addr_empty
    for src, rr in r_rows_by_src.items():
        n = len(rr)
        lo, hi = r_ptr[offset], r_ptr[offset + n]
        R = _to_matrix(r_cols[lo:hi], np.ones(hi - lo, np.float32), r_ptr[offset:offset + n + 1] - lo, n_col)
        offset += n
        is_empty = empty_flag[rr]
        channels = [(~is_empty, params.k)]
        if params.strategy == "B" and params.k_empty > 0:
            channels.append((is_empty, params.k_empty))
        else:
            channels = [(np.ones(n, dtype=bool), params.k)]
        for sel, k in channels:
            idx = np.flatnonzero(sel)
            RT = R[idx].T.tocsr()
            for s, e in _chunks_by_flops(row_flops, params.flops_per_chunk):
                prod = Q[s:e] @ RT
                rows, cols, score, rank = _topk_rows(prod, k, all_pairs=(params.strategy == "A"))
                out_q.append(q_rows[rows + s])
                out_r.append(rr[idx[cols]])
                out_s.append(score)
                out_k.append(rank)
                n_chunks += 1
            del RT
        del R
    cands = Candidates(np.concatenate(out_q).astype(np.int32), np.concatenate(out_r).astype(np.int32),
                       np.concatenate(out_s), np.concatenate(out_k))
    log.info("blocking[%s] %d S1 x %d S2/S3 -> %d pairs (%.1f/S1) vocab=%d flops/S1=%.0f chunks=%d in %.0fs",
             params.strategy, len(q_rows), n_r, len(cands), len(cands) / max(1, len(q_rows)), n_col,
             row_flops.mean() if len(row_flops) else 0, n_chunks, time.time() - t0)
    return cands


def partitions(table: RecordTable, q_subset: np.ndarray = None):
    """Yield (country_label, S1 rows, {source: S2/S3 rows}) for every country present."""
    src, ctry = table.source, table.country
    for code, label in enumerate(table.countries):
        in_c = ctry == code
        q = np.flatnonzero(in_c & (src == 1))
        if q_subset is not None:
            q = np.intersect1d(q, q_subset)
        r = {s: np.flatnonzero(in_c & (src == s)) for s in (2, 3)}
        if len(q):
            yield label, q, r


def run_blocking(table: RecordTable, tokens: Dict[str, TokenCSR], params: BlockingParams,
                 q_subset: np.ndarray = None) -> Candidates:
    parts = [block_country(table, tokens, q, r, params) for _, q, r in partitions(table, q_subset)]
    return Candidates(*(np.concatenate([getattr(c, f) for c in parts])
                        for f in ("q", "r", "score", "rank")))
