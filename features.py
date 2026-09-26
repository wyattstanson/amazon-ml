"""Stage 3: pair features for candidate (S1, S2/S3) pairs.

All features are language-agnostic similarity signals; country is deliberately NOT a
feature, so the same model applies to France (absent from training). Groups:

  name     rapidfuzz similarities on the normalised core name, its space-free compact
           form (domain-style names), the full name, a former-name alias, and a
           phonetic-key string (bridges transliterated Indic names); IDF-weighted overlap
           of name tokens incl. the rarest shared and the rarest *missing* token.
  address  similarities on the order-normalised address, IDF-weighted overlap of words
           and number-codes, and explicit house/unit-number agreement -- the signal
           that separates true matches from the planted "sibling" negatives (same
           building, related name, different unit: 618-A/8 vs 618-A/5).
  context  blocking score/rank, and how many S1s compete for the same S2/S3 record
           (each S2/S3 record belongs to at most one S1 in the ground truth).
  support  how many of the S1's *other* candidates share this candidate's house number /
           name -- separates a coherent "sibling" entity from a one-off typo.

Heavy string work runs in rapidfuzz's multi-threaded C++ `cpdist`; IDF overlaps are
sparse element-wise products; only the number-set features are a Python loop.
"""
from __future__ import annotations

import zlib
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

from .blocking import Candidates, _to_matrix, sorted_lookup
from .normalize import phonetic_key
from .store import RecordTable
from .tokens import TokenCSR

FEATURES: List[str] = [
    # name strings
    "n_ratio", "n_tsort", "n_tset", "n_partial", "n_jw", "n_compact_ratio", "n_compact_partial",
    "n_full_ratio", "n_alias_tset", "n_phon_ratio", "n_first_eq", "n_tok_q", "n_tok_r",
    # name IDF overlaps
    "n_idf_shared", "n_idf_frac_q", "n_idf_frac_r", "n_idf_max_shared", "n_idf_miss_q", "n_idf_miss_r",
    "p_idf_frac_q", "p_idf_frac_r",
    # record flags
    "legal_rel", "r_is_domain", "r_is_indic", "r_has_alias", "r_source",
    # address strings / overlaps
    "a_ratio", "a_tsort", "a_tset", "a_idf_frac_q", "a_idf_frac_r", "a_idf_miss_q",
    "c_idf_frac_q", "c_idf_frac_r", "street_shared", "state_eq", "r_addr_empty",
    # numbers
    "num_first_q_in_r", "num_first_r_in_q", "num_jacc", "num_q_missing", "num_r_extra",
    "code_first_lev", "code_first_eq", "code_jacc",
    # blocking / competition context
    "blk_score", "blk_rank", "blk_frac_qmax", "r_ncand", "r_rank", "blk_frac_rmax",
    # support: agreement among the S1's other candidates (see support_features)
    "code_support", "code_support_s1", "name_support",
]
IDF_TYPES = ("name", "phon", "addrw", "addrc", "street")


# --------------------------------------------------------------------------------------
# IDF tables and sparse per-pair overlaps
# --------------------------------------------------------------------------------------
class IdfTable:
    """Sorted hashed-token vocabulary with IDF weights for one token type."""

    def __init__(self, tok: TokenCSR, rows: np.ndarray):
        ids = tok.rows(rows).ids
        self.vocab, df = np.unique(ids, return_counts=True)
        self.idf = np.log((len(rows) + 1) / df).astype(np.float32)

    def matrix(self, tok: TokenCSR, rows: np.ndarray) -> sp.csr_matrix:
        sub = tok.rows(rows)
        col = sorted_lookup(self.vocab, sub.ids)
        m = _to_matrix(col, self.idf[np.maximum(col, 0)], sub.indptr, len(self.vocab))
        m.sum_duplicates()          # canonical form required by element-wise products
        return m


def _row_max(m: sp.csr_matrix) -> np.ndarray:
    if m.nnz == 0:
        return np.zeros(m.shape[0], dtype=np.float32)
    return np.asarray(m.max(axis=1).todense()).ravel()


def idf_overlap(Q: sp.csr_matrix, R: sp.csr_matrix) -> Dict[str, np.ndarray]:
    """Per-row overlap stats between aligned rows of Q and R (both idf-weighted)."""
    Rb = R.copy()
    Rb.data[:] = 1
    Qb = Q.copy()
    Qb.data[:] = 1
    shared = Q.multiply(Rb).tocsr()
    q_tot = np.asarray(Q.sum(axis=1)).ravel()
    r_tot = np.asarray(R.sum(axis=1)).ravel()
    s = np.asarray(shared.sum(axis=1)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_q = np.where(q_tot > 0, s / q_tot, np.nan)
        frac_r = np.where(r_tot > 0, s / r_tot, np.nan)
    return {
        "shared": s, "frac_q": frac_q, "frac_r": frac_r,
        "max_shared": _row_max(shared),
        "miss_q": _row_max((Q - shared).tocsr()),
        "miss_r": _row_max((R - R.multiply(Qb)).tocsr()),
    }


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
# rapidfuzz threads per call; lowered in pool workers to avoid oversubscribing the CPU.
CPDIST_WORKERS = -1


def _sim(scorer, a: List[str], b: List[str]) -> np.ndarray:
    return cpdist(a, b, scorer=scorer, workers=CPDIST_WORKERS, dtype=np.float32)


def _phon_string(core: str) -> str:
    return " ".join(phonetic_key(w) for w in core.split())


def _legal_rel(a: str, b: str) -> int:
    if not a and not b:
        return 0
    if not a or not b:
        return 4
    sa, sb = set(a.split()), set(b.split())
    if sa == sb:
        return 1
    if sa <= sb or sb <= sa:
        return 2
    return 3


def _number_features(qn: List[str], rn: List[str], qc: List[str], rc: List[str]) -> np.ndarray:
    out = np.full((len(qn), 7), np.nan, dtype=np.float32)
    for i, (a, b, ca, cb) in enumerate(zip(qn, rn, qc, rc)):
        na, nb = a.split(), b.split()
        if na and nb:
            sa, sb = set(na), set(nb)
            out[i, 0] = na[0] in sb
            out[i, 1] = nb[0] in sa
            out[i, 2] = len(sa & sb) / len(sa | sb)
            out[i, 3] = len(sa - sb)
            out[i, 4] = len(sb - sa)
        ka, kb = ca.split(), cb.split()
        if ka and kb:
            out[i, 5] = ka[0] == kb[0]
            out[i, 6] = len(set(ka) & set(kb)) / len(set(ka) | set(kb))
    return out


def context_features(c: Candidates) -> Dict[str, np.ndarray]:
    """Blocking-score context over the whole candidate set (needs all pairs at once)."""
    n = len(c)
    score = c.score.astype(np.float64)
    q_max = np.zeros(int(c.q.max()) + 1 if n else 1)
    np.maximum.at(q_max, c.q, score)
    r_max = np.zeros(int(c.r.max()) + 1 if n else 1)
    np.maximum.at(r_max, c.r, score)
    r_n = np.bincount(c.r, minlength=len(r_max))
    # rank of this S1 among all S1s that have r as a candidate (0 = highest score)
    order = np.lexsort((c.q, -score, c.r))
    rs = c.r[order]
    first = np.searchsorted(rs, rs, side="left")
    r_rank = np.empty(n, dtype=np.float32)
    r_rank[order] = np.arange(n) - first
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "blk_score": c.score.astype(np.float32),
            "blk_rank": c.rank.astype(np.float32),
            "blk_frac_qmax": (score / q_max[c.q]).astype(np.float32),
            "r_ncand": r_n[c.r].astype(np.float32),
            "r_rank": r_rank,
            "blk_frac_rmax": (score / r_max[c.r]).astype(np.float32),
        }


SUPPORT_FEATURES = FEATURES[-3:]


def _hashes(strings: List[str], first_token: bool) -> np.ndarray:
    out = np.zeros(len(strings), dtype=np.int64)
    for i, s in enumerate(strings):
        if s:
            t = s.split(" ", 1)[0] if first_token else s.replace(" ", "")
            out[i] = zlib.crc32(t.encode("utf-8")) + 1          # 0 is reserved for "missing"
    return out


def support_features(table: RecordTable, c: Candidates) -> Dict[str, np.ndarray]:
    """How many *other* candidates of the same S1 agree with this candidate.

    Error analysis showed the planted "sibling" negatives (same name, neighbouring house
    number) are a coherent second entity: several of its records share its number. A true
    match's number typo is a one-off. So agreement among an S1's candidates separates a
    sibling cluster from noise where the pair features alone cannot.
    """
    ur, rinv = np.unique(c.r, return_inverse=True)
    uq, qinv = np.unique(c.q, return_inverse=True)
    rcode = _hashes(table.codes.take(ur), True)[rinv]
    qcode = _hashes(table.codes.take(uq), True)[qinv]
    rname = _hashes(table.name_core.take(ur), False)[rinv]
    q64 = c.q.astype(np.int64) << 32

    def group_count(h: np.ndarray) -> np.ndarray:
        _, inv, cnt = np.unique(q64 | (h & 0xFFFFFFFF), return_inverse=True, return_counts=True)
        return (cnt[inv] - 1).astype(np.float32)

    agree = ((rcode == qcode) & (qcode != 0)).astype(np.float64)
    nq = int(c.q.max()) + 1 if len(c.q) else 1
    return {
        "code_support": np.where(rcode != 0, group_count(rcode), np.nan).astype(np.float32),
        "code_support_s1": np.bincount(c.q, weights=agree, minlength=nq)[c.q].astype(np.float32),
        "name_support": np.where(rname != 0, group_count(rname), np.nan).astype(np.float32),
    }


# --------------------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------------------
class FeatureBuilder:
    """Computes the feature matrix for chunks of candidate pairs of one split."""

    def __init__(self, table: RecordTable, tokens: Dict[str, TokenCSR]):
        self.table = table
        self.tokens = tokens
        self._idf: Dict[Tuple[int, str], IdfTable] = {}

    def idf(self, country: int, ttype: str) -> IdfTable:
        key = (country, ttype)
        if key not in self._idf:
            rows = np.flatnonzero(self.table.country == country)
            self._idf[key] = IdfTable(self.tokens[ttype], rows)
        return self._idf[key]

    def build(self, q: np.ndarray, r: np.ndarray, ctx: Dict[str, np.ndarray]) -> np.ndarray:
        """Feature matrix (len(q) x len(FEATURES)) for pairs (q[i], r[i]); `ctx` holds the
        context features already sliced to the same pairs."""
        t = self.table
        n = len(q)
        F: Dict[str, np.ndarray] = {}
        uq, q_inv = np.unique(q, return_inverse=True)
        ur, r_inv = np.unique(r, return_inverse=True)

        def pull(col, uniq, inv):
            vals = col.take(uniq)
            return [vals[i] for i in inv]

        qn, rn = pull(t.name_core, uq, q_inv), pull(t.name_core, ur, r_inv)
        F["n_ratio"] = _sim(fuzz.ratio, qn, rn)
        F["n_tsort"] = _sim(fuzz.token_sort_ratio, qn, rn)
        F["n_tset"] = _sim(fuzz.token_set_ratio, qn, rn)
        F["n_partial"] = _sim(fuzz.partial_ratio, qn, rn)
        F["n_jw"] = _sim(JaroWinkler.normalized_similarity, qn, rn) * 100
        qc = [s.replace(" ", "") for s in qn]
        rc = [s.replace(" ", "") for s in rn]
        F["n_compact_ratio"] = _sim(fuzz.ratio, qc, rc)
        F["n_compact_partial"] = _sim(fuzz.partial_ratio, qc, rc)
        F["n_full_ratio"] = _sim(fuzz.ratio, pull(t.name_full, uq, q_inv), pull(t.name_full, ur, r_inv))
        ra = pull(t.name_alias, ur, r_inv)
        has_alias = np.array([bool(a) for a in ra])
        alias_sim = _sim(fuzz.token_set_ratio, qn, ra)
        F["n_alias_tset"] = np.where(has_alias, alias_sim, np.nan).astype(np.float32)
        qp_u = [_phon_string(s) for s in t.name_core.take(uq)]
        rp_u = [_phon_string(s) for s in t.name_core.take(ur)]
        F["n_phon_ratio"] = _sim(fuzz.ratio, [qp_u[i] for i in q_inv], [rp_u[i] for i in r_inv])
        F["n_first_eq"] = np.array([(a.split()[:1] == b.split()[:1]) and bool(a) for a, b in zip(qn, rn)],
                                   dtype=np.float32)
        F["n_tok_q"] = np.array([len(s.split()) for s in qn], dtype=np.float32)
        F["n_tok_r"] = np.array([len(s.split()) for s in rn], dtype=np.float32)

        # IDF overlaps, per country (IDF is computed within the country's records)
        ov = {k: np.full(n, np.nan, dtype=np.float32) for k in
              ("n_idf_shared", "n_idf_frac_q", "n_idf_frac_r", "n_idf_max_shared", "n_idf_miss_q",
               "n_idf_miss_r", "p_idf_frac_q", "p_idf_frac_r", "a_idf_frac_q", "a_idf_frac_r",
               "a_idf_miss_q", "c_idf_frac_q", "c_idf_frac_r", "street_shared")}
        ctry = t.country[q]
        for code in np.unique(ctry):
            m = np.flatnonzero(ctry == code)
            res = {}
            for tt in IDF_TYPES:
                tab = self.idf(int(code), tt)
                res[tt] = idf_overlap(tab.matrix(self.tokens[tt], q[m]), tab.matrix(self.tokens[tt], r[m]))
            ov["n_idf_shared"][m] = res["name"]["shared"]
            ov["n_idf_frac_q"][m] = res["name"]["frac_q"]
            ov["n_idf_frac_r"][m] = res["name"]["frac_r"]
            ov["n_idf_max_shared"][m] = res["name"]["max_shared"]
            ov["n_idf_miss_q"][m] = res["name"]["miss_q"]
            ov["n_idf_miss_r"][m] = res["name"]["miss_r"]
            ov["p_idf_frac_q"][m] = res["phon"]["frac_q"]
            ov["p_idf_frac_r"][m] = res["phon"]["frac_r"]
            ov["a_idf_frac_q"][m] = res["addrw"]["frac_q"]
            ov["a_idf_frac_r"][m] = res["addrw"]["frac_r"]
            ov["a_idf_miss_q"][m] = res["addrw"]["miss_q"]
            ov["c_idf_frac_q"][m] = res["addrc"]["frac_q"]
            ov["c_idf_frac_r"][m] = res["addrc"]["frac_r"]
            ov["street_shared"][m] = res["street"]["shared"]
        F.update(ov)

        F["legal_rel"] = np.array([_legal_rel(a, b) for a, b in
                                   zip(pull(t.legal, uq, q_inv), pull(t.legal, ur, r_inv))], dtype=np.float32)
        F["r_is_domain"] = t.is_domain[r].astype(np.float32)
        F["r_is_indic"] = t.is_indic[r].astype(np.float32)
        F["r_has_alias"] = has_alias.astype(np.float32)
        F["r_source"] = t.source[r].astype(np.float32)

        qa, ra_ = pull(t.addr_norm, uq, q_inv), pull(t.addr_norm, ur, r_inv)
        r_empty = t.addr_empty[r]
        for name, scorer in (("a_ratio", fuzz.ratio), ("a_tsort", fuzz.token_sort_ratio),
                             ("a_tset", fuzz.token_set_ratio)):
            F[name] = np.where(r_empty, np.nan, _sim(scorer, qa, ra_)).astype(np.float32)
        qs, rs = pull(t.state, uq, q_inv), pull(t.state, ur, r_inv)
        F["state_eq"] = np.array([(a == b) if (a and b) else np.nan for a, b in zip(qs, rs)], dtype=np.float32)
        F["r_addr_empty"] = r_empty.astype(np.float32)

        qnum, rnum = pull(t.nums, uq, q_inv), pull(t.nums, ur, r_inv)
        qcod, rcod = pull(t.codes, uq, q_inv), pull(t.codes, ur, r_inv)
        nf = _number_features(qnum, rnum, qcod, rcod)
        for j, name in enumerate(("num_first_q_in_r", "num_first_r_in_q", "num_jacc", "num_q_missing",
                                  "num_r_extra", "code_first_eq", "code_jacc")):
            F[name] = nf[:, j]
        q_first = [c.split()[0] if c else "" for c in qcod]
        r_first = [c.split()[0] if c else "" for c in rcod]
        lev = _sim(Levenshtein.distance, q_first, r_first)
        F["code_first_lev"] = np.where([bool(a) and bool(b) for a, b in zip(q_first, r_first)], lev, np.nan
                                       ).astype(np.float32)
        F.update(ctx)
        return np.column_stack([F[k] for k in FEATURES]).astype(np.float32)
