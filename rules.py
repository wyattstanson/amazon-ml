"""Matcher 1: transparent rule-based scorer (the baseline the learned model must beat).

    score = 0.55 * name_sim + 0.30 * addr_sim + 0.15 * number_agreement

name_sim  best of: token-set ratio on the core name; compact partial ratio when the
          candidate name is a domain ("schroederprinting.com"); former-name alias
          similarity; phonetic-key ratio when the candidate was transliterated.
addr_sim  token-set ratio of the order-normalised address (0.5 = neutral if empty).
number    1 if the first address codes agree, 0 if they disagree, 0.5 if unknown.

The weights are fixed by hand; only the decision threshold is tuned (for F0.5).
"""
from __future__ import annotations

import numpy as np

from .features import FEATURES

_IX = {f: i for i, f in enumerate(FEATURES)}


def rule_score(X: np.ndarray) -> np.ndarray:
    col = lambda f: X[:, _IX[f]].astype(np.float64)
    name = col("n_tset")
    name = np.where(col("r_is_domain") > 0, np.maximum(name, col("n_compact_partial")), name)
    name = np.maximum(name, np.nan_to_num(col("n_alias_tset"), nan=0.0))
    name = np.where(col("r_is_indic") > 0, np.maximum(name, col("n_phon_ratio")), name)
    addr = np.where(col("r_addr_empty") > 0, 50.0, col("a_tset"))
    num = np.nan_to_num(col("code_first_eq"), nan=0.5)
    return (0.55 * name / 100 + 0.30 * addr / 100 + 0.15 * num).astype(np.float32)
