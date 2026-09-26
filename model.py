"""Matcher 2: gradient-boosted trees (LightGBM, MIT licence) over the pair features.

Determinism: fixed seed, `deterministic=True`, `force_row_wise=True` and a fixed thread
count make repeated runs produce bit-identical models and probabilities.
Folds are assigned per S1 entity (all candidates of one S1 share a fold), so
out-of-fold predictions never see their own entity during training.
"""
from __future__ import annotations

from typing import Dict, Optional

import lightgbm as lgb
import numpy as np

from .config import SEED
from .features import FEATURES

LGB_PARAMS: Dict = dict(
    objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255,
    seed=SEED, deterministic=True, force_row_wise=True, num_threads=8, verbose=-1,
)
NUM_ROUNDS = 600


def s1_folds(q_rows: np.ndarray, n_table: int, k: int, seed: int = SEED) -> np.ndarray:
    """Fold id per pair, derived from its S1 row through a seeded permutation."""
    perm = np.random.default_rng(seed).permutation(n_table)
    return (perm[q_rows] % k).astype(np.int8)


def train(X: np.ndarray, y: np.ndarray, rounds: int = NUM_ROUNDS, params: Optional[Dict] = None,
          valid: Optional[tuple] = None, feature_names: Optional[list] = None) -> lgb.Booster:
    p = dict(LGB_PARAMS, **(params or {}))
    dtrain = lgb.Dataset(X, label=y, feature_name=list(feature_names or FEATURES), free_raw_data=True)
    valid_sets, callbacks = [], []
    if valid is not None:
        valid_sets = [lgb.Dataset(valid[0], label=valid[1], reference=dtrain)]
        callbacks = [lgb.log_evaluation(100)]
        p["metric"] = "binary_logloss"
    return lgb.train(p, dtrain, num_boost_round=rounds, valid_sets=valid_sets, callbacks=callbacks)


def predict(booster: lgb.Booster, X: np.ndarray) -> np.ndarray:
    return booster.predict(X, num_threads=LGB_PARAMS["num_threads"]).astype(np.float32)
