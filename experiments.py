"""Reproducible experiment runners behind the decisions logged in EXPERIMENTS.md.

`python -m ber experiment-blocking` compares blocking strategies / parameters on a fixed
random sample of training S1 entities, always against the *full* Source 2/3 pool (so
candidate density is exactly what the real run sees), and appends one JSON line per
configuration to work/experiments_blocking.jsonl.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, replace
from typing import List

import numpy as np

from . import preprocess, tokens as tokens_mod
from .blocking import BlockingParams, run_blocking
from .config import Config
from .evaluate import IdIndex, blocking_report_arrays, truth_pairs

log = logging.getLogger(__name__)


def load_train(cfg: Config):
    table = preprocess.run(cfg, "train")
    toks = tokens_mod.run(cfg, "train", table)
    index = IdIndex(table.key)
    s1_rows, gt_q, gt_r = truth_pairs(cfg.truth_path(), index)
    return table, toks, s1_rows, gt_q, gt_r


def sample_s1(s1_rows: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(s1_rows, size=min(n, len(s1_rows)), replace=False))


def blocking_grid(cfg: Config, configs: List[BlockingParams], n_sample: int, tag: str) -> None:
    table, toks, s1_rows, gt_q, gt_r = load_train(cfg)
    q = sample_s1(s1_rows, n_sample, cfg.seed)
    group = np.array(table.countries, dtype=object)[table.country]
    out_path = cfg.work("experiments_blocking.jsonl")
    for params in configs:
        t0 = time.time()
        c = run_blocking(table, toks, params, q_subset=q)
        rep = blocking_report_arrays(q, gt_q, gt_r, c.q, c.r, len(table), group=group)
        rep = {k: (float(v) if not isinstance(v, int) else v) for k, v in rep.items()}
        row = {"tag": tag, "n_s1": int(len(q)), "seconds": round(time.time() - t0, 1),
               "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(params).items()},
               **rep}
        log.info("EXP %s", json.dumps(row))
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


UNIGRAMS = ("name", "phon", "pref", "addrw", "addrc")
CONJ = ("street", "pa", "ca")


def _caps(uni: int, conj: int):
    return {**{t: uni for t in UNIGRAMS}, **{t: conj for t in CONJ}}


def blocking_grids():
    """Named, reproducible grids (python -m ber experiment-blocking --grid NAME)."""
    base = BlockingParams()
    return {
        # strategy A vs B, and the unigram-only vs conjunction-token ablation
        "strategies": [
            BlockingParams(strategy="A", max_df=500),
            BlockingParams(strategy="A", max_df=2000),
            replace(base, token_types=UNIGRAMS + ("street",), max_df=2000),
            replace(base, max_df=2000),
        ],
        # after name/phonetic bigrams + empty-address channel: depth and caps
        "v2": [
            replace(base),
            replace(base, k=16, k_empty=6),
            replace(base, max_df=5000),
        ],
        # after word segmentation + zero-run normalisation; deeper empty-address channel
        "v3": [
            replace(base, max_df=5000),
            replace(base, max_df=5000, k_empty=6),
        ],
        # per-type frequency caps: cost vs recall
        "caps": [
            replace(base, max_df=2000),
            replace(base, max_df_by_type=_caps(300, 2000)),
            replace(base, max_df_by_type=_caps(1000, 2000)),
            replace(base, max_df_by_type=_caps(300, 5000)),
            replace(base, token_types=("name", "phon") + CONJ, max_df_by_type=_caps(300, 2000)),
        ],
    }


def _finished_run_artefacts(cfg: Config):
    """Records, candidates, feature memmap and out-of-fold p1 of a completed training run,
    loaded as-is (no cache re-validation), so experiments never mix artefact versions."""
    import os
    from .blocking import Candidates
    from .store import RecordTable
    table = RecordTable.open(os.path.join(cfg.work_dir, "train", "records"))
    z = np.load(os.path.join(cfg.work_dir, "train", "candidates.npz"))
    cands = Candidates(z["q"], z["r"], z["score"], z["rank"])
    X = np.load(os.path.join(cfg.work_dir, "train", "features.npy"), mmap_mode="r")
    p1 = np.load(os.path.join(cfg.work_dir, "train", "oof_stage1.npy"))
    if not (len(cands) == X.shape[0] == len(p1)):
        raise ValueError("candidate / feature / out-of-fold artefacts are from different runs")
    return table, cands, X, p1


def feature_ablation(cfg: Config) -> dict:
    """Like-for-like comparison of feature sets on held-out macro F0.5.

    Protocol (identical for every variant): models are fit on a 5 % S1-entity sample of
    folds 1-2 (300 rounds, lr 0.1) and evaluated on every S1 of fold 0 (~735k entities,
    ~23M pairs) with the full decision-tuning grid; exclusivity is applied within the
    evaluation fold. Variants: base pair features; + support features; + support +
    stage-2 competition features; + the probability-weighted cluster-support block; + the
    neighbour-similarity block (all from the pipeline's out-of-fold stage-1 probabilities).
    Reads the artefacts of a completed `python -m ber train` run."""
    import os
    from . import model as model_mod, pipeline, stage2
    from .blocking import Candidates
    from .features import FEATURES, SUPPORT_FEATURES
    table, cands, X, p1 = _finished_run_artefacts(cfg)
    s1_rows, gt_q, gt_r = pipeline.train_truth(cfg, table)
    S = np.load(os.path.join(cfg.work_dir, "train", "stage2_features.npy"), mmap_mode="r")
    folds = model_mod.s1_folds(cands.q, len(table), pipeline.N_FOLDS, cfg.seed)
    tr = pipeline._sample_rows(np.flatnonzero(folds != 0), cands.q, 0.05, cfg.seed)
    ev = np.flatnonzero(folds == 0)
    s1_ev = s1_rows[model_mod.s1_folds(s1_rows, len(table), pipeline.N_FOLDS, cfg.seed) == 0]
    in_ev = np.isin(gt_q, s1_ev)
    cev = Candidates(cands.q[ev], cands.r[ev], cands.score[ev], cands.rank[ev])
    scorer = pipeline.F05Scorer(s1_ev, gt_q[in_ev], gt_r[in_ev], cev, len(table))
    y_all = pipeline.pair_labels(cands, gt_q, gt_r)
    base_cols = [i for i, f in enumerate(FEATURES) if f not in SUPPORT_FEATURES]
    all_cols = list(range(len(FEATURES)))
    n_comp = len(stage2.STAGE2_FEATURES)
    variants = {                                   # (pair-feature columns, stage-2 columns used)
        "base": (base_cols, 0),
        "base+support": (all_cols, 0),
        "base+support+stage2": (all_cols, n_comp),
        "base+support+stage2+mass": (all_cols, n_comp + len(stage2.MASS_FEATURES)),
        "base+support+stage2+mass+neighbours": (all_cols, len(pipeline.STAGE2_BLOCK)),
    }

    def gather(rows, cols, n_s2):
        parts = [np.asarray(X[rows])[:, cols]]
        if n_s2:
            parts.append(np.asarray(S[rows])[:, :n_s2])
        return np.hstack(parts)

    log_path = cfg.work("decision_tuning.jsonl")
    results = {}
    for name, (cols, n_s2) in variants.items():
        names = [FEATURES[i] for i in cols] + pipeline.STAGE2_BLOCK[:n_s2]
        booster = model_mod.train(gather(tr, cols, n_s2), y_all[tr], rounds=300,
                                  params={"learning_rate": 0.1}, feature_names=names)
        p = np.empty(len(ev), dtype=np.float32)
        for s in range(0, len(ev), 2_000_000):
            p[s:s + 2_000_000] = model_mod.predict(booster, gather(ev[s:s + 2_000_000], cols, n_s2))
        dp, res = pipeline.tune_decision(scorer, cev, p, f"ablation:{name}", log_path)
        results[name] = {"threshold": dp.threshold, "exclusive": dp.exclusive, "method": dp.method, **res}
        log.info("ABLATION %s -> F0.5 %.4f (singleton %.4f, P %.4f, R %.4f) at %s t=%.3f", name, res["f05"],
                 res["f05_singleton"], res["precision_micro"], res["recall_micro"], dp.method, dp.threshold)
    with open(cfg.work("ablation.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results
