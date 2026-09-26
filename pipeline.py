"""End-to-end orchestration.

train:  preprocess -> tokens -> blocking (+ recall report) -> pair features
        -> matcher 1 (rules) and matcher 2 (two-stage LightGBM), both scored out of fold
        -> decision tuning -> final models
test:   preprocess -> tokens -> blocking -> pair features -> stage-1 -> stage-2 -> decide
        -> output/candidate_pairs.tsv + output/matching_results.tsv

Matcher 2 is two LightGBM models. Stage 1 scores each pair from its own features.
Stage 2 adds "competition" features computed from stage-1 probabilities across the whole
candidate graph (is there a better candidate for this S1? does another S1 claim this
record more strongly?) -- trained on out-of-fold stage-1 probabilities, so it never sees
scores that were fit on its own rows.

Every stage caches its artefact under work/ with a fingerprint of its code and settings
(cache.py); an interrupted run resumes, a changed setting recomputes only what depends on
it. All randomness is seeded.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from multiprocessing import Pool
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import model as model_mod
from . import preprocess, stage2
from . import tokens as tokens_mod
from .blocking import BlockingParams, Candidates, run_blocking
from .cache import code_fp, fingerprint, stamp_ok, tokens_fp, write_stamp
from .config import Config
from .decide import DecisionParams, decide, exclusive_mask
from .evaluate import IdIndex, blocking_report_arrays, f05_from_counts, truth_pairs
from .features import FEATURES, FeatureBuilder, context_features, support_features
from .rules import rule_score
from .store import RecordTable

log = logging.getLogger(__name__)

# The configuration selected by the experiments recorded in EXPERIMENTS.md.
BLOCKING = BlockingParams(max_df=5000, k=20, k_empty=8)   # depth re-measured in EXPERIMENTS.md section 7
N_FOLDS = 3
TRAIN_FRACTION = 0.40          # share of an S1 fold-complement's entities used to fit each model (EXPERIMENTS.md 7.2)
FEATURE_CHUNK = 500_000        # candidate pairs per feature-computation task
FEATURE_PROCS = 4              # feature worker processes
PREDICT_CHUNK = 2_000_000
STAGE1_NAMES = list(FEATURES)
STAGE2_BLOCK = list(stage2.STAGE2_FEATURES) + list(stage2.MASS_FEATURES) + list(stage2.NEIGHBOUR_FEATURES)
STAGE2_NAMES = list(FEATURES) + STAGE2_BLOCK


# --------------------------------------------------------------------------------------
# Cache fingerprints (code content + settings, chained upstream; see cache.py)
# --------------------------------------------------------------------------------------
def blocking_fp() -> str:
    return fingerprint(upstream=tokens_fp(), code=code_fp("blocking"), blocking=asdict(BLOCKING))


def features_fp() -> str:
    return fingerprint(upstream=blocking_fp(), code=code_fp("features"), features=FEATURES)


def stage1_fp(cfg: Config) -> str:
    return fingerprint(upstream=features_fp(), code=code_fp("model"), lgb=model_mod.LGB_PARAMS,
                       rounds=model_mod.NUM_ROUNDS, folds=N_FOLDS, frac=TRAIN_FRACTION, seed=cfg.seed)


def stage2_fp(cfg: Config) -> str:
    return fingerprint(upstream=stage1_fp(cfg), code=code_fp("stage2"), features=STAGE2_NAMES)


def _p(cfg: Config, split: str, name: str) -> str:
    return os.path.join(cfg.work_dir, split, name)


# --------------------------------------------------------------------------------------
# Blocking
# --------------------------------------------------------------------------------------
def stage_block(cfg: Config, split: str, table: RecordTable, toks, force=False) -> Candidates:
    path = _p(cfg, split, "candidates.npz")
    if stamp_ok(path, blocking_fp()) and not force:
        z = np.load(path)
        return Candidates(z["q"], z["r"], z["score"], z["rank"])
    t0 = time.time()
    c = run_blocking(table, toks, BLOCKING)
    # canonical order: by S1 row, then candidate row -> deterministic downstream chunks
    order = np.lexsort((c.r, c.q))
    c = Candidates(c.q[order], c.r[order], c.score[order], c.rank[order])
    np.savez(path, q=c.q, r=c.r, score=c.score, rank=c.rank)
    write_stamp(path, blocking_fp())
    log.info("block %s: %d pairs in %.0fs", split, len(c), time.time() - t0)
    return c


# --------------------------------------------------------------------------------------
# Pair features (process pool writing straight into an on-disk memmap)
# --------------------------------------------------------------------------------------
_WORKER: Dict = {}


def _feat_init(work_dir: str, split: str, cpdist_workers: int) -> None:
    """Pool initialiser: open the memory-mapped table/tokens once per worker process."""
    from . import features as feat_mod
    feat_mod.CPDIST_WORKERS = cpdist_workers
    table = RecordTable.open(os.path.join(work_dir, split, "records"))
    toks = tokens_mod.open_tokens(os.path.join(work_dir, split, "tokens"))
    _WORKER["fb"] = FeatureBuilder(table, toks)


def _feat_task(args) -> int:
    path, s, e, q, r, ctx = args
    X = _WORKER["fb"].build(q, r, ctx)
    out = np.load(path, mmap_mode="r+")
    out[s:e] = X
    out.flush()
    del out
    return e - s


def stage_features(cfg: Config, split: str, table: RecordTable, cands: Candidates, force=False) -> np.ndarray:
    path = _p(cfg, split, "features.npy")
    if stamp_ok(path, features_fp()) and not force:
        return np.load(path, mmap_mode="r")
    # context + support features need the whole candidate graph at once
    ctx = {**context_features(cands), **support_features(table, cands)}
    X = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(cands), len(FEATURES)))
    del X
    bounds = list(range(0, len(cands), FEATURE_CHUNK)) + [len(cands)]
    tasks = ((path, s, e, cands.q[s:e], cands.r[s:e], {k: v[s:e] for k, v in ctx.items()})
             for s, e in zip(bounds[:-1], bounds[1:]))
    n_proc = max(1, min(FEATURE_PROCS, cfg.workers))
    t0 = time.time()
    done_rows = 0
    with Pool(n_proc, initializer=_feat_init,
              initargs=(cfg.work_dir, split, max(1, (os.cpu_count() or 2) // n_proc))) as pool:
        for n in tokens_mod.bounded_imap(pool, _feat_task, tasks, 2 * n_proc):
            done_rows += n
            if done_rows % 5_000_000 < FEATURE_CHUNK or done_rows == len(cands):
                log.info("features %s: %d/%d pairs (%.0fs)", split, done_rows, len(cands), time.time() - t0)
    write_stamp(path, features_fp())
    return np.load(path, mmap_mode="r")


def stage_competition(cfg: Config, split: str, table: RecordTable, cands: Candidates, p1: np.ndarray,
                      fp: str) -> np.ndarray:
    """Stage-2 features from stage-1 probabilities (competition, probability-weighted cluster
    support, similarity to the S1's other likely matches), written block by block into an
    on-disk memmap."""
    path = _p(cfg, split, "stage2_features.npy")
    if stamp_ok(path, fp):
        return np.load(path, mmap_mode="r")
    t0 = time.time()
    a = len(stage2.STAGE2_FEATURES)
    b = a + len(stage2.MASS_FEATURES)
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(cands), len(STAGE2_BLOCK)))
    out[:, :a] = stage2.matrix(cands, p1)
    out[:, a:b] = stage2.mass_matrix(table, cands, p1)
    out[:, b:] = stage2.neighbour_matrix(table, cands, p1)
    out.flush()
    del out
    write_stamp(path, fp)
    log.info("stage-2 features %s: %d pairs in %.0fs", split, len(cands), time.time() - t0)
    return np.load(path, mmap_mode="r")


# --------------------------------------------------------------------------------------
# Labels and scoring
# --------------------------------------------------------------------------------------
def train_truth(cfg: Config, table: RecordTable) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return truth_pairs(cfg.truth_path(), IdIndex(table.key))


def pair_labels(cands: Candidates, gt_q: np.ndarray, gt_r: np.ndarray) -> np.ndarray:
    key = cands.q.astype(np.int64) * (1 << 32) + cands.r.astype(np.int64)
    gkey = np.sort(gt_q.astype(np.int64) * (1 << 32) + gt_r.astype(np.int64))
    pos = np.minimum(np.searchsorted(gkey, key), len(gkey) - 1)
    return (gkey[pos] == key).astype(np.int8)


class F05Scorer:
    """Macro F0.5 of any keep-mask over a fixed candidate set, in two bincounts.

    Precomputes the S1 index of every candidate, the pair labels and each S1's number of
    true matches (including matches blocking missed), so a tuning grid costs O(pairs)
    per configuration instead of a sort of every pair key."""

    def __init__(self, s1_rows, gt_q, gt_r, cands: Candidates, n_rows: int):
        idx = np.full(n_rows, -1, dtype=np.int64)
        idx[s1_rows] = np.arange(len(s1_rows))
        self.s1_rows = s1_rows
        self.n = len(s1_rows)
        self.qd = idx[cands.q]
        self.y = pair_labels(cands, gt_q, gt_r).astype(bool)
        self.n_true = np.bincount(idx[gt_q], minlength=self.n)

    def per_entity(self, keep: np.ndarray) -> np.ndarray:
        tp = np.bincount(self.qd[keep & self.y], minlength=self.n)
        n_pred = np.bincount(self.qd[keep], minlength=self.n)
        return f05_from_counts(self.n_true, n_pred, tp)

    def summary(self, keep: np.ndarray, group: np.ndarray = None) -> Dict[str, float]:
        f = self.per_entity(keep)
        tp = int((keep & self.y).sum())
        out = {"f05": float(f.mean()),
               "f05_singleton": float(f[self.n_true == 0].mean()),
               "f05_nonsingleton": float(f[self.n_true > 0].mean()),
               "precision_micro": tp / max(1, int(keep.sum())),
               "recall_micro": tp / max(1, int(self.n_true.sum()))}
        if group is not None:
            g = group[self.s1_rows]
            for label in np.unique(g):
                out[f"f05[{label}]"] = float(f[g == label].mean())
        return out


THRESHOLD_GRID = np.round(np.arange(0.20, 0.91, 0.025), 3)
EXPECTED_F_FLOORS = (0.02, 0.05, 0.10, 0.20)
EXPECTED_F_BOOSTS = (0.8, 1.0, 1.25, 1.5, 2.0, 2.5)


def tune_decision(scorer: F05Scorer, cands: Candidates, p: np.ndarray, label: str,
                  log_path: str) -> Tuple[DecisionParams, Dict[str, float]]:
    """Grid-search the decision layer for macro F0.5; logs every configuration tried."""
    configs = [DecisionParams(exclusive=e, method="threshold", threshold=float(t))
               for e in (False, True) for t in THRESHOLD_GRID]
    configs += [DecisionParams(exclusive=True, method="expected_f", threshold=fl, singleton_boost=b)
                for fl in EXPECTED_F_FLOORS for b in EXPECTED_F_BOOSTS]
    excl = exclusive_mask(cands.q, cands.r, p)
    best, best_res = None, None
    with open(log_path, "a", encoding="utf-8") as f:
        for dp in configs:
            res = scorer.summary(decide(cands.q, cands.r, p, dp, excl=excl))
            f.write(json.dumps({"matcher": label, **asdict(dp), **res}) + "\n")
            if best_res is None or res["f05"] > best_res["f05"]:
                best, best_res = dp, res
    log.info("tune[%s]: best %s -> %s", label, best, best_res)
    return best, best_res


# --------------------------------------------------------------------------------------
# Model training (S1-grouped cross-fitting) and prediction over memmapped feature blocks
# --------------------------------------------------------------------------------------
GATHER_CHUNK = 2_000_000


def _gather(blocks: Sequence[np.ndarray], rows: np.ndarray) -> np.ndarray:
    """Rows of the column-concatenated blocks, filled into one preallocated float32 matrix in
    row chunks (an hstack of full-size slices would briefly hold the training matrix twice)."""
    widths = [b.shape[1] for b in blocks]
    out = np.empty((len(rows), sum(widths)), dtype=np.float32)
    for s in range(0, len(rows), GATHER_CHUNK):
        r = rows[s:s + GATHER_CHUNK]
        c0 = 0
        for b, w in zip(blocks, widths):
            out[s:s + len(r), c0:c0 + w] = b[r]
            c0 += w
    return out


def _sample_rows(mask_rows: np.ndarray, q: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Rows (indices into the candidate arrays) whose S1 is kept by a seeded S1-level draw."""
    uq = np.unique(q[mask_rows])
    rng = np.random.default_rng(seed)
    keep_q = uq[rng.random(len(uq)) < frac]
    return mask_rows[np.isin(q[mask_rows], keep_q)]


def _predict(booster, blocks: Sequence[np.ndarray], rows: np.ndarray) -> np.ndarray:
    out = np.empty(len(rows), dtype=np.float32)
    for s in range(0, len(rows), PREDICT_CHUNK):
        out[s:s + PREDICT_CHUNK] = model_mod.predict(booster, _gather(blocks, rows[s:s + PREDICT_CHUNK]))
    return out


def train_oof(cfg: Config, tag: str, blocks: Sequence[np.ndarray], names: List[str], y: np.ndarray,
              cands: Candidates, n_table: int, fp: str) -> np.ndarray:
    """K-fold (grouped by S1) out-of-fold probabilities for every training candidate pair."""
    path = _p(cfg, "train", f"oof_{tag}.npy")
    if stamp_ok(path, fp):
        return np.load(path)
    folds = model_mod.s1_folds(cands.q, n_table, N_FOLDS, cfg.seed)
    oof = np.zeros(len(cands), dtype=np.float32)
    for k in range(N_FOLDS):
        t0 = time.time()
        tr = _sample_rows(np.flatnonzero(folds != k), cands.q, TRAIN_FRACTION, cfg.seed + k)
        booster = model_mod.train(_gather(blocks, tr), y[tr], feature_names=names)
        te = np.flatnonzero(folds == k)
        oof[te] = _predict(booster, blocks, te)
        log.info("%s fold %d: trained on %d pairs, predicted %d in %.0fs", tag, k, len(tr), len(te),
                 time.time() - t0)
    np.save(path, oof)
    write_stamp(path, fp)
    return oof


def train_final(cfg: Config, tag: str, blocks: Sequence[np.ndarray], names: List[str], y: np.ndarray,
                cands: Candidates, fp: str):
    """Final model: same recipe and training-set size as one fold model, drawn from all S1s,
    so decision parameters tuned on out-of-fold probabilities transfer unchanged."""
    import lightgbm as lgb
    path = os.path.join(cfg.work_dir, f"model_{tag}.txt")
    if stamp_ok(path, fp):
        return lgb.Booster(model_file=path)
    frac = TRAIN_FRACTION * (N_FOLDS - 1) / N_FOLDS
    tr = _sample_rows(np.arange(len(cands)), cands.q, frac, cfg.seed + 100)
    booster = model_mod.train(_gather(blocks, tr), y[tr], feature_names=names)
    booster.save_model(path)
    write_stamp(path, fp)
    log.info("final %s model trained on %d pairs", tag, len(tr))
    return booster


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------
def write_outputs(cfg: Config, table: RecordTable, cands: Candidates, keep: np.ndarray,
                  chunk: int = 100_000) -> None:
    """Stream candidate_pairs.tsv and matching_results.tsv in one pass.

    One row per Source-1 entity in source-file order (empty list when nothing is
    found/matched). Candidates are sorted by S1 row, so each chunk of S1 entities maps to
    one contiguous slice -- only that slice's ids are ever decoded (the full candidate set
    would be ~54M strings). Matches are always a subset of the same row's candidates."""
    if len(cands.q) and not np.all(cands.q[1:] >= cands.q[:-1]):
        raise ValueError("candidates must be sorted by S1 row")
    os.makedirs(cfg.output_dir, exist_ok=True)
    s1_rows = np.flatnonzero(table.source == 1)
    starts = np.searchsorted(cands.q, s1_rows, side="left")
    ends = np.searchsorted(cands.q, s1_rows, side="right")
    ids = table.entity_id
    n_match = 0
    with open(os.path.join(cfg.output_dir, "candidate_pairs.tsv"), "w", encoding="utf-8", newline="\n") as fc, \
            open(os.path.join(cfg.output_dir, "matching_results.tsv"), "w", encoding="utf-8", newline="\n") as fm:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for c0 in range(0, len(s1_rows), chunk):
            c1 = min(len(s1_rows), c0 + chunk)
            s1_ids = ids.take(s1_rows[c0:c1])
            lo, hi = int(starts[c0]), int(ends[c1 - 1])
            r_ids = ids.take(cands.r[lo:hi])
            kp = keep[lo:hi]
            cand_lines, match_lines = [], []
            for j in range(c1 - c0):
                a, b = int(starts[c0 + j]) - lo, int(ends[c0 + j]) - lo
                row = r_ids[a:b]
                matched = [x for x, k in zip(row, kp[a:b]) if k]
                n_match += len(matched)
                cand_lines.append(f"{s1_ids[j]}\t{','.join(row)}\n")
                match_lines.append(f"{s1_ids[j]}\t{','.join(matched)}\n")
            fc.writelines(cand_lines)
            fm.writelines(match_lines)
    if n_match != int(keep.sum()):
        raise AssertionError("some matches fall outside the S1 rows of this table")
    log.info("wrote outputs for %d S1 entities: %d candidate pairs, %d matches", len(s1_rows),
             len(cands), n_match)


# --------------------------------------------------------------------------------------
# Full runs
# --------------------------------------------------------------------------------------
def _rule_scores(X) -> np.ndarray:
    out = np.empty(X.shape[0], dtype=np.float32)
    for s in range(0, X.shape[0], PREDICT_CHUNK):
        out[s:s + PREDICT_CHUNK] = rule_score(np.asarray(X[s:s + PREDICT_CHUNK]))
    return out


def run_train(cfg: Config, force: bool = False) -> Dict:
    """Everything that needs labels: blocking report, both matchers, decision tuning, final models."""
    table = preprocess.run(cfg, "train", force=force)
    toks = tokens_mod.run(cfg, "train", table, force=force)
    cands = stage_block(cfg, "train", table, toks, force)
    s1_rows, gt_q, gt_r = train_truth(cfg, table)
    group = np.array(table.countries, dtype=object)[table.country]
    metrics: Dict = {"blocking": blocking_report_arrays(s1_rows, gt_q, gt_r, cands.q, cands.r, len(table), group)}
    log.info("blocking report (full train): %s", metrics["blocking"])
    X = stage_features(cfg, "train", table, cands, force)
    scorer = F05Scorer(s1_rows, gt_q, gt_r, cands, len(table))
    y = scorer.y.astype(np.int8)
    tune_log = os.path.join(cfg.work_dir, "decision_tuning.jsonl")
    if os.path.exists(tune_log):
        os.remove(tune_log)

    # Matcher 1: rules (no training, so its scores are already out of sample)
    dp, res = tune_decision(scorer, cands, _rule_scores(X), "rules", tune_log)
    metrics["rules"] = {"decision": asdict(dp), **res}
    # Matcher 2, stage 1: pair features only
    p1 = train_oof(cfg, "stage1", [X], STAGE1_NAMES, y, cands, len(table), stage1_fp(cfg))
    dp, res = tune_decision(scorer, cands, p1, "lgbm_stage1", tune_log)
    metrics["lgbm_stage1"] = {"decision": asdict(dp), **res}
    # Matcher 2, stage 2: + competition features from out-of-fold stage-1 probabilities
    S = stage_competition(cfg, "train", table, cands, p1, stage2_fp(cfg))
    p2 = train_oof(cfg, "stage2", [X, S], STAGE2_NAMES, y, cands, len(table), stage2_fp(cfg))
    dp, res = tune_decision(scorer, cands, p2, "lgbm_stage2", tune_log)
    keep = decide(cands.q, cands.r, p2, dp)
    metrics["lgbm_stage2"] = {"decision": asdict(dp), **scorer.summary(keep, group)}
    np.save(_p(cfg, "train", "oof_final.npy"), p2)
    np.save(_p(cfg, "train", "oof_keep.npy"), keep)

    train_final(cfg, "stage1", [X], STAGE1_NAMES, y, cands, stage1_fp(cfg))
    train_final(cfg, "stage2", [X, S], STAGE2_NAMES, y, cands, stage2_fp(cfg))
    with open(os.path.join(cfg.work_dir, "decision.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(dp), f, indent=2)
    with open(os.path.join(cfg.work_dir, "metrics_train.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=float)
    log.info("train metrics: %s", json.dumps({k: v.get("f05") for k, v in metrics.items() if isinstance(v, dict)}))
    return metrics


def run_test(cfg: Config, force: bool = False) -> None:
    import lightgbm as lgb
    with open(os.path.join(cfg.work_dir, "decision.json"), encoding="utf-8") as f:
        dp = DecisionParams(**json.load(f))
    m1 = lgb.Booster(model_file=os.path.join(cfg.work_dir, "model_stage1.txt"))
    m2 = lgb.Booster(model_file=os.path.join(cfg.work_dir, "model_stage2.txt"))
    table = preprocess.run(cfg, "test", force=force)
    toks = tokens_mod.run(cfg, "test", table, force=force)
    cands = stage_block(cfg, "test", table, toks, force)
    X = stage_features(cfg, "test", table, cands, force)
    rows = np.arange(len(cands))
    p1 = _predict(m1, [X], rows)
    S = stage_competition(cfg, "test", table, cands, p1, stage2_fp(cfg))
    p2 = _predict(m2, [X, S], rows)
    np.save(_p(cfg, "test", "proba_stage1.npy"), p1)
    np.save(_p(cfg, "test", "proba_final.npy"), p2)
    write_outputs(cfg, table, cands, decide(cands.q, cands.r, p2, dp))
