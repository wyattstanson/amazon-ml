"""Held-out matcher experiments (EXPERIMENTS.md section 7.2).

Protocol: fit one model on a seeded S1-entity sample of folds 1-2, evaluate on every S1 of
fold 0 with the exact macro F0.5 after a decision grid (exclusivity + global threshold, or
expected-F0.5 selection). Stage-2 inputs are the out-of-fold stage-1 probabilities of a
completed `python -m ber train` run, so every number is out of sample.

    python tools/matcher_experiments.py --work-dir work --stage 2 --extra mass,nb --tag E3

Needs the artefacts of a finished training run in --work-dir (records, candidates,
features.npy, stage2_features.npy, oof_stage1.npy); appends one JSON line per run to
<work-dir>/experiments_matcher.jsonl. (The table in EXPERIMENTS.md 7.2 was produced with an
earlier copy of this script whose rival-cluster feature broke exact mass ties differently;
numbers can differ in the fifth decimal.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, "src"))

from ber import model as M  # noqa: E402
from ber import stage2  # noqa: E402
from ber.blocking import Candidates  # noqa: E402
from ber.decide import DecisionParams, decide, exclusive_mask  # noqa: E402
from ber.evaluate import IdIndex, f05_from_counts, truth_pairs  # noqa: E402
from ber.features import FEATURES  # noqa: E402
from ber.pipeline import pair_labels  # noqa: E402
from ber.store import RecordTable  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class HeldOut:
    """Fold-0 evaluation set with a two-bincount macro-F0.5 scorer."""

    def __init__(self, work: str, truth_path: str):
        t = os.path.join(work, "train")
        self.table = RecordTable.open(os.path.join(t, "records"))
        self.n = len(self.table)
        z = np.load(os.path.join(t, "candidates.npz"))
        self.c = Candidates(z["q"], z["r"], z["score"], z["rank"])
        s1, self.gq, self.gr = truth_pairs(truth_path, IdIndex(np.asarray(self.table.key)))
        self.y = pair_labels(self.c, self.gq, self.gr).astype(bool)
        perm = np.random.default_rng(M.SEED).permutation(self.n)
        self.row_fold = (perm % 3).astype(np.int8)
        self.fold = self.row_fold[self.c.q]
        self.eval_idx = np.flatnonzero(self.fold == 0)
        self.eval_s1 = s1[self.row_fold[s1] == 0]
        self.X = np.load(os.path.join(t, "features.npy"), mmap_mode="r")
        self.S = np.load(os.path.join(t, "stage2_features.npy"), mmap_mode="r")
        self.dir = t
        log(f"{len(self.c):,} pairs; eval fold {len(self.eval_idx):,} pairs / {len(self.eval_s1):,} S1")

    def train_rows(self, frac: float, seed: int = M.SEED + 1) -> np.ndarray:
        pick = np.random.default_rng(seed).random(self.n) < frac
        return np.flatnonzero((self.fold != 0) & pick[self.c.q])

    def score(self, p: np.ndarray):
        idx = self.eval_idx
        q, r = self.c.q[idx], self.c.r[idx]
        dense = np.full(self.n, -1, dtype=np.int64)
        dense[self.eval_s1] = np.arange(len(self.eval_s1))
        n_true = np.bincount(dense[self.gq[self.row_fold[self.gq] == 0]], minlength=len(self.eval_s1))
        qd, y = dense[q], self.y[idx]
        excl = exclusive_mask(q, r, p)
        cfgs = [DecisionParams(True, "threshold", float(t)) for t in np.arange(0.40, 0.91, 0.05)]
        cfgs += [DecisionParams(True, "expected_f", f, b) for f in (0.02, 0.05, 0.1)
                 for b in (1.0, 1.25, 1.5, 2.0, 2.5)]
        best = (-1.0, None)
        for d in cfgs:
            keep = decide(q, r, p, d, excl=excl)
            f = f05_from_counts(n_true, np.bincount(qd[keep], minlength=len(self.eval_s1)),
                                np.bincount(qd[keep & y], minlength=len(self.eval_s1))).mean()
            if f > best[0]:
                best = (float(f), d)
        return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", default=os.path.join(PKG, "work"))
    ap.add_argument("--truth", default=os.path.join(PKG, "..", "..", "student_resource", "dataset", "train",
                                                    "train_ground_truth.tsv"))
    ap.add_argument("--stage", type=int, default=2, choices=[1, 2])
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--leaves", type=int, default=127)
    ap.add_argument("--min-leaf", type=int, default=200)
    ap.add_argument("--extra", default="", help="comma list of blocks recomputed from --extra-p: mass, nb")
    ap.add_argument("--extra-p", default="oof_stage1.npy")
    ap.add_argument("--comp-p", default="", help="stage 3: competition features from these probabilities")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    h = HeldOut(a.work_dir, a.truth)
    n_comp = len(stage2.STAGE2_FEATURES)
    blocks = [lambda idx: np.asarray(h.X[idx])]         # row readers: memmaps stay on disk
    names = list(FEATURES)
    if a.stage == 2:
        blocks.append(lambda idx: np.asarray(h.S[idx])[:, :n_comp])
        names += stage2.STAGE2_FEATURES
    for e in [x for x in a.extra.split(",") if x]:
        p_in = np.load(os.path.join(h.dir, a.extra_p))
        fn = {"mass": stage2.mass_matrix, "nb": stage2.neighbour_matrix}[e]
        blocks.append(lambda idx, m=fn(h.table, h.c, p_in): m[idx])
        names += {"mass": stage2.MASS_FEATURES, "nb": stage2.NEIGHBOUR_FEATURES}[e]
    if a.comp_p:
        blocks.append(lambda idx, m=stage2.matrix(h.c, np.load(os.path.join(h.dir, a.comp_p))): m[idx])
        names += [f"{x}@{a.comp_p}" for x in stage2.STAGE2_FEATURES]

    def rows(idx):
        return np.hstack([read(idx) for read in blocks])

    t0 = time.time()
    tr = h.train_rows(a.frac)
    booster = M.train(rows(tr), h.y[tr].astype(np.float32), rounds=a.rounds,
                      params={"num_leaves": a.leaves, "min_data_in_leaf": a.min_leaf}, feature_names=names)
    p = np.empty(len(h.eval_idx), dtype=np.float32)
    for s in range(0, len(p), 4_000_000):
        p[s:s + 4_000_000] = M.predict(booster, rows(h.eval_idx[s:s + 4_000_000]))
    f, d = h.score(p)
    rec = {"tag": a.tag, "stage": a.stage, "frac": a.frac, "rounds": a.rounds, "leaves": a.leaves,
           "extra": a.extra, "comp_p": a.comp_p, "train_rows": int(len(tr)), "f05": f, "decision": d.__dict__,
           "seconds": round(time.time() - t0)}
    log(json.dumps(rec))
    with open(os.path.join(a.work_dir, "experiments_matcher.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
