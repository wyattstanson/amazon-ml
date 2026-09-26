"""Error analysis on out-of-fold training predictions (`python -m ber analyze`).

Splits every error of the chosen decision into categories and dumps concrete, randomly
sampled examples with their raw records and key features to
`work/error_analysis.md`, which is what the methodology write-up discusses.

* false negatives (missed true matches) = blocking misses (never a candidate) +
  matcher misses (candidate, but rejected);
* false positives (wrong merges), split into "S1 is a singleton" (costs the entity's
  whole 1.0) and "S1 has other true matches" (dilutes precision);
* each error type is profiled by candidate traits (empty address, Indic-script name,
  domain-style name, fka alias, low name similarity, unit-number disagreement) and
  compared with how common that trait is among all true matches.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

from . import preprocess, tokens as tokens_mod
from .config import Config
from .features import FEATURES
from .io_utils import iter_source_lines, parse_line
from .pipeline import F05Scorer, stage_block, train_truth
from .store import RecordTable

_F = {f: i for i, f in enumerate(FEATURES)}


def _raw_records(cfg: Config, split: str, wanted: set) -> Dict[str, List[str]]:
    """Original (un-normalised) name/address/country for the given entity ids."""
    out: Dict[str, List[str]] = {}
    for src in (1, 2, 3):
        for lines in iter_source_lines(cfg.source_path(split, src), 200_000):
            for line in lines:
                eid = line.split("\t", 1)[0]
                if eid in wanted:
                    out[eid] = parse_line(line)[1:]
    return out


def _traits(table: RecordTable, X: np.ndarray, r: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "empty_address": table.addr_empty[r],
        "indic_name": table.is_indic[r],
        "domain_name": table.is_domain[r],
        "fka_alias": ~np.isnan(X[:, _F["n_alias_tset"]]),
        "name_sim<60": X[:, _F["n_tset"]] < 60,
        "first_code_differs": X[:, _F["code_first_eq"]] == 0,
    }


def run(cfg: Config, n_examples: int = 25) -> str:
    table = preprocess.run(cfg, "train")
    toks = tokens_mod.run(cfg, "train", table)
    cands = stage_block(cfg, "train", table, toks)
    s1_rows, gt_q, gt_r = train_truth(cfg, table)
    X = np.load(os.path.join(cfg.work_dir, "train", "features.npy"), mmap_mode="r")
    oof = np.load(os.path.join(cfg.work_dir, "train", "oof_final.npy"))
    keep = np.load(os.path.join(cfg.work_dir, "train", "oof_keep.npy"))
    scorer = F05Scorer(s1_rows, gt_q, gt_r, cands, len(table))
    y = scorer.y
    group = np.array(table.countries, dtype=object)[table.country]

    # ---- false negatives: blocking vs matcher --------------------------------------
    n_true = int(scorer.n_true.sum())
    fn_matcher = np.flatnonzero(y & ~keep)
    fn_blocking = n_true - int(y.sum())
    fp = np.flatnonzero(keep & ~y)
    singleton_s1 = scorer.n_true[scorer.qd[fp]] == 0
    lines = ["# Error analysis (out-of-fold predictions, full training set)", ""]
    lines += [f"* true pairs: {n_true:,}",
              f"* missed by blocking (never a candidate): {fn_blocking:,} ({fn_blocking / n_true:.2%})",
              f"* missed by the matcher (candidate rejected): {len(fn_matcher):,} ({len(fn_matcher) / n_true:.2%})",
              f"* wrong merges (false positive pairs): {len(fp):,} of {int(keep.sum()):,} predicted "
              f"({len(fp) / max(1, keep.sum()):.2%}); on singleton S1s: {int(singleton_s1.sum()):,}", ""]
    ent = scorer.per_entity(keep)
    lines += ["| slice | S1 entities | macro F0.5 |", "|---|---|---|"]
    g = group[s1_rows]
    for label in np.unique(g):
        m = g == label
        lines.append(f"| {label} | {int(m.sum()):,} | {ent[m].mean():.4f} |")
    for name, m in (("singletons", scorer.n_true == 0), ("1 true match", scorer.n_true == 1),
                    ("2-4 true matches", (scorer.n_true >= 2) & (scorer.n_true <= 4)),
                    ("5+ true matches", scorer.n_true >= 5)):
        lines.append(f"| {name} | {int(m.sum()):,} | {ent[m].mean():.4f} |")
    lines.append("")

    # ---- trait profile ---------------------------------------------------------------
    pos = np.flatnonzero(y)
    lines += ["## Candidate traits: share among all true candidate pairs vs among errors", "",
              "| trait | all true pairs | matcher misses (FN) | wrong merges (FP) |", "|---|---|---|---|"]
    t_pos = _traits(table, np.asarray(X[pos]), cands.r[pos])
    t_fn = _traits(table, np.asarray(X[fn_matcher]), cands.r[fn_matcher])
    t_fp = _traits(table, np.asarray(X[fp]), cands.r[fp])
    for k in t_pos:
        lines.append(f"| {k} | {t_pos[k].mean():.1%} | {t_fn[k].mean():.1%} | {t_fp[k].mean():.1%} |")
    lines.append("")

    # ---- concrete examples ------------------------------------------------------------
    rng = np.random.default_rng(cfg.seed)
    picks = {
        "False positives on singleton S1s (each costs a full 1.0)": fp[singleton_s1],
        "False positives on non-singleton S1s": fp[~singleton_s1],
        "Matcher false negatives (true candidate rejected)": fn_matcher,
    }
    chosen = {k: (rng.choice(v, size=min(n_examples, len(v)), replace=False) if len(v) else v)
              for k, v in picks.items()}
    wanted = set()
    for v in chosen.values():
        wanted |= set(table.entity_id.take(cands.q[v])) | set(table.entity_id.take(cands.r[v]))
    raw = _raw_records(cfg, "train", wanted)
    show = ("n_tset", "n_jw", "a_tset", "num_jacc", "code_first_eq", "blk_frac_qmax", "r_ncand")
    for title, idx in chosen.items():
        lines += [f"## {title}", ""]
        for i in idx:
            qi, ri = table.entity_id[int(cands.q[i])], table.entity_id[int(cands.r[i])]
            feats = ", ".join(f"{f}={float(X[i, _F[f]]):.2f}" for f in show)
            lines.append(f"* p={oof[i]:.3f} | {feats}")
            lines.append(f"    * `{qi}` {' | '.join(raw.get(qi, ['?']))}")
            lines.append(f"    * `{ri}` {' | '.join(raw.get(ri, ['?']))}")
        lines.append("")
    path = os.path.join(cfg.work_dir, "error_analysis.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(cfg.work_dir, "error_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"true_pairs": n_true, "fn_blocking": fn_blocking, "fn_matcher": int(len(fn_matcher)),
                   "fp_pairs": int(len(fp)), "fp_on_singletons": int(singleton_s1.sum())}, f, indent=2)
    return path
