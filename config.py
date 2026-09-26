"""Paths, seeds and tunable parameters for the whole pipeline (single source of truth)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

SEED = 2026


@dataclass
class Config:
    data_dir: str                      # folder containing train/ and test/
    work_dir: str                      # cache for intermediate artefacts
    output_dir: str                    # where matching_results.tsv / candidate_pairs.tsv go
    workers: int = field(default_factory=lambda: max(1, min(7, (os.cpu_count() or 2) - 1)))
    seed: int = SEED

    def split_dir(self, split: str) -> str:
        return os.path.join(self.data_dir, split)

    def source_path(self, split: str, source: int) -> str:
        return os.path.join(self.split_dir(split), f"{split}_source{source}.tsv")

    def truth_path(self) -> str:
        return os.path.join(self.split_dir("train"), "train_ground_truth.tsv")

    def work(self, *parts: str) -> str:
        path = os.path.join(self.work_dir, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
