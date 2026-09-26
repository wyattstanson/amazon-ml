"""Stage 1: normalise every record of a split once and cache it as a RecordTable.

Normalisation is pure-Python string work (~75 us/record), so it is spread over a
process pool; each worker returns already-encoded columns for its chunk to keep
inter-process traffic and peak memory small.

Source 1 is normalised first; its name vocabulary then drives the word segmenter used
for Sources 2/3 (concatenated / domain-style names such as "searsplatforms").
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from collections import Counter
from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np

from .cache import preprocess_fp, stamp_ok, write_stamp
from .config import Config
from .io_utils import id_key, iter_source_lines, parse_line
from .normalize import Segmenter, normalize_address, normalize_name
from .store import RecordTable, StrCol
from .tokens import bounded_imap

log = logging.getLogger(__name__)
CHUNK = 50_000
_SEGMENTER = None            # set in pool workers that normalise Sources 2/3


def _init_worker(segmenter) -> None:
    global _SEGMENTER
    _SEGMENTER = segmenter


def build_segmenter(name_cols) -> Segmenter:
    """Unigram vocabulary of Source-1 name words (core + full name, all countries)."""
    counts: Counter = Counter()
    for col in name_cols:
        for text in col.to_list():
            counts.update(text.split())
    return Segmenter(counts)


def _normalize_chunk(lines: List[str]) -> Tuple[Dict[str, StrCol], Dict[str, np.ndarray], List[str]]:
    cols: Dict[str, List[str]] = {k: [] for k in RecordTable.TEXT}
    is_domain, is_indic, addr_empty, countries, keys = [], [], [], [], []
    for line in lines:
        eid, name, addr, country = parse_line(line)
        n = normalize_name(name, _SEGMENTER)
        a = normalize_address(addr, country)
        keys.append(id_key(eid))
        cols["entity_id"].append(eid)
        cols["name_full"].append(n.full)
        cols["name_core"].append(n.core)
        cols["name_alias"].append(n.alias)
        cols["legal"].append(n.legal)
        cols["addr_norm"].append(a.norm)
        cols["state"].append(a.state)
        cols["nums"].append(" ".join(a.nums))
        cols["codes"].append(" ".join(a.codes))
        is_domain.append(n.is_domain)
        is_indic.append(n.is_indic)
        addr_empty.append(a.empty)
        countries.append(country)
    text = {k: StrCol.from_list(v) for k, v in cols.items()}
    numeric = {"is_domain": np.array(is_domain, dtype=bool),
               "is_indic": np.array(is_indic, dtype=bool),
               "addr_empty": np.array(addr_empty, dtype=bool),
               "key": np.array(keys, dtype=np.int64)}
    return text, numeric, countries


def build_table(cfg: Config, split: str) -> RecordTable:
    text_parts: Dict[str, List[StrCol]] = {k: [] for k in RecordTable.TEXT}
    num_parts: Dict[str, List[np.ndarray]] = {k: [] for k in ("is_domain", "is_indic", "addr_empty", "key")}
    source_parts: List[np.ndarray] = []
    country_parts: List[np.ndarray] = []
    labels: List[str] = []
    label_code: Dict[str, int] = {}
    segmenter = None
    for source in (1, 2, 3):
        if source == 2:           # S1 is done: build the segmenter from its name vocabulary
            s1_core = StrCol.concat(text_parts["name_core"])
            s1_full = StrCol.concat(text_parts["name_full"])
            segmenter = build_segmenter([s1_core, s1_full])
            log.info("segmenter vocabulary: %d words", len(segmenter.logp))
            del s1_core, s1_full
        path = cfg.source_path(split, source)
        t0 = time.time()
        n = 0
        with Pool(cfg.workers, initializer=_init_worker, initargs=(segmenter,)) as pool:
            for text, numeric, countries in bounded_imap(pool, _normalize_chunk,
                                                         iter_source_lines(path, CHUNK), 2 * cfg.workers):
                for k, v in text.items():
                    text_parts[k].append(v)
                for k, v in numeric.items():
                    num_parts[k].append(v)
                codes = np.empty(len(countries), dtype=np.int8)
                for i, c in enumerate(countries):
                    if c not in label_code:            # open set: codes assigned on first sight
                        label_code[c] = len(labels)
                        labels.append(c)
                    codes[i] = label_code[c]
                country_parts.append(codes)
                source_parts.append(np.full(len(countries), source, dtype=np.int8))
                n += len(countries)
        log.info("preprocess %s source%d: %d records in %.0fs", split, source, n, time.time() - t0)
    text_cols = {k: StrCol.concat(v) for k, v in text_parts.items()}
    numeric_cols = {k: np.concatenate(v) for k, v in num_parts.items()}
    numeric_cols["source"] = np.concatenate(source_parts)
    numeric_cols["country"] = np.concatenate(country_parts)
    return RecordTable(text_cols, numeric_cols, labels)


def table_dir(cfg: Config, split: str) -> str:
    return os.path.join(cfg.work_dir, split, "records")


def run(cfg: Config, split: str, force: bool = False) -> RecordTable:
    path = table_dir(cfg, split)
    if RecordTable.exists(path) and stamp_ok(path, preprocess_fp()) and not force:
        log.info("preprocess %s: using cached %s", split, path)
        return RecordTable.open(path)
    table = build_table(cfg, split)
    tmp = f"{path}.tmp-{os.getpid()}"            # atomic: build aside, then rename
    table.save(tmp)
    del table
    if os.path.exists(path):
        shutil.rmtree(path)
    os.replace(tmp, path)
    write_stamp(path, preprocess_fp())
    table = RecordTable.open(path)
    log.info("preprocess %s: %d records, countries=%s -> %s", split, len(table), table.countries, path)
    return table
