"""Typed blocking tokens, hashed to stable 64-bit ids and stored as CSR per token type.

Token types (each a separate CSR so blocking experiments can combine them freely):
  name   core-name words and alias words             ("herrera", "family")
  phon   phonetic keys of those words                ("1613" for private/praivet)
  pref   5-char prefixes of name words and of the     ("schro" links "schroederprinting"
         compact (space-free) core name                to "schroeder printing")
  addrw  address words, state excluded                ("martin", "st", "andover")
  addrc  address codes containing digits              ("13802", "662c6")
  street house-number + following word bigrams        ("13802|martin")
  pa     phonetic name key x address word            ("2165|jheel")  -- see below
  ca     address code x address word                  ("101|lucknow")
  nb     name bigrams (consecutive core words, order-   ("brothers baba", "=elementalbrothers")
         free) + the whole space-free core name
  pb     the same over phonetic keys                    ("1 16" for smart power/smart pavar)

Why the cross-field conjunctions: Indian business names are extremely generic
("shivam builders", "raj construction": name words in 10-20K records) and so are city
words ("kolkata" 80K, "howrah" 166K). Any unigram frequent enough to be dropped by the
block-size cap takes the decisive evidence with it -- in a diagnostic of missed training
pairs, 100% shared tokens but 70% ranked outside the top-k and 30% had every shared
token capped away. The *combination* "shivam near rewa" is rare, so conjunction tokens
keep that evidence both cheap to index and highly weighted.

Token ids are CRC32("type:token") (stdlib zlib; deterministic across runs and platforms,
unlike Python's salted `hash()`), so no vocabulary dict over millions of strings is kept.
With ~10M distinct tokens in a 2^32 space about 0.1% of tokens share an id with another
token -- a negligible amount of extra blocking noise, in exchange for halving memory.
"""
from __future__ import annotations

import logging
import os
import shutil
import time
import zlib
from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np

from .cache import stamp_ok, tokens_fp, write_stamp
from .config import Config
from .normalize import phonetic_key
from .store import RecordTable

log = logging.getLogger(__name__)
TOKEN_TYPES = ("name", "phon", "pref", "addrw", "addrc", "street", "pa", "ca", "nb", "pb")
CHUNK = 100_000


def hid(s: str) -> int:
    return zlib.crc32(s.encode("utf-8"))


def _has_digit(t: str) -> bool:
    return any(c.isdigit() for c in t)


def record_tokens(core: str, alias: str, addr_norm: str, state: str) -> Dict[str, List[str]]:
    words = core.split()
    if alias:
        words = words + [w for w in alias.split() if w not in words]
    out: Dict[str, List[str]] = {k: [] for k in TOKEN_TYPES}
    out["name"] = words
    out["phon"] = sorted({phonetic_key(w) for w in words if len(w) > 2 and not w.isdigit()})
    pref = {w[:5] for w in words if len(w) >= 5}
    compact = core.replace(" ", "")
    if len(compact) >= 5:
        pref.add(compact[:5])
    out["pref"] = sorted(pref)
    toks = addr_norm.split()
    state_seen = False
    for i, t in enumerate(toks):
        if _has_digit(t):
            out["addrc"].append(t)
            if i + 1 < len(toks) and not _has_digit(toks[i + 1]):
                out["street"].append(t + "|" + toks[i + 1])
        elif t == state and not state_seen:
            state_seen = True            # the state code is not a discriminating word
        else:
            out["addrw"].append(t)
    for k in ("addrw", "addrc", "street"):
        out[k] = sorted(set(out[k]))
    out["pa"] = [p + "|" + a for p in out["phon"] for a in out["addrw"]]
    out["ca"] = [c + "|" + a for c in out["addrc"] for a in out["addrw"]]
    nb, pb = set(), set()
    for text in (core, alias):
        ws = text.split()
        if not ws:
            continue
        keys = [phonetic_key(w) for w in ws]
        nb.add("=" + "".join(ws))                 # whole compact name (concatenations, domains)
        pb.add("=" + " ".join(keys))
        for a, b in zip(ws, ws[1:]):
            nb.add(" ".join(sorted((a, b))))
        for a, b in zip(keys, keys[1:]):
            pb.add(" ".join(sorted((a, b))))
    out["nb"] = sorted(nb)
    out["pb"] = sorted(pb)
    return out


def _tokenize_chunk(args: Tuple[List[str], List[str], List[str], List[str]]):
    cores, aliases, addrs, states = args
    res = {}
    for k in TOKEN_TYPES:
        res[k] = ([], [0])
    for core, alias, addr, state in zip(cores, aliases, addrs, states):
        toks = record_tokens(core, alias, addr, state)
        for k in TOKEN_TYPES:
            ids, ptr = res[k]
            ids.extend(hid(k + ":" + t) for t in toks[k])
            ptr.append(len(ids))
    return {k: (np.array(ids, dtype=np.uint32), np.array(ptr, dtype=np.int64)) for k, (ids, ptr) in res.items()}


class TokenCSR:
    """Per-record lists of hashed token ids: row i -> ids[indptr[i]:indptr[i+1]]."""

    def __init__(self, ids: np.ndarray, indptr: np.ndarray):
        self.ids = ids
        self.indptr = indptr

    def __len__(self):
        return len(self.indptr) - 1

    def rows(self, idx: np.ndarray) -> "TokenCSR":
        idx = np.asarray(idx, dtype=np.int64)
        starts = np.asarray(self.indptr[idx])
        lens = np.asarray(self.indptr[idx + 1]) - starts
        indptr = np.zeros(len(idx) + 1, dtype=np.int64)
        np.cumsum(lens, out=indptr[1:])
        # vectorised gather of variable-length slices
        pos = np.repeat(starts - indptr[:-1], lens) + np.arange(indptr[-1])
        return TokenCSR(np.asarray(self.ids[pos]), indptr)


def bounded_imap(pool, func, items, window: int):
    """Like pool.imap, but keeps at most `window` tasks in flight, so a lazy input
    iterator is never materialised in full (Pool.imap drains its input eagerly)."""
    pending = []
    for item in items:
        pending.append(pool.apply_async(func, (item,)))
        if len(pending) >= window:
            yield pending.pop(0).get()
    for res in pending:
        yield res.get()


def build_tokens(cfg: Config, split: str, table: RecordTable) -> Dict[str, TokenCSR]:
    n = len(table)

    def chunks():
        for s in range(0, n, CHUNK):
            idx = np.arange(s, min(n, s + CHUNK))
            yield (table.name_core.take(idx), table.name_alias.take(idx),
                   table.addr_norm.take(idx), table.state.take(idx))

    t0 = time.time()
    parts: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {k: [] for k in TOKEN_TYPES}
    with Pool(cfg.workers) as pool:
        for res in bounded_imap(pool, _tokenize_chunk, chunks(), 2 * cfg.workers):
            for k in TOKEN_TYPES:
                parts[k].append(res[k])
    out = {}
    for k in TOKEN_TYPES:
        ids = np.concatenate([p[0] for p in parts[k]])
        lens = np.concatenate([np.diff(p[1]) for p in parts[k]])
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(lens, out=indptr[1:])
        out[k] = TokenCSR(ids, indptr)
    log.info("tokens %s: %s in %.0fs", split, {k: len(v.ids) for k, v in out.items()}, time.time() - t0)
    return out


def tokens_dir(cfg: Config, split: str) -> str:
    return os.path.join(cfg.work_dir, split, "tokens")


def open_tokens(directory: str) -> Dict[str, TokenCSR]:
    return {k: TokenCSR(np.load(os.path.join(directory, f"{k}_ids.npy"), mmap_mode="r"),
                        np.load(os.path.join(directory, f"{k}_ptr.npy"), mmap_mode="r"))
            for k in TOKEN_TYPES}


def run(cfg: Config, split: str, table: RecordTable, force: bool = False) -> Dict[str, TokenCSR]:
    directory = tokens_dir(cfg, split)
    if os.path.exists(os.path.join(directory, "done")) and stamp_ok(directory, tokens_fp()) and not force:
        return open_tokens(directory)
    # Build into a private temp dir, then rename: a crash or a concurrent run can never
    # leave a half-written cache that a later run would trust.
    tmp = f"{directory}.tmp-{os.getpid()}"
    os.makedirs(tmp, exist_ok=True)
    toks = build_tokens(cfg, split, table)
    for k, v in toks.items():
        np.save(os.path.join(tmp, f"{k}_ids.npy"), v.ids)
        np.save(os.path.join(tmp, f"{k}_ptr.npy"), v.indptr)
    del toks
    with open(os.path.join(tmp, "done"), "w") as f:
        f.write("ok")
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.replace(tmp, directory)
    write_stamp(directory, tokens_fp())
    return open_tokens(directory)
