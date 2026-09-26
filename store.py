"""Compact, memory-mapped storage for millions of normalised records.

A Python `str` costs ~50 bytes of object overhead, so 12M records x 9 text fields as
ordinary lists would need ~7 GB. `StrCol` keeps one concatenated UTF-8 buffer plus an
offsets array per column (the Arrow layout, without pyarrow -- which Windows
Application Control blocks on the development machine). On disk each column is a raw
`.bin` buffer and an `.off.npy` offsets file; both are memory-mapped on load, so the
OS pages in only what a processing chunk touches instead of committing gigabytes.
"""
from __future__ import annotations

import json
import mmap
import os
from typing import Dict, Iterable, List, Sequence

import numpy as np


class StrCol:
    """Immutable column of strings backed by one bytes-like buffer + int64 offsets."""

    __slots__ = ("buf", "off", "_keep")

    def __init__(self, buf, off: np.ndarray, keep=None):
        self.buf = buf          # bytes or mmap.mmap (both slice to bytes)
        self.off = off
        self._keep = keep       # open file object backing an mmap

    @classmethod
    def from_list(cls, strings: Sequence[str]) -> "StrCol":
        enc = [s.encode("utf-8") for s in strings]
        off = np.zeros(len(enc) + 1, dtype=np.int64)
        np.cumsum(np.fromiter((len(b) for b in enc), dtype=np.int64, count=len(enc)), out=off[1:])
        return cls(b"".join(enc), off)

    @classmethod
    def concat(cls, cols: Iterable["StrCol"]) -> "StrCol":
        bufs, offs, base = [], [np.zeros(1, dtype=np.int64)], 0
        for c in cols:
            bufs.append(bytes(c.buf))
            offs.append(np.asarray(c.off[1:]) + base)
            base += len(c.buf)
        return cls(b"".join(bufs), np.concatenate(offs))

    def __len__(self) -> int:
        return len(self.off) - 1

    def __getitem__(self, i: int) -> str:
        return self.buf[int(self.off[i]):int(self.off[i + 1])].decode("utf-8")

    def take(self, idx) -> List[str]:
        """Decode rows `idx` (any integer array) into a list of str."""
        idx = np.asarray(idx, dtype=np.int64)
        starts = self.off[idx].tolist()
        ends = self.off[idx + 1].tolist()
        buf = self.buf
        return [buf[s:e].decode("utf-8") for s, e in zip(starts, ends)]

    def to_list(self) -> List[str]:
        return self.take(np.arange(len(self)))

    def save(self, prefix: str) -> None:
        with open(prefix + ".bin", "wb") as f:
            f.write(self.buf)
        np.save(prefix + ".off.npy", np.asarray(self.off))

    @classmethod
    def open(cls, prefix: str) -> "StrCol":
        off = np.load(prefix + ".off.npy", mmap_mode="r")
        f = open(prefix + ".bin", "rb")
        if os.path.getsize(prefix + ".bin") == 0:
            f.close()
            return cls(b"", off)
        return cls(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ), off, keep=f)


class RecordTable:
    """Columns of normalised records for one split (all sources, all countries).

    Text columns are `StrCol`; numeric/flag columns are numpy arrays. Row order is
    source file order: all Source 1, then Source 2, then Source 3.
    """

    TEXT = ("entity_id", "name_full", "name_core", "name_alias", "legal",
            "addr_norm", "state", "nums", "codes")
    NUMERIC = ("source", "country", "is_domain", "is_indic", "addr_empty", "key")

    def __init__(self, text: Dict[str, StrCol], numeric: Dict[str, np.ndarray],
                 countries: List[str]):
        self.text = text
        self.numeric = numeric
        self.countries = countries          # country label for each country code

    def __len__(self) -> int:
        return len(self.numeric["source"])

    def __getattr__(self, name):
        text = self.__dict__.get("text", {})
        if name in text:
            return text[name]
        numeric = self.__dict__.get("numeric", {})
        if name in numeric:
            return numeric[name]
        raise AttributeError(name)

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        for k, v in self.text.items():
            v.save(os.path.join(directory, k))
        for k, v in self.numeric.items():
            np.save(os.path.join(directory, k + ".npy"), v)
        with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"countries": self.countries, "n": len(self)}, f)

    @classmethod
    def open(cls, directory: str) -> "RecordTable":
        with open(os.path.join(directory, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        text = {k: StrCol.open(os.path.join(directory, k)) for k in cls.TEXT}
        numeric = {k: np.load(os.path.join(directory, k + ".npy")) for k in cls.NUMERIC}
        return cls(text, numeric, meta["countries"])

    @staticmethod
    def exists(directory: str) -> bool:
        return os.path.exists(os.path.join(directory, "meta.json"))
