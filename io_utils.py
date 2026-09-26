"""Strict TSV input/output (stdlib only).

Inputs are read line by line with an explicit TAB split: no quote interpretation
(a few hundred names contain a literal `"`), no NA coercion (a business called
"NA" stays "NA"), and every row must have exactly the four documented columns.
"""
from __future__ import annotations

from typing import Iterator, List

SOURCE_HEADER = ["entity_id", "business_name", "business_address", "country"]


def iter_source_lines(path: str, chunk: int) -> Iterator[List[str]]:
    """Yield lists of raw data lines (header validated and skipped)."""
    with open(path, encoding="utf-8", newline="") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        if header != SOURCE_HEADER:
            raise ValueError(f"{path}: unexpected header {header}, expected {SOURCE_HEADER}")
        buf: List[str] = []
        for line in f:
            buf.append(line)
            if len(buf) >= chunk:
                yield buf
                buf = []
        if buf:
            yield buf


def id_key(entity_id: str) -> int:
    """'S2-123456' -> 2 * 10**12 + 123456: a unique int64 per entity id, used to map ids to
    table rows with a sorted-array lookup instead of a 12M-entry dict."""
    e = entity_id
    if len(e) < 4 or e[0] != "S" or e[2] != "-" or not e[1].isdigit() or not e[3:].isdigit() \
            or len(e) - 3 > 12:
        raise ValueError(f"unexpected entity id format: {e!r}")
    return int(e[1]) * 10**12 + int(e[3:])


def parse_line(line: str) -> List[str]:
    parts = line.rstrip("\r\n").split("\t")
    if len(parts) != 4:
        raise ValueError(f"expected 4 tab-separated fields, got {len(parts)}: {line[:200]!r}")
    return parts
