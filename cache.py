"""Content-addressed cache fingerprints.

Every cached artefact under work/ carries a fingerprint of (a) the source code of the
modules that produced it and (b) its settings, chained to its upstream artefact's
fingerprint. Changing a normalisation rule therefore invalidates records, tokens,
candidates, features and models -- and nothing else -- while an unchanged rerun reuses
everything. Nothing depends on file timestamps.
"""
from __future__ import annotations

import hashlib
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def code_fp(*module_names: str) -> str:
    h = hashlib.sha256()
    for name in module_names:
        with open(os.path.join(_HERE, f"{name}.py"), "rb") as f:
            h.update(name.encode() + b"\0" + f.read())
    return h.hexdigest()[:16]


def fingerprint(**parts) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def stamp_ok(path: str, fp: str) -> bool:
    stamp = path.rstrip("/\\") + ".stamp"
    if not (os.path.exists(path) and os.path.exists(stamp)):
        return False
    with open(stamp, encoding="utf-8") as f:
        return f.read().strip() == fp


def write_stamp(path: str, fp: str) -> None:
    with open(path.rstrip("/\\") + ".stamp", "w", encoding="utf-8") as f:
        f.write(fp)


def preprocess_fp() -> str:
    return fingerprint(code=code_fp("translit", "normalize", "io_utils", "store", "preprocess"))


def tokens_fp() -> str:
    return fingerprint(upstream=preprocess_fp(), code=code_fp("tokens"))
