"""Build <team>_submission.zip exactly in the layout the challenge requires.

    <team>_submission.zip
    ├── output/matching_results.tsv
    ├── output/candidate_pairs.tsv
    ├── code/business_entity_resolution/   (src/, README.md, requirements.txt, ...)
    ├── Documentation_template.md
    └── MANIFEST.sha256                    (hash of every file in the archive)

The outputs are taken verbatim from the pipeline's output directory -- never edited --
and are format-checked before anything is written, so the archive cannot drift from
what the code produced. Stdlib only.

Usage (from this folder):  python package_submission.py --team <team_name>
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CODE_EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".venv", ".git"}   # plus every work* cache dir
CODE_EXCLUDE_EXT = {".pyc", ".pyo"}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _check_output(path: str, header: list) -> int:
    """Minimal structural check (the official validator does the full one)."""
    with open(path, encoding="utf-8") as f:
        got = f.readline().rstrip("\n").split("\t")
        if got != header:
            raise SystemExit(f"{path}: header {got} != {header}")
        seen = set()
        n = 0
        for line in f:
            s1, tab, rest = line.rstrip("\n").partition("\t")
            if not tab or s1 in seen:
                raise SystemExit(f"{path}: malformed or duplicate row for {s1!r}")
            seen.add(s1)
            ids = [x for x in rest.split(",") if x]
            if len(ids) != len(set(ids)) or any(not x.startswith(("S2-", "S3-")) for x in ids):
                raise SystemExit(f"{path}: invalid id list for {s1}")
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", required=True, help="team name used in the zip file name")
    ap.add_argument("--output-dir", default=os.path.join(ROOT, "output"))
    ap.add_argument("--doc", default=os.path.join(ROOT, "Documentation_template.md"))
    ap.add_argument("--dest", default=ROOT, help="folder to write the zip into")
    args = ap.parse_args()

    match = os.path.join(args.output_dir, "matching_results.tsv")
    cand = os.path.join(args.output_dir, "candidate_pairs.tsv")
    for p in (match, cand, args.doc):
        if not os.path.isfile(p):
            print(f"missing required file: {p}", file=sys.stderr)
            return 1
    n_match = _check_output(match, ["source1_entity_id", "matched_entity_ids"])
    n_cand = _check_output(cand, ["source1_entity_id", "candidate_entity_ids"])
    if n_match != n_cand:
        print(f"row count mismatch: {n_match} matching vs {n_cand} candidate rows", file=sys.stderr)
        return 1

    entries = [(match, "output/matching_results.tsv"), (cand, "output/candidate_pairs.tsv"),
               (args.doc, "Documentation_template.md")]
    for dirpath, dirnames, filenames in os.walk(HERE):
        dirnames[:] = sorted(d for d in dirnames if d not in CODE_EXCLUDE_DIRS and not d.startswith("work")
                             and not d.endswith(".egg-info"))
        for name in sorted(filenames):
            if os.path.splitext(name)[1] in CODE_EXCLUDE_EXT:
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, HERE).replace(os.sep, "/")
            entries.append((full, f"code/business_entity_resolution/{rel}"))

    zip_path = os.path.join(args.dest, f"{args.team}_submission.zip")
    manifest = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, arc in entries:
            z.write(src, arc)
            manifest.append(f"{_sha256(src)}  {arc}")
        z.writestr("MANIFEST.sha256", "\n".join(manifest) + "\n")
    print(f"wrote {zip_path}: {len(entries)} files, {n_match} S1 rows, "
          f"{os.path.getsize(zip_path) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
