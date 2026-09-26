"""One command, any OS: tests -> full pipeline -> official validator -> submission zip.

    python reproduce.py --data-dir /path/to/dataset --team <team_name> \
        [--validator /path/to/utils/validate_submission.py] [--workers 6]

Runs with the current interpreter (create the environment first:
`python -m venv .venv` + `pip install -r requirements.txt`, then run this file with the
venv's python). Stdlib only; every step's exit code is checked and the script stops at
the first failure. Makes no network calls.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def step(title: str, cmd: list, env: dict) -> None:
    print(f"\n=== {title} ===\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=HERE, env=env)
    if rc != 0:
        sys.exit(f"step failed ({title}), exit code {rc}")
    print(f"=== {title}: ok in {time.time() - t0:.0f}s ===", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="folder containing train/ and test/")
    ap.add_argument("--team", required=True, help="team name for <team>_submission.zip")
    ap.add_argument("--output-dir", default=os.path.join(ROOT, "output"))
    ap.add_argument("--work-dir", default=os.path.join(HERE, "work"))
    ap.add_argument("--validator", default=None, help="path to the official validate_submission.py")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    env = dict(os.environ, PYTHONPATH=os.path.join(HERE, "src"), PYTHONIOENCODING="utf-8")
    py = sys.executable
    data = os.path.abspath(args.data_dir)
    if not args.skip_tests:
        step("unit tests", [py, "-m", "pytest", "-q", "tests"], env)
    step("pipeline (train + test)", [py, "-m", "ber", "all", "--data-dir", data, "--work-dir", args.work_dir,
                                     "--output-dir", args.output_dir, "--workers", str(args.workers)], env)
    validator = args.validator or os.path.join(ROOT, "student_resource", "utils", "validate_submission.py")
    if os.path.isfile(validator):
        step("official validator", [py, validator, "--matching", os.path.join(args.output_dir, "matching_results.tsv"),
                                    "--candidate", os.path.join(args.output_dir, "candidate_pairs.tsv"),
                                    "--test-dir", os.path.join(data, "test"), "--check-ids"], env)
    else:
        print(f"\n(official validator not found at {validator}; pass --validator to run it)")
    step("package", [py, "package_submission.py", "--team", args.team, "--output-dir", args.output_dir], env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
