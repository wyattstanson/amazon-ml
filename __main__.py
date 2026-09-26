"""Command-line entry point: `python -m ber <stage> [options]`."""
from __future__ import annotations

import argparse
import logging
import os
import sys

from .config import Config


def _default_paths():
    here = os.path.dirname(os.path.abspath(__file__))              # .../src/ber
    project = os.path.abspath(os.path.join(here, "..", ".."))      # .../business_entity_resolution
    return (os.path.join(project, "..", "..", "student_resource", "dataset"),
            os.path.join(project, "work"),
            os.path.join(project, "..", "..", "output"))


def main(argv=None) -> int:
    data, work, out = _default_paths()
    p = argparse.ArgumentParser(prog="python -m ber", description=__doc__)
    p.add_argument("stage", choices=["all", "train", "test", "preprocess", "experiment-blocking", "experiment-features", "analyze"],
                   help="all = train (fit + evaluate + tune) then test (predict + write outputs)")
    p.add_argument("--sample", type=int, default=150_000, help="S1 sample size for experiments")
    p.add_argument("--tag", default="dev", help="label written to the experiment log")
    p.add_argument("--grid", default="strategies", help="named experiment grid")
    p.add_argument("--data-dir", default=data, help="folder containing train/ and test/")
    p.add_argument("--work-dir", default=work, help="cache for intermediate artefacts")
    p.add_argument("--output-dir", default=out, help="where the submission TSVs are written")
    p.add_argument("--split", choices=["train", "test"], default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--force", action="store_true", help="recompute even if cached")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)
    cfg = Config(data_dir=os.path.abspath(args.data_dir), work_dir=os.path.abspath(args.work_dir),
                 output_dir=os.path.abspath(args.output_dir))
    if args.workers:
        cfg.workers = args.workers

    if args.stage == "preprocess":
        from . import preprocess
        for split in ([args.split] if args.split else ["train", "test"]):
            preprocess.run(cfg, split, force=args.force)
    elif args.stage == "experiment-features":
        from . import experiments
        print(experiments.feature_ablation(cfg))
    elif args.stage == "analyze":
        from . import analysis
        print("wrote", analysis.run(cfg))
    elif args.stage == "experiment-blocking":
        from . import experiments
        experiments.blocking_grid(cfg, experiments.blocking_grids()[args.grid], args.sample, args.grid)
    else:
        from . import pipeline
        if args.stage in ("all", "train"):
            pipeline.run_train(cfg, force=args.force)
        if args.stage in ("all", "test"):
            pipeline.run_test(cfg, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
