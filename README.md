# Business Entity Resolution — Amazon ML Challenge 2026

For every Source-1 business, find all Source-2 / Source-3 records describing the same
real-world business, using only name + address. The pipeline is **blocking → pair
features → LightGBM matcher → F0.5-tuned decision layer**, built to run on a laptop CPU
over ~12M records per split, fully offline, with deterministic output.

Methodology, numbers and error analysis: `../../Documentation_template.md`.
Every design decision and the measurement behind it: `EXPERIMENTS.md`.

## Layout

```
code/business_entity_resolution/
├── src/ber/
│   ├── __main__.py      CLI: python -m ber {all,train,test,preprocess,analyze,experiment-blocking,experiment-features}
│   ├── config.py        paths, seed
│   ├── cache.py         content fingerprints (module source + settings) for every cached stage
│   ├── io_utils.py      strict TSV reading (explicit TAB, header check, exactly 4 fields, no NA coercion)
│   ├── translit.py      Indic-script -> Latin transliteration (9 Brahmic scripts, no external data)
│   ├── normalize.py     name/address normalisation (legal forms, street types, states, leetspeak,
│   │                    aliases, word segmentation of concatenated names)
│   ├── store.py         memory-mapped columnar record store
│   ├── preprocess.py    normalise every record once (multi-process)
│   ├── tokens.py        typed blocking tokens (incl. cross-field conjunctions), CRC32-hashed, CSR
│   ├── blocking.py      key blocking (strategy A) and IDF meta-blocking (strategy B, used)
│   ├── features.py      54 pair features (similarities, IDF overlaps, numbers/codes, context, support)
│   ├── stage2.py        27 candidate-graph features from out-of-fold stage-1 probabilities
│   │                    (competition, probability-weighted cluster support, neighbour similarity)
│   ├── rules.py         matcher 1: rule-based baseline
│   ├── model.py         matcher 2: LightGBM (stage 1 and stage 2, used)
│   ├── decide.py        exclusivity + threshold / expected-F0.5 set selection
│   ├── evaluate.py      exact macro-F0.5 metric, blocking recall, oracle ceiling
│   ├── pipeline.py      orchestration with fingerprinted caches
│   ├── analysis.py      error analysis report (work/error_analysis.md)
│   └── experiments.py   reproducible blocking grids and feature ablation
├── tests/               unit + synthetic end-to-end tests (pytest, 35 tests)
├── results/             raw logs: blocking grids, matcher experiments, decision tuning, metrics, error analyses
├── tools/               matcher_experiments.py: the held-out matcher experiments of EXPERIMENTS.md 7.2
├── EXPERIMENTS.md       running log of what was tried and what it scored
├── requirements.txt     pinned dependencies
├── reproduce.py         one command: tests -> pipeline -> official validator -> zip (stdlib only)
├── Makefile             setup / test / run / validate / package / all (wraps reproduce.py)
├── Dockerfile           pinned runtime (run with --network none)
└── package_submission.py builds <team>_submission.zip from the produced outputs
```

## Requirements

* Python 3.10 – 3.13 (developed on 3.13, Docker image uses 3.11), 4+ CPU cores.
* RAM: 32 GB recommended. Feature arrays are memory-mapped and processed in bounded chunks,
  but each LightGBM fit holds its training matrix (~32M pairs x 81 features) in memory:
  the measured peak of the training process was 24 GB.
* Disk: ~85 GB free for the stage caches in `work/` (79 GB after a full run: 119M training
  and 93M test candidate pairs with their feature matrices), plus ~1.3 GB for the outputs.
* The only network use is `pip install` during setup. The pipeline itself makes no network
  calls and downloads nothing; it reads only the provided dataset files.

## Run

From `code/business_entity_resolution/`, with `DATA` pointing at the folder that
contains `train/` and `test/`:

```bash
make setup
make test
make run DATA=/path/to/dataset
make validate DATA=/path/to/dataset VALIDATOR=/path/to/utils/validate_submission.py
make package TEAM=<team_name>
```

Without `make` (e.g. plain Windows):

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt          # .venv/bin/python on Linux/macOS
PYTHONPATH=src .venv/Scripts/python -m ber all --data-dir /path/to/dataset --work-dir work --output-dir ../../output
python /path/to/utils/validate_submission.py --matching ../../output/matching_results.tsv --candidate ../../output/candidate_pairs.tsv --test-dir /path/to/dataset/test
.venv/Scripts/python package_submission.py --team <team_name>
```

`python -m ber all` runs, in order: preprocess → tokens → blocking → features → rule
baseline + LightGBM cross-fitting → decision tuning (all on train, reporting metrics to
`work/metrics_train.json`) → final model → the same stages on test → writes
`output/matching_results.tsv` and `output/candidate_pairs.tsv`.

Each stage caches its artefact in `work/` with a fingerprint of the settings that
produced it, so an interrupted run resumes where it stopped and a changed setting
recomputes only what depends on it. `--force` recomputes everything.

### Docker

```bash
docker build -t ber .
docker run --rm --network none -v /path/to/dataset:/data:ro -v "$PWD/../../output":/out -v ber-work:/work ber
```

## Determinism

Seed `2026` (config.py) drives the S1 fold split and training-row sampling; LightGBM
runs with `deterministic=True`, `force_row_wise=True` and a fixed thread count; blocking
ties are broken by record index; token hashing uses CRC32 (not Python's salted `hash`).
Re-running produces byte-identical output files.

## Reproduce the leaderboard submission from scratch

From an unpacked submission zip (or this folder), with the official `student_resource`
folder somewhere on disk:

```bash
cd code/business_entity_resolution
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # .venv/bin/python on Linux/macOS
.venv/Scripts/python reproduce.py --data-dir /path/to/student_resource/dataset --team <team_name> --validator /path/to/student_resource/utils/validate_submission.py
```

`reproduce.py` stops at the first failing step. It runs:

1. the unit tests;
2. `python -m ber all`: train preprocessing → tokens → blocking → features → rule baseline
   → stage-1 and stage-2 cross-fitting → decision tuning → final models, then the same
   stages on test → `../../output/matching_results.tsv` and `candidate_pairs.tsv`;
3. the official validator with `--check-ids`;
4. `package_submission.py`, which writes `../../<team_name>_submission.zip`.

Expected results (`work/metrics_train.json`, out-of-fold on all 2,206,821 training S1):
blocking pair recall 0.9782, oracle F0.5 0.9926; rule baseline 0.8525; LightGBM stage 1
0.9771; **stage 2 0.9820** (the submitted model). The submitted outputs have these SHA-256
hashes:

```
52a48b778fefb58dd5d6a6c6781550e14b4a74fd9c54df2e9a37decddd98c1bc  matching_results.tsv
dc9d3ca9d2445e1ab75d79905c24a9d9f4a3d8801eaf6c4a27bd55278e09112e  candidate_pairs.tsv
```

## Runtime (reference laptop: Intel Core Ultra 7 258V, 8 cores, 32 GB RAM, Windows 11, CPU only)

Measured on the final run, about 8 hours end to end (peak memory 24 GB):

| stage | train | test |
|---|---|---|
| preprocessing + blocking tokens | ~11 min | ~34 min |
| blocking (IDF meta-blocking, 20 + 8 per source) | 65 min | 54 min |
| pair features | ~40 min | ~34 min |
| rule baseline + decision tuning (3 matchers) | ~24 min | – |
| stage-1 cross-fitting (3 folds, ~32M pairs each) | 78 min | – |
| stage-2 features (competition, cluster support, neighbours) | 16 min | 14 min |
| stage-2 cross-fitting (3 folds) | 53 min | – |
| final models | 29 min | – |
| prediction + writing outputs | – | ~30 min |

Some stages ran alongside other work on the same laptop, so a dedicated machine should be
somewhat faster. Every stage is cached, so an interrupted run resumes where it stopped.
