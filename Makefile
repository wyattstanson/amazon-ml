# Convenience targets (Linux/macOS, or Windows with GNU make). Everything is also available
# without make:  python reproduce.py --data-dir DATA --team TEAM   (see README.md).
#
#   make setup      create .venv and install pinned requirements
#   make test       unit tests
#   make run        full pipeline: preprocess -> block -> features -> train/evaluate -> predict -> write
#   make validate   official validator on the produced outputs
#   make package    build <TEAM>_submission.zip (outputs + code + methodology document)
#   make all        setup + (tests, pipeline, validator, package) via reproduce.py
#
# DATA must point at the folder containing train/ and test/.
DATA      ?= ../../student_resource/dataset
OUT       ?= ../../output
WORK      ?= work
TEAM      ?= team
VALIDATOR ?= ../../student_resource/utils/validate_submission.py
WORKERS   ?= 6

ifeq ($(OS),Windows_NT)
  PY := .venv/Scripts/python
else
  PY := .venv/bin/python
endif
RUN := PYTHONPATH=src PYTHONIOENCODING=utf-8 $(PY) -m ber

.PHONY: setup test run validate package all clean

setup:
	python -m venv .venv
	$(PY) -m pip install --disable-pip-version-check -r requirements.txt

test:
	PYTHONPATH=src $(PY) -m pytest -q tests

run:
	$(RUN) all --data-dir $(DATA) --work-dir $(WORK) --output-dir $(OUT) --workers $(WORKERS)

validate:
	$(PY) $(VALIDATOR) --matching $(OUT)/matching_results.tsv --candidate $(OUT)/candidate_pairs.tsv --test-dir $(DATA)/test --check-ids

package:
	$(PY) package_submission.py --team $(TEAM) --output-dir $(OUT)

all: setup
	$(PY) reproduce.py --data-dir $(DATA) --team $(TEAM) --validator $(VALIDATOR) --workers $(WORKERS) \
	    --output-dir $(OUT) --work-dir $(WORK)

clean:
	rm -rf $(WORK)
