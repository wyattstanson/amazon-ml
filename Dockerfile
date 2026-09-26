# Pinned runtime for reproducing the submission. Network is only needed at build time
# (pip install); the pipeline itself is run with `--network none`.
#
#   docker build -t ber .
#   docker run --rm --network none \
#       -v /path/to/dataset:/data:ro -v "$PWD/../../output":/out -v ber-work:/work \
#       ber
FROM python:3.11.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PYTHONPATH=/app/src
# libgomp1: OpenMP runtime required by the LightGBM wheel
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --disable-pip-version-check -r requirements.txt
COPY src ./src
COPY tests ./tests

ENTRYPOINT ["python", "-m", "ber"]
CMD ["all", "--data-dir", "/data", "--work-dir", "/work", "--output-dir", "/out"]
