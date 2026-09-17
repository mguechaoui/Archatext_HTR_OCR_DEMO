# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Stage 1 — builder: compile wheels into a self-contained virtualenv.
# Nothing from this stage ships except /opt/venv, so build toolchains
# (gcc, headers) never end up in the runtime image.
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        git \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

# CPU-only torch FIRST, from PyTorch's CPU index. The default PyPI wheel
# bundles CUDA and pulls ~2.5GB of nvidia-* packages we can never use on a
# CPU host — this single step is the difference between a ~1GB image and a
# ~4GB one.
RUN pip install --no-cache-dir \
        torch==2.4.1 torchvision==0.19.1 \
        --index-url https://download.pytorch.org/whl/cpu

# Constraints pin torch so that kraken/transformers cannot silently pull the
# CUDA build back in as a transitive dependency.
COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    HF_HOME=/tmp/hf \
    OMP_NUM_THREADS=1 \
    OCR_MODEL_DIR=/models \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
    

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libxml2 \
        libxslt1.1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /srv/app

COPY app ./app
COPY static ./static
COPY scripts ./scripts
COPY models.manifest.json ./models.manifest.json
COPY entrypoint.sh ./entrypoint.sh

RUN chmod +x entrypoint.sh \
    && useradd --create-home --uid 10001 ocr \
    && mkdir -p /models /srv/app/uploads/images /srv/app/uploads/temp /srv/app/outputs \
    && chown -R ocr:ocr /models /srv/app /tmp

USER ocr

EXPOSE 8000

# The readiness probe reports *which models actually loaded*, not just that
# the process is alive — a container that booted with a missing checkpoint is
# not ready to serve, and the orchestrator should know that.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -fsS http://localhost:${PORT:-8000}/health/ready || exit 1

ENTRYPOINT ["./entrypoint.sh"]
