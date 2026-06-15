FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install .

FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache \
    TRADING_RESEARCH_OUTPUT_DIR=/app/outputs \
    TRADING_RESEARCH_CACHE_DIR=/app/outputs/cache

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/outputs \
    && chown -R appuser:appuser /app

COPY --from=builder /opt/venv /opt/venv
COPY README.md .env.example ./
COPY docs ./docs
COPY examples ./examples

USER appuser

VOLUME ["/app/outputs"]

HEALTHCHECK --interval=5m --timeout=10s --start-period=30s --retries=3 \
    CMD trade-research --help >/dev/null || exit 1

ENTRYPOINT ["trade-research"]
CMD ["--help"]
