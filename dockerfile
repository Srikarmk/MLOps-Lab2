ARG PYTHON_VERSION=3.10

# ---- Stage 1: install dependencies into a virtual environment ----
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY src/requirements.txt .
# pip/setuptools are only needed to install, so drop them before copying the venv
RUN pip install -r requirements.txt \
    && pip uninstall -y pip setuptools

# ---- Stage 2: runtime image with only the venv and the app code ----
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="iris-trainer" \
      org.opencontainers.image.description="Trains and evaluates a Random Forest on the Iris dataset" \
      org.opencontainers.image.authors="Srikarmk" \
      org.opencontainers.image.source="https://github.com/Srikarmk/MLOps-Lab2"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    OUTPUT_DIR=/app/output

RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=appuser:appuser src/ .
RUN mkdir -p "$OUTPUT_DIR" && chown appuser:appuser "$OUTPUT_DIR"

USER appuser

VOLUME ["/app/output"]

CMD ["python", "main.py"]
