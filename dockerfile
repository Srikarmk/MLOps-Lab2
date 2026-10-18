FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so this layer is cached between code changes
COPY src/requirements.txt .
RUN pip install --root-user-action=ignore -r requirements.txt

COPY src/ .

ENV OUTPUT_DIR=/app/output

CMD ["python", "main.py"]
