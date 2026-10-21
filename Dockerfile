# Multi-stage Dockerfile for LangGraph Cyclic Agent Engine
FROM python:3.12-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final Runtime Image
FROM python:3.12-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:$PATH"

COPY --from=builder /root/.local /root/.local
COPY pyproject.toml .
COPY src/ src/
COPY tests/ tests/

RUN python -m unittest discover -s tests -p "test_*.py"

EXPOSE 8080

CMD ["python", "-m", "src.learn_langgraph.main"]
