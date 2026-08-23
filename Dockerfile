FROM ghcr.io/astral-sh/uv:0.11.28-python3.13-trixie-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --extra serving

RUN useradd --create-home --uid 10001 radfusion
USER radfusion

EXPOSE 8000

ENTRYPOINT ["/app/.venv/bin/python", "-m", "radfusion.serving.cli"]
