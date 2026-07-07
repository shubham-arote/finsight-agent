# finsight — one lean image, two roles (compose/Cloud Run pick the command):
#   default        the FastAPI app (uvicorn finsight.server:app)
#   mcp sidecar    python -m finsight.mcp_server.main
# Lean by construction: cloud-first, no torch/CUDA/local models.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1

# dependency layer (cached until pyproject/uv.lock change)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# the project itself
COPY src ./src
COPY samples ./samples
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "finsight.server:app", "--host", "0.0.0.0", "--port", "8000"]
