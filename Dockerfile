# syntax=docker/dockerfile:1
#
# FRIDAY container image — multi-stage, uv-based.
#
# Stage base: python:3.12-slim is the safe, supported container base. Local dev
# runs on 3.14, but the project targets >=3.12 and 3.12-slim has the broadest
# wheel coverage for our deps, so the image pins 3.12.
#
# Stage 1 (builder): install uv, copy the project, and `uv sync --frozen
# --no-dev` into an in-tree .venv (resolved from the committed uv.lock, prod
# deps only — no pytest/mypy/ruff in the runtime image).
#
# Stage 2 (runtime): copy that .venv + source, drop to a non-root user, expose
# 8000, and run uvicorn against the app factory.
#
# Secrets are NEVER baked in: no .env is copied (see .dockerignore); env is
# supplied at run time (docker compose `env_file: .env` or `-e` flags).

# ---- builder ----------------------------------------------------------------
FROM python:3.12-slim AS builder

# Copy the standalone uv binary from its official image (pinned, no pip needed).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# uv build/runtime knobs:
#  - copy mode so the venv is self-contained and movable between stages
#  - bytecode compile for faster cold starts
#  - don't let uv try to manage/download a different Python; use the base image's
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Resolve dependencies first (better layer caching) using only the lock + manifest.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Now copy the source and install the project itself into the venv.
COPY src ./src
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---- runtime ----------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Put the venv on PATH so `uvicorn`/`python` resolve to the synced environment.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create a non-root user and the writable data dir (SQLite db / volume mount).
RUN useradd --create-home --uid 10001 friday \
    && mkdir -p /app/data \
    && chown -R friday:friday /app

WORKDIR /app

# Bring over the built virtualenv and the application source.
COPY --from=builder --chown=friday:friday /app/.venv /app/.venv
COPY --chown=friday:friday src ./src
COPY --chown=friday:friday pyproject.toml README.md ./

USER friday

EXPOSE 8000

# Serve the FastAPI factory. --factory because friday.app:create_app is a builder,
# not a module-level ASGI app. Bind 0.0.0.0 so the port is reachable from the host.
CMD ["uvicorn", "friday.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
