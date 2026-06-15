install: ; uv sync --all-groups
install-voice: ; uv pip install -r requirements-voice.txt
install-perception: ; uv pip install -r requirements-perception.txt
install-dashboard: ; uv pip install -r requirements-dashboard.txt
dashboard: ; uv run streamlit run dashboard/app.py
test: ; uv run pytest -q
lint: ; uv run ruff check src tests
fmt: ; uv run ruff format src tests
type: ; uv run mypy
run: ; uv run uvicorn friday.app:create_app --factory --reload
docker-build: ; docker build -t friday .
docker-up: ; docker compose up -d
docker-down: ; docker compose down
gate-0: lint type ; uv run pytest -q tests/unit
gate-1: lint type ; uv run pytest -q
gate-2: lint type ; uv run pytest -q
gate-3: lint type ; uv run pytest -q
gate-4: lint type ; uv run pytest -q
gate-5: lint type ; uv run pytest -q
gate-6: lint type ; uv run pytest -q
gate-7: lint type ; uv run pytest -q
.PHONY: install install-voice install-perception install-dashboard dashboard test lint fmt type run docker-build docker-up docker-down gate-0 gate-1 gate-2 gate-3 gate-4 gate-5 gate-6 gate-7
