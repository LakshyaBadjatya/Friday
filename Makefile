install: ; uv sync --all-groups
install-voice: ; uv pip install -r requirements-voice.txt
test: ; uv run pytest -q
lint: ; uv run ruff check src tests
fmt: ; uv run ruff format src tests
type: ; uv run mypy
run: ; uv run uvicorn friday.app:create_app --factory --reload
gate-0: lint type ; uv run pytest -q tests/unit
gate-1: lint type ; uv run pytest -q
gate-2: lint type ; uv run pytest -q
gate-3: lint type ; uv run pytest -q
gate-4: lint type ; uv run pytest -q
.PHONY: install install-voice test lint fmt type run gate-0 gate-1 gate-2 gate-3 gate-4
