.PHONY: up down test test-unit coverage lint format migrate calculate logs

up:
	docker compose up --build

down:
	docker compose down

test:
	uv run pytest

# Unit tests only: no database needed.
test-unit:
	uv run pytest tests/unit

coverage:
	uv run pytest tests/unit --cov=app/rules --cov=app/domain --cov-branch --cov-report=term-missing

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

migrate:
	uv run alembic upgrade head

# Price SUDESTADA at Durban with the hand-written golden rules (no LLM, no DB).
calculate:
	uv run python scripts/calculate.py --rules tests/fixtures/rules/durban \
		--vessel tests/fixtures/vessels/sudestada.json --port Durban \
		--facts '{"is_cargo_working": true}' --explain

logs:
	docker compose logs -f
