.PHONY: up down test lint format migrate logs

up:
	docker compose up --build

down:
	docker compose down

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

migrate:
	uv run alembic upgrade head

logs:
	docker compose logs -f
