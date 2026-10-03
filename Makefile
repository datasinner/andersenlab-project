.PHONY: up down test test-unit coverage lint format migrate ingest compile rulebook-export rulebook-import calculate smoke eval eval-retrieval logs

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
	uv run pytest tests/unit --cov=app/rules --cov=app/domain --cov=app/llm --cov=app/ingestion --cov-branch --cov-report=term-missing

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

migrate:
	uv run alembic upgrade head

# Ingest every PDF in data/tariffs (idempotent).
ingest:
	uv run python scripts/ingest.py --dir data/tariffs

# Compile (or reuse) the rulebook for a port: make compile PORT=Durban
compile:
	uv run python scripts/compile_rules.py --port "$(PORT)"

# Write a port's compiled rulebook to data/rulebooks/ (loaded at container start).
rulebook-export:
	uv run python scripts/rulebook.py export --port "$(PORT)"

# Load every rulebook in data/rulebooks/ into the database (idempotent).
rulebook-import:
	uv run python scripts/rulebook.py import

# Price SUDESTADA at Durban with the hand-written golden rules (no LLM, no DB).
calculate:
	uv run python scripts/calculate.py --rules tests/fixtures/rules/durban \
		--vessel tests/fixtures/vessels/sudestada.json --port Durban \
		--facts '{"is_cargo_working": true}' --explain

# One real structured extraction + embedding call (needs OPENAI_API_KEY);
# `uv run python scripts/llm_smoke.py --fake` runs offline.
smoke:
	uv run python scripts/llm_smoke.py

# End-to-end accuracy on the reference vessel calls (real OpenAI; cached rules).
eval:
	uv run python eval/run_eval.py

# Recall@k of hybrid search on the ingested TNPA document (real embeddings).
eval-retrieval:
	uv run python eval/run_retrieval_eval.py

logs:
	docker compose logs -f
