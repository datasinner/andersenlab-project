# Port Tariff Agent

An agentic RAG service that reads a port's tariff document (PDF) plus a vessel call and calculates
every due the vessel must pay at that port, with the formula and tariff citations behind each
figure. The LLM finds and interprets the rules in the document; a deterministic engine does the
arithmetic. No tariff data is hard-coded.

> **Status:** under construction. Phases 0–1 of [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md) are done
> (skeleton, data model). The architecture is described in
> [docs/architecture.md](docs/architecture.md).

## Quick start

```bash
cp .env.example .env    # set OPENAI_API_KEY (not needed yet for phases 0-1)
docker compose up --build
```

The API runs behind Nginx on `http://localhost:8080`:

- `GET /health`: process is alive
- `GET /ready`: database is reachable, plus the number of ingested tariff documents
- `GET /docs`: Swagger UI, used to upload tariff PDFs and run calculations

`docker compose down -v` removes everything, including the database volume.

### Local development (no Docker for the API)

```bash
uv sync
docker compose up -d db
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
```

`make test` runs the test suite against a throwaway `test` database on the same Postgres server,
with the LLM provider forced to a fake (no API key or network needed). `make lint` runs
`ruff check` and `ruff format --check`.
