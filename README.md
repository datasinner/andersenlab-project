# Port Tariff Agent

An agentic RAG service that reads a port's tariff document (PDF) plus a vessel call and calculates
every due the vessel must pay at that port, with the formula and tariff citations behind each
figure. The LLM finds and interprets the rules in the document; a deterministic engine does the
arithmetic. No tariff data is hard-coded.

> **Status:** under construction. Phases 0–5 of [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md) are done
> (skeleton, data model, rule DSL and calculation engine, OpenAI client layer, PDF ingestion,
> retrieval and charge catalogue). The architecture is described in
> [docs/architecture.md](docs/architecture.md).

## Quick start

```bash
cp .env.example .env    # set OPENAI_API_KEY, or LLM_PROVIDER=fake to run offline
docker compose up --build
```

On start the container runs migrations, then ingests every PDF in `data/tariffs/` (the bundled
TNPA Tariff Book 2024/25; already-ingested files are skipped), then serves the API behind Nginx on
`http://localhost:8080`:

- `GET /health`: process is alive
- `GET /ready`: database is reachable, plus the number of ingested (`ready`) tariff documents
- `GET /docs`: Swagger UI, used to upload tariff PDFs and run calculations
- `GET /v1/documents`, `GET /v1/documents/{id}`: ingested tariff documents and their profile
- `GET /v1/documents/{id}/charges`: the charges the document defines (the charge catalogue)
- `GET /v1/documents/{id}/sections/{ref}`: one section's text, e.g. `3.6`

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

### Calculation engine (no LLM, no database)

`make calculate` prices the SUDESTADA validation call at Durban using hand-written rules in
`tests/fixtures/rules/durban/`, printing each charge with its formula and assumptions. Rules are
JSON in the rule DSL (`app/rules/dsl.py`); in later phases the agent extracts them from the tariff
PDF instead.

```bash
uv run python scripts/calculate.py --rules <rules dir> --vessel <profile.json> \
    --port <port> --facts '{"is_cargo_working": true}' --explain   # or --json
```

### LLM smoke test

`make smoke` makes one real Structured Outputs call (extracting a rule from an invented tariff
excerpt, then pricing it with the engine) and one embedding call, using `OPENAI_API_KEY` from
`.env`. `uv run python scripts/llm_smoke.py --fake` does the same offline with the fake clients.

### Ingesting tariff documents

`make ingest` (or `uv run python scripts/ingest.py --file <pdf>`) parses a tariff PDF into a
section tree and retrieval chunks (tables kept whole, running headers and page numbers removed,
two-up landscape sheets split into their printed pages), embeds the chunks, and reads the
document's profile (issuer, ports, currency, VAT, validity period) and its charge catalogue (every
charge the document defines, with payer and trigger) with two concurrent LLM calls. Ingestion is
idempotent by file checksum; a failed attempt is recorded on the document and retried on the next
run, and `--force` rebuilds a document that is already ingested.

`make eval-retrieval` measures recall@5 of the hybrid search (pgvector + Postgres full-text, fused
with reciprocal rank fusion) on 17 queries against the TNPA document; it currently finds the right
section for 16 of them (0.94).
