# Port Tariff Agent

An agentic RAG service that reads a port's tariff document (PDF) plus a vessel call and calculates
every due the vessel must pay at that port, with the formula and tariff citations behind each
figure. The LLM finds and interprets the rules in the document; a deterministic engine does the
arithmetic. No tariff data is hard-coded.

> **Status:** under construction. Phases 0–7 of [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md) are done
> (skeleton, data model, rule DSL and calculation engine, OpenAI client layer, PDF ingestion,
> retrieval and charge catalogue, rule-compilation agent, end-to-end calculation). The architecture is described in
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
- `POST /v1/calculations`: price a vessel call (profile JSON, plain-language `query`, or both)
- `GET /v1/calculations`, `GET /v1/calculations/{id}`: past calculations, with the agent's trace
- `POST /v1/rules/compile`: have the agent compile a port's rules (cached; minutes when cold)
- `GET /v1/rules?port=Durban`: the compiled rulebook for a port, for review

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

### Compiling rules with the agent

For each charge a vessel routinely pays at a port, a LangGraph agent researches the tariff with
tools (search, read a section, look up a definition, list the outline), extracts the charge as a
typed `ChargeRule`, checks that every number in it is printed in the text it cites, and has a
second model call review it; blocking review issues go back to the extractor. Rules are cached per
document, port and prompt version.

```bash
uv run python scripts/compile_rules.py --port Durban --out-dir build/rulebook/durban --verbose
uv run python scripts/calculate.py --rules build/rulebook/durban \
    --vessel tests/fixtures/vessels/sudestada.json --port Durban --explain
```

`--verbose` prints every step the agent took (sections opened, searches, revisions, review
notes). Compiled from the TNPA PDF, the Durban rulebook prices SUDESTADA exactly like the
hand-written golden rules in `tests/fixtures/rules/durban/`: light dues 60,062.04, port dues
199,371.35, towage 147,074.38, VTS 33,345.00, pilotage 47,189.94, berthing 19,639.50, running of
lines 3,309.12 (ZAR); berth dues not applicable to a cargo-working call. Compiling those eight
charges from scratch took about three minutes and 390k tokens.

### Pricing a vessel call

```bash
curl -s -X POST localhost:8080/v1/calculations -H 'Content-Type: application/json' -d '{
  "query": "Bulk carrier SUDESTADA, 51,300 GT, LOA 229.2 m, calling at Durban from 15 to 22 Nov 2024, 3.39 days alongside loading 40,000 t of iron ore for export."
}'
```

Give the vessel as a profile JSON (`vessel`, in the brief's format), a plain-language `query`, or
both (profile fields win). The response lists every charge with its amount, step-by-step
formula, cited tariff text, the facts decided for the call (from the data, presumed for a call of
this kind, a rule default, or your `overrides`) and assumptions, plus the charges that don't
apply, aren't priced in the document, are only charged on request, or are paid by others.
Swagger (`/docs`) has the SUDESTADA profile as a ready-made example.

`make eval` prices the reference call from the profile JSON and from a plain-language description
and compares both with the reference values (`eval/cases/sudestada_durban.json`; tolerance 1%).

### Accuracy and reliability, honestly

With a reviewed rulebook in the cache, a calculation is deterministic and fast (about 12 s from
profile JSON, one batch of fact-resolution calls), and matches the reference values:

| Reference item | Reference (ZAR) | Computed (ZAR) | Δ |
|---|---|---|---|
| Light dues | 60,062.04 | 60,062.04 | 0 |
| Port dues | 199,549.22 | 199,371.35 | −0.09% (reference used 3.396 days; the profile says 3.39) |
| Towage dues | 147,074.38 | 147,074.38 | 0 |
| VTS dues | 33,315.75 | 33,345.00 | +0.09% (reference used GT 51,255; the profile says 51,300) |
| Pilotage dues | 47,189.94 | 47,189.94 | 0 |
| Running lines (§3.8 berthing services) | 19,639.50 | 19,639.50 | 0 |

The profile JSON and the plain-language query give identical results, and the total (509,991.33,
including §3.9 running of vessel lines, 3,309.12) equals the hand-written golden rulebook's.

Compiling a rulebook from scratch is where the uncertainty is. In four cold compiles of the 23
routine Durban charges with `gpt-5.6-luna`, each run got 2–3 of the complex charges wrong in a
different way: reading "self-propelled vessels, vessels licensed by ..., at their registered
port" as applying to any self-propelled ship (light dues at the per-metre rate), charging drydock
dues on an ordinary call, or leaving a charge unpriced because one of its cases is. Each failure
mode led to a general fix (an `unpriced` component, tier widths, deductions, shared rule
semantics for extractor and critic, a critic that grades issues, keeping the best reviewed
version, citation repair, facts decided with their rule context), but variance remains. In
practice:

- compile once per document and port (`scripts/compile_rules.py --verbose`), review the rulebook
  (`GET /v1/rules?port=...`, flags and open issues included), and recompile single charges with
  `--charge ... --refresh`;
- set `LLM_COMPILE_MODEL` to a stronger model for the offline compile step (`gpt-5.6-sol`
  produced a better-structured light-dues rule in one experiment);
- rules flagged `low_confidence` are priced and marked (`confidence: low`, with open issues).
