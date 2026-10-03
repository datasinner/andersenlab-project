# Port Tariff Agent

An agentic RAG service that reads a port's tariff document (PDF) and a vessel call, given as a
profile JSON or in plain language, and calculates every due the vessel must pay at that port. Each
figure comes with its step-by-step formula, the tariff text it rests on, the facts decided for the
call and the assumptions made.

The guiding rule is **the LLM reads, the code calculates**. An agent finds each charge in the
document and writes it down as a typed, cited rule; a deterministic `Decimal` engine turns rules
into money. No port, charge or rate is hard-coded: the same code prices the bundled TNPA Tariff
Book (South Africa) and an invented port whose tariff shares nothing with it but the subject.

- [Results](#results)
- [Quick start](#quick-start)
- [Using the API](#using-the-api)
- [Command-line tools](#command-line-tools)
- [How it works](#how-it-works)
- [Configuration](#configuration)
- [Design decisions](#design-decisions)
- [Limitations](#limitations)
- [Project layout](#project-layout)

## Results

**SUDESTADA at Durban** (the reference call; `eval/cases/sudestada_durban.json`), priced from the
compiled TNPA rulebook:

| Reference item | Reference (ZAR) | Computed (ZAR) | Δ |
|---|---|---|---|
| Light dues | 60,062.04 | 60,062.04 | 0 |
| Port dues | 199,549.22 | 199,371.35 | −0.09%: the reference used 3.396 days; the profile says 3.39 |
| Towage dues | 147,074.38 | 147,074.38 | 0 |
| VTS dues | 33,315.75 | 33,345.00 | +0.09%: the reference used GT 51,255; the profile says 51,300 |
| Pilotage dues | 47,189.94 | 47,189.94 | 0 |
| Running lines (§3.8 berthing services) | 19,639.50 | 19,639.50 | 0 |

The profile JSON and the plain-language query give identical results. The system also charges
§3.9 running of vessel lines (3,309.12), for a total of 509,991.33 ZAR (VAT not included). It
lists the charges that don't apply to this call (berth dues for a cargo-working vessel, drydock
dues, and others), the SAMSA levy as not priced in the document (its rates are set by separate
regulations), services only charged on request, and charges paid by cargo owners.

**Another port, no code change.** `scripts/make_synthetic_tariff.py` renders a tariff for an
invented port, Exampleville (`data/synthetic/`): one column, unnumbered headings, euros, harbour
dues per 100 NT with a per-day charge beyond a free period and a minimum, exclusive and stackable
reductions, pilotage by length bands, towage that joins a tug-allocation table to a
charge-per-tugs table, berth dues in marginal length tiers, a per-GT levy with a cap, an
exemption, and charges paid by others. Two vessels, worked out by hand, are priced exactly from
both input forms:

| Case | Expected | Computed |
|---|---|---|
| NORDIC TERN, 172.5 m general cargo, 6.4 days (`eval/cases/exampleville_nordic_tern.json`) | 13,688.80 EUR | 13,688.80 EUR |
| MORVEN, 78 m coaster: hits the harbour-dues minimum, too short for compulsory towage (`eval/cases/exampleville_morven.json`) | 3,264.20 EUR | 3,264.20 EUR |

NORDIC TERN passed with `app/` untouched since the TNPA work. MORVEN then exposed two gaps, each
fixed in general terms: numbers written as words ("beyond the fifth day") now count as printed
for the grounding check, and an `any_of` condition expresses "compulsory for some vessels,
otherwise only on request". `make eval` runs all three cases; `tests/test_no_hardcoding.py`
fails if any name or figure from either tariff appears under `app/`.

## Quick start

Requirements: Docker with Compose, and an OpenAI API key with credit.

```bash
cp .env.example .env    # set OPENAI_API_KEY; LLM_COMPILE_MODEL=gpt-5.6-sol is recommended
docker compose up --build
```

On start the API container:

1. applies database migrations;
2. ingests every PDF in `data/tariffs/` (the bundled TNPA Tariff Book 2024/25): parsing,
   embeddings, and two LLM calls that read the document profile and its charge catalogue. Files
   already ingested are skipped;
3. loads the rulebook files in `data/rulebooks/`: the reviewed, compiled Durban rulebook, so
   Durban calls are priced at once instead of after a 6-minute, 1.2M-token compile;
4. serves the API behind Nginx on `http://localhost:8080` (Swagger UI at `/docs`, with the
   SUDESTADA request as a ready-made example).

```bash
curl -s localhost:8080/ready      # {"status":"ready","documents_ready":1}
curl -s -X POST localhost:8080/v1/calculations -H 'Content-Type: application/json' \
  -d '{"query": "Bulk carrier SUDESTADA, 51,300 GT, LOA 229.2 m, calling at Durban from 15 to 22 Nov 2024, 3.39 days alongside loading 40,000 t of iron ore for export."}'
```

A calculation with a compiled rulebook takes 10–20 s, most of it one round of fact-resolution
calls. Other ports in the TNPA book have no rulebook yet: compile one first (see
[Compiling rules](#compiling-and-reviewing-rules)), because the first compile takes longer than a
calculation may run. `docker compose down -v` removes everything, including the database.

`LLM_PROVIDER=fake` starts the stack without an API key (Swagger, health checks), but nothing can
be ingested or priced: an unscripted fake call fails like an unavailable provider.

### Local development

```bash
uv sync
docker compose up -d db          # Postgres 16 + pgvector, port 5432 published by compose.override.yaml
uv run alembic upgrade head
uv run python scripts/ingest.py --dir data/tariffs
uv run python scripts/rulebook.py import
uv run uvicorn app.main:app --reload
```

`make test` runs the 328 tests (unit and integration) against a throwaway `test` database on the
same Postgres server, with the LLM provider forced to a fake: no API key or network needed.
`make lint` runs `ruff check` and `ruff format --check`.

## Using the API

| Endpoint | Purpose |
|---|---|
| `POST /v1/calculations` | Price a vessel call: profile JSON (`vessel`), plain-language `query`, or both |
| `GET /v1/calculations`, `GET /v1/calculations/{id}` | Past calculations, each with the agent's full trace |
| `POST /v1/documents` | Upload a tariff PDF; ingested in the background |
| `GET /v1/documents`, `GET /v1/documents/{id}` | Ingested documents with their profile (ports, currency, VAT, validity) |
| `GET /v1/documents/{id}/charges` | The charges the document defines (its charge catalogue) |
| `GET /v1/documents/{id}/sections/{ref}` | One section's text, e.g. `3.6` |
| `POST /v1/rules/compile` | Compile (or return cached) rules for a port |
| `GET /v1/rules?port=Durban` | The compiled rulebook for a port, with review flags and open issues |
| `GET /health`, `GET /ready` | Liveness; database reachable and number of ready documents |

### Pricing a vessel call

From a profile in the brief's format (profile fields win over anything in a query):

```bash
curl -s -X POST localhost:8080/v1/calculations -H 'Content-Type: application/json' -d '{
  "port": "Durban",
  "vessel": {
    "vessel_metadata": {"name": "SUDESTADA", "built_year": 2010, "flag": "MLT - Malta"},
    "technical_specs": {"type": "Bulk Carrier", "dwt": 93274, "gross_tonnage": 51300,
      "net_tonnage": 31192, "loa_meters": 229.2, "beam_meters": 38.0,
      "draft_sw_s_w_t": [14.9, 0.0, 0.0]},
    "operational_data": {"cargo_quantity_mt": 40000, "days_alongside": 3.39,
      "arrival_time": "2024-11-15T10:12:00", "departure_time": "2024-11-22T13:00:00",
      "activity": "Exporting Iron Ore"}
  }
}'
```

Optional fields: `overrides` (`num_services`, and `facts` to settle any fact a rule asks about,
by the names listed in each line item's `facts`, e.g. `{"facts": {"is_used_for_gain": true}}`),
`charge_ids` to price only some charges,
`requested_charges` to include services charged only on request, and `document_id` to pick a
document explicitly.

The response (abridged):

```json
{
  "status": "success",
  "port": "Durban",
  "currency": "ZAR",
  "vat": {"rate": "0.1500", "included": false},
  "total": "509991.33",
  "line_items": [
    {
      "charge_id": "light_dues",
      "name": "LIGHT DUES",
      "amount": "60062.04",
      "confidence": "high",
      "formula": [
        "Self-propelled vessel licensed by the Department of Environmental Affairs and Tourism at its registered port, per financial year or part thereof: not applicable to this call",
        "All other vessels, per 100 tons or part thereof: ceil(51,300 / 100) = 513; 513 × 117.08 = 60,062.04",
        "Amount (rounded to cents): 60,062.04"
      ],
      "facts": [{"name": "is_used_for_gain", "value": "true", "source": "vessel_data", "reason": "Exporting iron ore cargo"}, "..."],
      "assumptions": [],
      "citations": [{"chunk_id": 469, "section_ref": "1.1", "page": 5, "quote": "Light dues in accordance with the vessels tonnage definition as follows: ..."}, "..."]
    }
  ],
  "not_applicable": [{"name": "Fees for surveying/examination of life saving appliances", "reason": "Not applicable: requires The service of surveying/examination of life saving appliances is performed. = yes"}, "..."],
  "not_priced": [{"name": "SAMSA LEVY", "reason": "The tariff provides no levy rate; it incorporates rates prescribed in the SAMSA Levy Determination Regulations in force. ..."}],
  "on_request": ["..."], "excluded": ["..."], "failed": [], "warnings": []
}
```

Every fact a rule depends on is reported with its source: `vessel_data` (stated or implied by
the input), `presumed` (typical for a call like this one), `default` (the rule's default for an
ordinary call), or `override`. A rule the compiler's critic didn't fully approve is priced and
marked `confidence: low`, with its open issues. `status` is `partial` when a charge failed to
compile; it is then listed under `failed` and left out of the total.

### Uploading a tariff

```bash
curl -s -F "file=@tariff.pdf;type=application/pdf" localhost:8080/v1/documents   # 202 + document id
curl -s localhost:8080/v1/documents/<id>                                          # poll until "ready"
```

The same file again returns the existing document (`200`); `?force=true` rebuilds it, which also
deletes its compiled rules. Non-PDFs are rejected (`415`), as are files over `MAX_UPLOAD_MB`
(`413`).

### Compiling and reviewing rules

A port's rules are compiled once per document, cached in the database, and reused by every
calculation. Compiling a whole port takes minutes (Durban: 23 charges, about 6 minutes and 1.2M
tokens with `gpt-5.6-sol`), longer than an HTTP request may run, so use the CLI for that:

```bash
docker compose exec api python scripts/compile_rules.py --port "Richards Bay" --verbose
curl -s "localhost:8080/v1/rules?port=Richards%20Bay"   # review: status, open issues, rules
```

`POST /v1/rules/compile` with `charge_ids` compiles a few charges through the API. To recompile
one charge after reviewing it: `--charge <id> --refresh`.

### API key and tracing

- **API key:** set `API_AUTH_KEY` and every `/v1/*` call needs an `X-API-Key` header (Swagger's
  Authorize button sends it). `/health` and `/ready` stay open.
- **Tracing:** every calculation is stored with the agent's steps (`GET /v1/calculations/{id}`:
  sections opened, searches, evidence, extractions, validation problems, reviews, fact
  decisions). Optional Langfuse tracing (`LANGFUSE_TRACING_ENABLED=true` plus keys) adds one
  trace per calculation and per standalone compilation, with every model call nested in it;
  prompts and responses are left out unless `LANGFUSE_CAPTURE_CONTENT=true`.
- **Logs:** structured JSON; every line of a request carries its `request_id`.

## Command-line tools

| Command | What it does |
|---|---|
| `make ingest` / `scripts/ingest.py --file <pdf>` | Ingest PDFs (idempotent by checksum; `--force` rebuilds) |
| `make compile PORT=Durban` / `scripts/compile_rules.py` | Compile a port's rules (`--charge`, `--refresh`, `--all`, `--out-dir`, `--verbose`) |
| `make rulebook-export PORT=Durban` | Write the compiled rulebook to `data/rulebooks/<pdf>.json` |
| `make rulebook-import` | Load every file in `data/rulebooks/` (the container does this on start) |
| `make calculate` / `scripts/calculate.py` | Price a call from rule JSON files, without the database or an LLM |
| `make eval` / `eval/run_eval.py` | Live accuracy eval on every case in `eval/cases/` (`--case`, `--refresh`) |
| `make eval-retrieval` | Recall@5 of the hybrid search on 17 queries against the TNPA book (0.94) |
| `make smoke` | One real structured-output call and one embedding call |

## How it works

The diagrams are in [docs/architecture.md](docs/architecture.md); the phased build plan with its
gates is [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md).

1. **Ingestion** (once per PDF). PyMuPDF parses the text layer: two-up landscape sheets are split
   into their printed pages, running headers and page numbers are removed, tables are kept whole
   as markdown, ligatures are spelled out. A section tree is built from numbered or styled
   headings, then chunks with breadcrumbs (whole tables, one chunk per defined term). Chunks are
   embedded and indexed for full-text search in Postgres. Two LLM calls read the document
   profile (ports, currency, VAT, validity) and the charge catalogue (each charge with its
   sections, payer and trigger).
2. **Rule compilation** (once per document and port; LangGraph, per charge, concurrently):
   - *research*: a tool-using agent searches (pgvector + full-text, fused with reciprocal rank
     fusion), opens sections, looks up definitions, and submits its evidence;
   - *extract*: a structured-output call writes the charge as a `ChargeRule` in the rule DSL
     (`app/rules/dsl.py`): fixed fees, rates per unit "or part thereof", size bands, marginal
     tiers, per-time charges, minimums and maximums, multipliers, conditions, exemptions,
     reductions and surcharges (with exclusive groups), facts with defaults, citations;
   - *validate*: schema checks plus **grounding**: every number in the rule must appear in the
     text it cites, and every quote must be found in its chunk;
   - *critique*: a second model call reviews the rule against the evidence, runs it on an
     example vessel, and grades issues as blocking or minor; blocking issues go back to the
     extractor (up to `AGENT_MAX_REVISIONS`), and the best version is kept.
3. **Calculation** (per request; LangGraph): normalise the input (a structured-output call reads a
   plain-language query) → resolve the document for the port and date → screen the catalogue
   deterministically (paid by cargo, licences, on-request services) → load or compile each
   remaining charge's rule → resolve the facts the rules ask about (batched LLM calls that see
   the call data and how each rule uses each fact) → evaluate every rule with the engine.

## Configuration

Set in `.env` (see `.env.example`); the main settings:

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | none | Required unless `LLM_PROVIDER=fake` |
| `LLM_PROVIDER` | `openai` | `fake` uses deterministic scripted clients |
| `LLM_MODEL` | `gpt-5.6-luna` | Runtime calls: query parsing, fact resolution |
| `LLM_COMPILE_MODEL` | empty (= `LLM_MODEL`) | Ingestion profile/catalogue and rule compilation; `gpt-5.6-sol` recommended |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Chunk and query embeddings (1536 dimensions) |
| `LLM_TIMEOUT_SECONDS` | `120` | Per model call |
| `LLM_MAX_RETRIES` | `2` | Retries on rate limits, 5xx and connection errors (not on an empty balance) |
| `LLM_MAX_CONCURRENCY` | `8` | Concurrent model calls per process |
| `AGENT_MAX_TOOL_CALLS` | `8` | Research tool calls per charge |
| `AGENT_MAX_REVISIONS` | `2` | Extract/review rounds per charge |
| `CALCULATION_TIMEOUT_SECONDS` | `300` | Whole calculation; Nginx allows 315 s |
| `TARIFFS_DIR` | `./data/tariffs` | PDFs ingested on start |
| `RULEBOOKS_DIR` | `./data/rulebooks` | Rulebook files loaded on start |
| `MAX_UPLOAD_MB` | `25` | Upload size limit |
| `API_AUTH_KEY` | empty | When set, required as `X-API-Key` on `/v1/*` |
| `LANGFUSE_*` | off | Optional tracing |
| `HTTP_PORT` | `8080` | Host port for Nginx (Compose) |

## Design decisions

- **The LLM never calculates.** Language models are good at reading "per 100 tons or part
  thereof, with a minimum of ..." and unreliable at multiplying 513 by 117.08. The model's
  output is a typed rule; the engine computes with `Decimal`, half-up rounding to cents, and
  explains every step. The same rule gives the same amount every time.
- **A rule DSL rather than free-form extraction.** The DSL is a vocabulary of general tariff
  mechanics, not of any document. Rules can be validated, grounded, reviewed, cached, exported
  and read by a human (`GET /v1/rules`), and new documents need no new code unless they use a
  mechanic the DSL lacks.
- **Grounding as a hard gate.** Every number in a rule must be printed in the text it cites
  (after normalising "30 960.46", "1,234.50", "35%", "the fifth day"), and quotes must be
  verbatim. It is a cheap, deterministic guard against invented or mistyped rates.
- **A critic with graded severity.** The reviewer shares the extractor's description of the
  engine's semantics, so they agree on what a rule means, and only blocking issues trigger a
  revision. Rules that still have open issues are kept, priced and flagged, never silently
  dropped.
- **Compile once, cache, export.** Rules are cached per document, port, rule schema version and
  prompt version, so a calculation never silently recompiles and changes its answer; a prompt
  change recompiles. Reviewed rulebooks are exported as files and loaded into fresh databases,
  together with the catalogue they were compiled against. They are data produced from the
  document, like the cache they fill.
- **Two models.** Compilation is offline and cached, so it can afford a stronger model
  (`LLM_COMPILE_MODEL`); runtime calls use a faster one.
- **Facts are decided, with sources.** What a rule needs to know about the call (is the vessel at
  its registered port, is it cargo-working, did it request towage) is decided per call by a
  model that sees how the rule uses each fact, and reported with its source; anything can be
  overridden in the request.
- **Postgres + pgvector, not a separate vector database.** One store for documents, chunks,
  embeddings (HNSW), full-text search, compiled rules and calculation history; hybrid retrieval
  is one SQL round trip, and the data volume (hundreds of chunks per document) is tiny.
- **LangGraph for the agents.** Compilation and calculation are graphs with loops (revise until
  approved), fan-out (one compile per charge, concurrently) and state that must be traced;
  LangGraph gives that explicitly, and each node is a plain async function that is easy to test
  with a fake LLM.
- **Strict structured outputs.** All model calls that produce data use OpenAI Structured Outputs
  (Responses API) in strict mode, so responses always parse into the Pydantic models.
- **Proving generality.** A guard test forbids tariff names and figures under `app/`, prompt
  examples use invented values, and a synthetic tariff with a different layout and mechanics is
  part of the eval.

## Limitations

- **Cold compiles cost time and money.** A whole port takes minutes and around a million tokens
  with a strong model, and results vary between runs on the most complex charges. The intended
  workflow is to compile once, review the rulebook (flags and open issues are listed), recompile
  single charges, and export it.
- **Some rules stay flagged.** 6 of the 23 compiled Durban rules are `low_confidence`, mostly
  for cases an ordinary call doesn't reach: surcharges that apply to only one of several
  movements (the engine applies adjustments to the whole charge), cancellation and standby
  cases, drydock dues, visiting pleasure vessels. They are priced and marked, not hidden.
- **Facts come from a model.** For ambiguous calls the fact resolver may decide differently from
  run to run; each fact is reported with its reason, and `overrides.facts` settles it.
- **Text-layer PDFs only.** Scanned tariffs need OCR first (`PARSER_VISION_FALLBACK` is reserved
  but not implemented).
- **VAT is reported, not added** (`vat.included: false`); amounts are as the tariff prints them.
- **One document per port and date.** The newest ready document in force on the arrival date is
  used, unless `document_id` is given.
- **No front end.** Swagger UI covers uploads and calculations.

## Project layout

```
app/
  agent/         LangGraph graphs and nodes: research, extract, validate, critique; normalise, screen, facts
  domain/        vessel call model and quantities, number parsing, port matching
  ingestion/     PDF parser, cleaner, section structure, chunker, profile and catalogue
  llm/           OpenAI client (structured outputs, resilience), embeddings, versioned prompts
  retrieval/     hybrid search (pgvector + full-text + RRF)
  rules/         rule DSL, engine, grounding
  services/      documents, rulebook (compile + cache), rulebook files, calculations
  routers/       FastAPI routes
data/
  tariffs/       bundled tariff PDFs (ingested on start)
  rulebooks/     exported compiled rulebooks (loaded on start)
  synthetic/     the synthetic Exampleville tariff (eval)
eval/            accuracy cases and harness, retrieval eval
examples/        Swagger request examples
scripts/         CLIs: ingest, compile_rules, rulebook, calculate, llm_smoke, make_synthetic_tariff
tests/           unit (engine, DSL, grounding, parsing, ...) and integration (Postgres, fake LLM)
docs/            BUILD_PLAN.md, architecture.md
```
