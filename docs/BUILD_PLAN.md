# BUILD PLAN — Port Tariff Agent (agentic RAG for port dues)

> **How to use this file:** keep it at `docs/BUILD_PLAN.md` next to `docs/architecture.md`, run
> Claude Code in the repository root and say:
> *"Read docs/BUILD_PLAN.md and docs/architecture.md and implement Phase 0 and Phase 1. Stop at the
> Phase 1 gate and show me the verification output."*
> Work through the phases one or two at a time. Do not let the agent skip the gates.

---

## 1. What we are building

A backend service that reads a port's tariff document (PDF) and a vessel call, given as a
structured vessel profile, a natural-language query, or both. It returns every due the vessel
must pay at that port: light dues, port dues, VTS, pilotage, towage, berthing, running of lines,
and anything else the document defines. Each line item comes with its amount, the formula that
produced it, the tariff sections and quotes it was derived from, and the assumptions made.

The system must work for **any** port's tariff document. An agent finds and interprets, from the
document itself, which charges exist, what they cost, how they scale and when they apply. None of
that is written into the code. Supporting a new port (Amsterdam, Rotterdam, …) or a new edition of
a tariff book means uploading a PDF, not changing code.

Validation case: Transnet National Ports Authority *Tariff Book April 2024 – March 2025* (bundled in
`data/tariffs/`) and the bulk carrier SUDESTADA calling at Durban (§13).

### Success criteria

1. `docker compose up --build` on a clean machine produces a working system with the bundled
   tariff document already ingested.
2. `POST /v1/calculations` for SUDESTADA at Durban returns all six reference charges within 1% of
   the ground truth, each with formula, citations and assumptions.
3. A second tariff document with a different structure (a synthetic test port) is ingested and
   priced correctly with **zero** changes under `app/`.
4. No port names, charge names, rates or section numbers anywhere under `app/`, enforced by a test.
5. `pytest` passes offline: LLM and embeddings stubbed, no API key, no network.
6. The README covers setup, execution, architecture, design decisions, the accuracy table and
   known limitations.

---

## 2. Core idea: the LLM reads, the code calculates

LLMs are good at reading dense conditional prose and messy tables. They are unreliable at
arithmetic and inconsistent between runs. The architecture splits the work along that line:

| Layer | Done by | Produces |
|---|---|---|
| **Knowledge:** parse the PDF into a section tree, keep tables intact, index for hybrid search, discover which charges the document defines | Ingestion pipeline (+2 LLM calls per document) | Sections, chunks, embeddings, document profile, charge catalogue |
| **Interpretation:** for each charge, research the document, extract the rule as typed data with citations, decide which conditions hold for this vessel, check its own work | LangGraph agent (OpenAI) | `ChargeRule` JSON, resolved facts, critic verdict |
| **Calculation:** evaluate the rule against the vessel's quantities | Pure-Python `Decimal` engine, no LLM | Line items with a step-by-step trace |

The contract between the agent and the engine is the **rule DSL** (§6.2), a small typed vocabulary
of tariff mechanics: fixed fees, rates per unit "or part thereof", tonnage bands, marginal tiers,
time pro-rata, minimums and maximums, and conditional percentage reductions and surcharges. The DSL
describes *how tariffs work in general*. The document supplies *which* rules and numbers apply.
That separation is what makes the system portable to other ports.

Compiled rules are cached per (document, port, charge). The first calculation for a port does the
research. Later calculations reuse the approved rules, so they are fast, cheap and repeatable. A
new edition of a tariff book has a new checksum and therefore gets its own rules.

---

## 3. Hard constraints

These decisions are made. Do not revisit them or propose alternatives mid-build.

### DO

- Python 3.12, FastAPI, SQLAlchemy 2.0 (async) + asyncpg, Alembic, Pydantic v2, pydantic-settings,
  structlog, uv, ruff, pytest.
- **PostgreSQL 16 + pgvector** as the only datastore: documents, sections, chunks, embeddings,
  full-text index, charge catalogue, compiled rules, calculations, agent traces.
- **OpenAI** for every model call: a chat model for extraction and reasoning, `text-embedding-3-*`
  for embeddings, through `langchain-openai`. Chat calls use the Responses API: reasoning models
  such as `gpt-5.6-luna` accept function tools only there.
- Every LLM extraction uses **Structured Outputs**
  (`with_structured_output(Model, method="json_schema", strict=True)`) into a Pydantic model.
  Never regex-parse model prose.
- **LangGraph** for the calculation agent only. Unlike the reference support-assistant project, this
  flow has real branching: a fan-out with one branch per charge, conditional routing (applicable,
  not applicable, not priced) and bounded revision loops.
- **PyMuPDF** for PDF parsing (text spans with font metadata, `page.find_tables()` → markdown).
- Money and quantities are `decimal.Decimal` end to end. JSON responses carry amounts as strings.
- Every number in a compiled rule must be **grounded**: after number normalisation, it must appear
  in the tariff text the rule cites. Ungrounded rules are rejected and re-extracted.
- Every line item carries its amount, formula trace, citations (section, page, quote) and
  assumptions.
- Concurrency to OpenAI is bounded by one `asyncio.Semaphore` per process, with the timeout and
  retry policy from §12 (carry over `app/llm/client.py` from the support-assistant project).
- All configuration comes from environment variables via `pydantic-settings`.

### DO NOT

- **Do not put tariff knowledge in code.** No port names, charge names, rates, section numbers,
  table layouts or document-specific regexes under `app/`. `tests/test_no_hardcoding.py` greps for
  them.
- **Do not let the LLM do arithmetic** or produce totals. Never `eval()` anything a model returns.
- Do not add a separate vector database (Chroma, Pinecone, Qdrant, …). A tariff book is a few
  hundred chunks; pgvector handles that and keeps one datastore.
- Do not add LlamaIndex alongside LangChain. Do not add a message broker, Celery, Redis, Kubernetes
  or a frontend.
- Do not use few-shot examples taken from the TNPA document. Prompt examples use an invented port,
  so prompts don't overfit to the validation case.
- Do not tune prompts or code to the reference values. The reference case is a test, not training
  data. When a figure is off, fix the general cause and add a unit test for it.
- Do not commit secrets. `.env` is gitignored; `.env.example` has dummy values.

---

## 4. Repository layout

```
.
├── app/
│   ├── main.py                    # app factory, lifespan, routers, error handlers
│   ├── config.py                  # Settings (pydantic-settings)
│   ├── db.py                      # async engine, session factory, get_session
│   ├── models.py                  # SQLAlchemy ORM models (§7)
│   ├── schemas.py                 # API request/response models (§11)
│   ├── errors.py                  # domain exceptions → HTTP error codes
│   ├── logging_conf.py            # structlog JSON logging
│   ├── middleware.py              # request_id, access log, timing
│   ├── observability.py           # optional Langfuse tracing
│   ├── domain/
│   │   ├── vessel.py              # VesselCall, derived quantities, profile mapping (§6.1)
│   │   ├── ports.py               # match a requested port to a document's ports
│   │   └── numbers.py             # Decimal helpers, number normalisation ("2  801.91" → 2801.91)
│   ├── rules/
│   │   ├── dsl.py                 # ChargeRule + components: the agent ↔ engine contract (§6.2)
│   │   ├── engine.py              # deterministic evaluator → LineItem + trace (§6.3)
│   │   └── grounding.py           # checks every number in a rule against its cited text
│   ├── ingestion/
│   │   ├── parser.py              # PDF → logical pages: text lines (with font info), tables
│   │   ├── cleaner.py             # running headers/footers, page labels, dot leaders
│   │   ├── structure.py           # heading detection → section tree
│   │   ├── chunker.py             # section-aware chunks, atomic tables, breadcrumbs
│   │   ├── profile.py             # LLM: document profile (authority, ports, currency, VAT, validity)
│   │   ├── catalogue.py           # LLM: charge discovery
│   │   └── pipeline.py            # orchestration + status transitions
│   ├── retrieval/
│   │   ├── search.py              # pgvector + Postgres full-text, reciprocal rank fusion
│   │   └── context.py             # section expansion, definition lookup
│   ├── llm/
│   │   ├── client.py              # LLMClient protocol, OpenAIClient, FakeLLMClient
│   │   ├── embeddings.py          # Embedder protocol, OpenAIEmbedder, FakeEmbedder
│   │   ├── resilience.py          # shared semaphore, timeout, retry policy, error types
│   │   ├── schema.py              # Pydantic model → strict Structured Outputs JSON schema
│   │   └── prompts/               # one module per prompt, each with a version
│   ├── agent/
│   │   ├── state.py               # graph state models
│   │   ├── compile.py             # LangGraph: research → extract → validate → critique (§9)
│   │   ├── graph.py               # LangGraph: the calculation graph (Phase 7)
│   │   ├── tools.py               # SearchTariff, ReadSection, LookupDefinition, ListSections, SubmitEvidence
│   │   └── nodes/                 # one module per node in §9
│   ├── services/
│   │   ├── documents.py           # register, ingest, list
│   │   ├── rulebook.py            # compiled-rule cache, compile all charges for a port
│   │   └── calculations.py        # run the graph, persist, map errors
│   └── routers/
│       ├── documents.py
│       ├── rules.py
│       ├── calculations.py
│       └── health.py
├── data/tariffs/                  # bundled tariff PDFs, ingested at startup
├── eval/
│   ├── cases/sudestada_durban.json
│   ├── cases/synthetic_port.json
│   ├── retrieval_cases.json       # query → expected section refs
│   ├── run_retrieval_eval.py      # recall@k of hybrid search (make eval-retrieval)
│   └── run_eval.py                # live end-to-end accuracy report
├── scripts/
│   ├── ingest.py                  # CLI: ingest a PDF (idempotent)
│   ├── compile_rules.py           # CLI: compile + export the rulebook for a port
│   ├── calculate.py               # CLI: price a vessel call from rule files, without the API
│   ├── llm_smoke.py               # one real extraction + embedding call (Phase 3 gate)
│   └── make_synthetic_tariff.py   # renders the synthetic test-port PDF
├── tests/
│   ├── unit/                      # engine, DSL, grounding, numbers, structure, chunker, fusion
│   ├── integration/               # ingestion + retrieval on the real PDF; API and graph with fakes
│   ├── fixtures/                  # golden rules (test data), recorded LLM outputs, synthetic tariff
│   └── test_no_hardcoding.py
├── migrations/                    # Alembic (async)
├── nginx/nginx.conf
├── docs/BUILD_PLAN.md, docs/architecture.md
├── Dockerfile, compose.yaml, compose.override.yaml, docker-entrypoint.sh
├── .env.example, .gitignore, .dockerignore
├── alembic.ini, pyproject.toml, uv.lock, Makefile
└── README.md
```

**Layering rule:** `domain/` and `rules/` import nothing from `llm/`, `db`, `retrieval/` or
`agent/`. They are pure and fully unit-testable. `agent/` depends on `rules/`, `retrieval/` and
`llm/`. Routers depend only on `services/`.

---

## 5. Configuration

All settings live in one `Settings` class in `app/config.py`. Every value appears in `.env.example`.

| Variable | Default | Purpose |
|---|---|---|
| `APP_ENV` | `development` | `development` \| `production` \| `test` |
| `LOG_LEVEL` | `INFO` | |
| `DATABASE_URL` | none | `postgresql+asyncpg://...` |
| `LLM_PROVIDER` | `openai` | `fake` selects the deterministic `FakeClient` + `FakeEmbedder` |
| `LLM_MODEL` | `gpt-5.6-luna` | extraction and reasoning. **Verify the current model id** |
| `LLM_COMPILE_MODEL` | empty | model for ingestion and rule compilation (cached, once per document and port); empty = `LLM_MODEL` |
| `OPENAI_API_KEY` | none | required unless `LLM_PROVIDER=fake` |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | verify the current model id |
| `EMBEDDING_DIMENSIONS` | `1536` | must match the `vector(n)` column in the migration |
| `LLM_TEMPERATURE` | `0` | omit the parameter if the chosen model rejects it |
| `LLM_REASONING_EFFORT` | empty | reasoning models only (`minimal` … `high`); empty keeps the model default |
| `LLM_TIMEOUT_SECONDS` | `120` | ceiling for one model call (large rule extractions take over a minute) |
| `LLM_MAX_RETRIES` | `2` | retries on 429/5xx only |
| `LLM_MAX_CONCURRENCY` | `8` | semaphore size **per worker process** |
| `AGENT_MAX_TOOL_CALLS` | `8` | per-charge research loop budget |
| `AGENT_MAX_REVISIONS` | `2` | validate/critique → re-extract loops per charge |
| `AGENT_CRITIC_ON_CACHE_HIT` | `false` | re-run the critic when a cached rule is reused |
| `CALCULATION_TIMEOUT_SECONDS` | `300` | budget for one whole calculation (cold cache) |
| `RETRIEVAL_TOP_K` | `8` | chunks returned per search |
| `RETRIEVAL_RRF_K` | `60` | reciprocal rank fusion constant |
| `PARSER_VISION_FALLBACK` | `false` | re-transcribe pages with suspect tables via an OpenAI vision call |
| `TARIFFS_DIR` | `./data/tariffs` | PDFs ingested on startup |
| `MAX_UPLOAD_MB` | `25` | |
| `API_AUTH_KEY` | empty | if set, every `/v1/*` call needs `X-API-Key` (for the public deployment) |
| `UVICORN_WORKERS` | `2` | |
| `LANGFUSE_*` | as in the support-assistant project | optional tracing |

`LLM_PROVIDER=fake` must select the fakes. Tests rely on it.

---

## 6. Domain model

### 6.1 Vessel call (canonical input)

`app/domain/vessel.py` defines `VesselCall`. The profile format from the brief
(`vessel_metadata` / `technical_specs` / `operational_data`) is accepted as-is through Pydantic
aliases, alongside a flat form.

| Field | From the brief's profile | Notes |
|---|---|---|
| `name`, `vessel_type`, `flag`, `built_year` | `vessel_metadata.*`, `technical_specs.type` | |
| `gross_tonnage`, `net_tonnage`, `deadweight` | `technical_specs.gross_tonnage`, `net_tonnage`, `dwt` | GT is required |
| `loa_m`, `beam_m`, `draft_m` | `loa_meters`, `beam_meters`, `draft_sw_s_w_t[0]` | |
| `arrival`, `departure`, `days_alongside` | `operational_data.*` | |
| `activity`, `cargo_tonnes`, `num_operations`, `num_holds` | `operational_data.*` | |
| `port` | request field or the NL query | resolved against the document's port list |

**Quantities:** a rule may only reference these numeric variables (the `Basis` enum).
`gross_tonnage, net_tonnage, deadweight, loa_m, beam_m, draft_m, cargo_tonnes,
time_in_port_hours, time_in_port_days, call_window_days, num_services, passengers`.

- `time_in_port_days` is `days_alongside` when given, otherwise `departure − arrival`, and the
  choice is recorded as an assumption. Arrival/departure timestamps often include waiting at the
  outer anchorage, outside port limits, where time-based port dues are not charged.
  `call_window_days` is always `departure − arrival`.
- `num_services` is the number of marine-service movements: 2 (entering + leaving) unless the call
  says otherwise (shifts, multiple berths). It can be overridden per request.

**Facts:** qualitative conditions such as "engaged in cargo working", "bona fide coaster",
"service outside ordinary working hours", "first port of call in the country" or "self-propelled"
are open vocabulary. A rule declares the facts its conditions need, each with a description lifted
from the tariff, and the agent resolves them for the call (§9). Nothing in code enumerates them.

### 6.2 Rule DSL (`app/rules/dsl.py`)

These Pydantic models are the JSON schema the extraction call must fill, so every field has a
`description` written for the model.

```python
class Rounding(StrEnum):
    CEIL = "ceil"            # "per 100 tons or part thereof"
    FLOOR = "floor"
    PRO_RATA = "pro_rata"    # "a part of a 24 hour period being applied pro rata"

class Units(BaseModel):      # how many billable units of a quantity
    basis: Basis             # e.g. gross_tonnage, time_in_port_hours, num_services
    unit_size: Decimal = 1   # e.g. 100 (tons), 24 (hours)
    rounding: Rounding
    above: Decimal = 0       # count only the part above this value ("per 100 tons above 50 000")
    less: str | None         # also deduct a number fact or quantity ("time in port less hours worked")

# Every component has: id; label; when: list[Condition]  (counts only if all hold, e.g. a
# different rate for vessels at their registered port)
class FixedFee(BaseModel):    kind: Literal["fixed"];    amount
class PerUnitFee(BaseModel):  kind: Literal["per_unit"]; rate; units: Units
                              per_time: Units | None     # rate × units × time units
class Increment(BaseModel):   rate; units: Units
class Band(BaseModel):        lower; upper: Decimal | None; base_fee; increment: Increment | None   # lower <= q <= upper
class BandedFee(BaseModel):   kind: Literal["banded"];   basis; bands: list[Band]   # first listed band containing q applies
class Tier(BaseModel):        up_to | width: Decimal | None; rate; unit_size; rounding   # printed bound, or printed slice size
class TieredFee(BaseModel):   kind: Literal["tiered"];   basis; tiers: list[Tier]   # marginal: each slice at its own rate
class UnpricedCase(BaseModel): kind: Literal["unpriced"]; when (required)   # a case left to agreement/application
Component = Annotated[FixedFee | PerUnitFee | BandedFee | TieredFee | UnpricedCase, Field(discriminator="kind")]

class Condition(BaseModel):   fact: str; op: Literal["eq","ne","lt","le","gt","ge","in"]; value
class Adjustment(BaseModel):  id; kind: Literal["reduction","surcharge"]; description; percent
                              applies_to: list[str] | Literal["all"]   # component ids
                              when: list[Condition]; exclusive_group: str | None; citation
class Exemption(BaseModel):   description; when: list[Condition]; citation
class FactSpec(BaseModel):    name; type: Literal["bool","number","text"]; description
                              default_value: bool | str | None   # assumed when the call doesn't say; None = must resolve
class Citation(BaseModel):    chunk_id; section_ref; page; quote

class ChargeRule(BaseModel):
    charge_id; name; section_refs; port_key; currency; payer: Literal["vessel","cargo","other"]
    status: Literal["priced", "on_application", "not_priced_in_document", "not_applicable_at_port"]
    applies_when: list[Condition]          # all must hold, otherwise not applicable
    exemptions: list[Exemption]
    components: list[Component]            # summed → amount per service/call
    minimum: Decimal | None; maximum: Decimal | None
    multiplier: Units | None               # e.g. × num_services
    adjustments: list[Adjustment]
    facts: list[FactSpec]
    citations: list[Citation]
    notes: list[str]                       # interpretation notes shown to the reader
```

A rule describes the whole charge at one port (every band, every reduction), not just the branch
this vessel hits. That keeps compiled rules vessel-independent and therefore cacheable. A worked
example (towage at Durban) is in `docs/architecture.md`.

### 6.3 Engine semantics (`app/rules/engine.py`)

1. If `status != "priced"`, return that status with the rule's citation.
2. If any exemption matches or any `applies_when` condition fails, return `not_applicable` with the
   condition and citation that decided it. If an `unpriced` component's conditions hold, return
   `not_priced_in_document` with its label as the reason.
3. Evaluate each component in full `Decimal` precision and record a trace step, e.g.
   `ceil(51 300 / 100) = 513 × 117.08 = 60 062.04`.
4. Sum the components, clamp to `minimum`/`maximum`, then multiply by `multiplier` units.
5. Apply adjustments whose conditions hold. Within one `exclusive_group` only the largest applies
   ("not enjoyed in addition to …"). Percentages combine additively on the pre-adjustment amount
   of the components they target.
6. Round each line item to 2 dp with `ROUND_HALF_UP`. The total is the sum of the rounded items.
   Amounts exclude VAT; the response reports the VAT rate from the document profile separately.

Missing quantities never default to zero. They raise `MissingQuantityError`, which surfaces as an
`INSUFFICIENT_VESSEL_DATA` warning for that charge.

---

## 7. Data model

Alembic migration 1 runs `CREATE EXTENSION IF NOT EXISTS vector` and creates every table.

### `tariff_document`
`id uuid PK`, `title`, `authority`, `source_filename`, `content bytea` (the PDF),
`checksum char(64) unique`, `page_count`, `currency`, `vat_rate numeric`, `effective_from date`,
`effective_to date`, `ports jsonb` (canonical names + aliases), `status` (`pending` | `parsing` |
`indexing` | `cataloguing` | `ready` | `failed`), `error text`, `created_at`.
Re-uploading the same bytes returns the existing row. A new edition is a new row; nothing is
updated in place.

### `document_section`
`id`, `document_id FK`, `parent_id FK null`, `ref` (e.g. `3.6`, or a generated `s-17` when the
document doesn't number its headings), `title`, `path` (breadcrumb), `level`, `page_start`,
`page_end`, `ordinal`, `kind` (`body` | `definitions` | `contents`).

### `chunk`
`id`, `document_id FK`, `section_id FK`, `kind` (`text` | `table` | `definition`), `content text`
(markdown, prefixed with the breadcrumb), `page`, `printed_page`, `ordinal`, `token_count`,
`embedding vector(1536)`, `tsv tsvector GENERATED ALWAYS AS (...) STORED`: the breadcrumb is weighted
`A` and the body `B`, and `/` is split to a space (the full-text parser otherwise reads "TUGS/VESSEL" as
one file-path token).
Indexes: HNSW on `embedding` (cosine), GIN on `tsv`, `(document_id, section_id)`.

### `charge_catalogue_entry`
`id`, `document_id FK`, `charge_id` (slug the LLM produces), `name`, `section_refs jsonb`, `payer`,
`trigger` (`per_call` | `per_service` | `per_period` | `on_request` | `licence_or_permit`),
`description`. Unique `(document_id, charge_id)`.

### `compiled_rule`
`id`, `document_id FK`, `port_key`, `charge_id`, `rule jsonb`, `rule_schema_version`,
`prompt_version`, `model`, `critic_verdict jsonb` (outcome, open issues, review notes, revisions,
sections read, research notes), `created_at`.
Unique `(document_id, port_key, charge_id, rule_schema_version, prompt_version)`. Every outcome
is written: approved rules; rules the critic still had blocking issues with after the last revision
(outcome `low_confidence`, the version with the fewest issues, issues kept); and failures (`rule`
null: no extraction passed validation). So a calculation never silently recompiles and changes its
answer; `refresh` retries. Provider errors are not cached. Bumping a prompt or schema version
invalidates the cache without a migration.

### `calculation`
`id uuid PK`, `request_id`, `document_id FK`, `port_key`, `vessel_call jsonb`, `query_text`,
`status` (`success` | `partial` | `failed`), `result jsonb`, `total numeric`, `error_code`,
`model`, `prompt_tokens`, `completion_tokens`, `latency_ms`, `created_at`.
Failed calculations are still written; silently losing failures is a defect.

### `agent_step`
`id`, `calculation_id FK`, `charge_id null`, `node`, `tool null`, `input_summary`, `output jsonb`,
`latency_ms`, `prompt_tokens`, `completion_tokens`, `created_at`. This is the audit trail that
shows *how* the agent found and interpreted each rule. `GET /v1/calculations/{id}` returns it.

---

## 8. Ingestion pipeline (`app/ingestion/`)

Runs on startup for every PDF in `TARIFFS_DIR` and on `POST /v1/documents` (as a FastAPI background
task). Each step moves `tariff_document.status`; any exception sets `failed` with the error.

1. **Register:** compute sha256. If it is known, stop (idempotent).
2. **Parse** (PyMuPDF): per page, collect text spans with font size and weight, and tables from
   `page.find_tables()` rendered as markdown. Text inside table bounding boxes is dropped from the
   prose stream so it isn't duplicated. Record both the PDF page number and the printed page label
   (this PDF has two printed pages per sheet).
3. **Clean:** drop running headers and footers. A line counts as one if it repeats on more than
   half of the pages (e.g. "Tariff Book April 2024 – March 2025", "Tariffs subject to VAT at 15%");
   keep one copy for the document profile. Fix hyphenation across line breaks.
4. **Structure:** detect headings from numbering patterns (`1`, `1.1`, `4.1.2`, `A.`, `(a)`) and
   font cues (size above the body median, bold, all-caps), then build the tree by numbering depth or
   font rank. If the tree looks implausible (too few sections, or most text under one node), send
   the contents page plus the candidate headings to the LLM for an outline. That fallback is
   generic: it covers tariff books that don't number their sections.
5. **Chunk:** one chunk per leaf section up to ~1,200 tokens, split at paragraph boundaries beyond
   that. **Tables are never split.** Each table chunk includes its section breadcrumb and the
   paragraph that introduces it, because a table without its caption is meaningless. Every chunk is
   prefixed with its breadcrumb (`SECTION 3 MARINE SERVICES > 3.6 TUGS/VESSEL ASSISTANCE`) before
   embedding. Definition sections are split per term (`kind=definition`).
6. **Index:** embed in batches; `tsv` is a generated column.
7. **Document profile** (one structured LLM call over the first pages and the outline): title,
   authority, currency, VAT rate, effective period, ports covered with aliases, and global
   conventions such as how tonnage is defined.
8. **Charge catalogue** (one structured LLM call over the outline, with section titles and the
   first ~300 characters of each section): every charge the document defines, with section refs,
   payer and trigger. This is how the system knows that "light dues" or "running of vessel lines"
   exist, without a hard-coded list.

`PARSER_VISION_FALLBACK=true` adds one step after parsing. Pages whose tables look broken (cells
mixing labels and numbers, ragged columns) are rendered to PNG and transcribed to markdown by an
OpenAI vision call. The text layer stays the source of truth for grounding, so a vision
transcription is only used if its numbers also appear in the page text.

---

## 9. Calculation agent (`app/agent/`)

A LangGraph `StateGraph`. The diagrams are in `docs/architecture.md`.

| Node | Kind | What it does |
|---|---|---|
| `normalize_input` | code + LLM | Maps the profile JSON to `VesselCall`. If a `query` is given, a structured LLM call extracts a `VesselCall` + port from it; JSON fields win over NL ones. Missing GT → `422 INSUFFICIENT_VESSEL_DATA` listing the missing fields. |
| `resolve_document` | code | Picks the `ready` document whose `ports` match the port (alias match) and whose effective period contains the arrival date, unless `document_id` is given. None → `422 PORT_NOT_COVERED`. |
| `screen_charges` | code | From the catalogue's payer and trigger: charges paid by cargo or others, and licences/permits, are `excluded`; charges raised only on request or after an incident are listed `on_request` (priced only if the request names them); the rest are candidates. Whether a candidate applies to this call is decided by its own rule (`applies_when`, exemptions, conditional components) against the call's facts — no LLM screening call, so a warm request needs one LLM call in total. |
| *fan-out* | LangGraph `Send` | One charge sub-graph per candidate, run concurrently under the LLM semaphore. |
| `load_rule` | code | Looks up `compiled_rule`. A hit skips straight to `resolve_facts`. |
| `research` | LLM, ReAct | Tool loop (≤ `AGENT_MAX_TOOL_CALLS`) with the goal: "Collect everything needed to compute *<charge>* at *<port>*: rates, units, bands, minimums, conditions, exemptions, reductions, surcharges, and the definitions they rely on." Ends by calling `submit_evidence(chunk_ids)`. |
| `extract_rule` | LLM, structured | Evidence chunks → `ChargeRule` for this port. It picks the port's own column, or the "other ports" / "all other ports" column when the port isn't named, and says which in `notes`. Every number cites a chunk and quote. On a revision it also receives the validator or critic feedback. |
| `validate_rule` | code | Schema checks (contiguous bands, positive unit sizes, known `Basis` values, component ids referenced by adjustments) plus **grounding**: every numeric literal appears in its cited chunk after normalisation. Failure → back to `extract_rule` with the error list. |
| `resolve_facts` | code + LLM | Quantities come from `VesselCall` (code). The qualitative facts of every compiled rule for the call are decided by structured LLM calls over small batches (≤ 12 facts, a rule's facts kept together), run **concurrently**. For each fact the model sees its description in the tariff's words **and how the rule uses it** (which rate, condition, exemption or adjustment each value selects), because a description quoted from a tariff can be ambiguous on its own. It returns `{value, source: vessel_data \| presumed \| default, reason}`: `presumed` marks the typical value for a call of this kind when the data is silent on its ordinary circumstances (where the vessel came from, whether it handles cargo). Request `overrides` win. If a batch fails, its facts keep their rule defaults and the response carries a warning. |
| `compute` | code | Runs the engine (§6.3) → line item + trace. A charge that computes to zero is reported as not applicable. |
| `critique` | LLM, structured | Reviews the evidence and the rule, plus the engine's evaluation of it for one illustrative vessel. Did it use the right port column? Did it miss a reduction, surcharge or minimum? Is "per service" multiplied correctly? Is "or part thereof" rounded up? Each issue is `blocking` (wrong amount or applicability for an ordinary call, or a missing vessel/call-dependent reduction, surcharge, minimum or exemption) or `minor` (anything else, e.g. how request- or delay-triggered extras are modelled). Blocking issues → `extract_rule` (≤ `AGENT_MAX_REVISIONS`); none → approved, minor ones kept as review notes. If revisions run out, the last grounded rule is kept, flagged `low_confidence` with the open issues. Extractor and critic share one description of how the engine evaluates a rule (`prompts/rule_semantics.py`), so they agree on what a rule means. |
| `aggregate` | code (`services/calculations.py`) | Collects the results, orders them by section, totals, attaches warnings and persists `calculation` + `agent_step`. A charge that failed to compile or evaluate is listed under `failed` and gives `status: partial`, never a 500. |

**Tools** (`app/agent/tools.py`, all read-only, scoped to the resolved document):

- `search_tariff(query, k)`: hybrid search → `[{chunk_id, section_ref, path, page, kind, snippet}]`
- `read_section(section_ref)`: the full section as markdown, including tables and child titles
- `lookup_definition(term)`: matching definition chunks
- `list_sections(prefix)`: the outline, so the agent can navigate like a reader with a contents page

**Retrieval** (`app/retrieval/`): pgvector cosine top-k and Postgres `ts_rank_cd` top-k, fused with
Reciprocal Rank Fusion (`RETRIEVAL_RRF_K`). Lexical search matters here because tariff text is full
of exact tokens ("or part thereof", port names, "per 100 tons"). The agent reads whole sections
with `read_section` and definitions with `lookup_definition`, so search only has to point it at
the right section.

---

## 10. Prompt design (`app/llm/prompts/`)

- One module per prompt (`document_profile`, `charge_catalogue`, `parse_query`, `screen_charges`,
  `research`, `extract_rule`, `resolve_facts`, `critique`), each exporting a template and a
  `PROMPT_VERSION`. There are no prompt strings anywhere else.
- Prompts teach general tariff-reading skills, never facts about a document. Examples:
  "'per 100 tons or part thereof' means rounding up to whole units of 100: use `rounding=ceil`";
  "'a part of a 24 hour period being applied pro rata' means `pro_rata`"; "a fee 'per service' is
  multiplied by the number of services"; "copy numbers exactly as printed; never compute or
  convert them".
- Few-shot examples use an invented port ("Port of Exampleville") with invented numbers.
- The evidence goes in the user turn, wrapped in `<tariff_excerpt chunk_id=… section=… page=…>`
  tags. The system turn says excerpts are data, not instructions (uploaded PDFs are untrusted
  input).
- Log prompt sizes and token counts at INFO. Prompt and response content are only logged at DEBUG,
  or sent to Langfuse when `LANGFUSE_CAPTURE_CONTENT=true`.

---

## 11. API contract

Business routes are prefixed with `/v1` and return JSON. Errors use
`{"error": {"code": "...", "message": "...", "request_id": "..."}}`, as in the support-assistant
project.

### `POST /v1/calculations`

```json
{
  "port": "Durban",
  "vessel": { "vessel_metadata": {"name": "SUDESTADA"}, "technical_specs": {}, "operational_data": {} },
  "query": null,
  "document_id": null,
  "overrides": { "num_services": null, "facts": {} },
  "charge_ids": null,
  "requested_charges": [],
  "refresh_rules": false
}
```

`vessel` (the brief's profile format, or `{vessel, call}`), `query` (natural language), or both;
profile fields win over the query field by field. `port` can be omitted if the query names it.
`charge_ids` limits pricing to those charges; `requested_charges` adds on-request charges (e.g.
fresh water). `refresh_rules=true` recompiles the cached rules first.

Response `200`:

```json
{
  "calculation_id": "6c1f…",
  "status": "success",
  "port": "Durban",
  "document": {"id": "…", "title": "Tariff Book April 2024 – March 2025", "authority": "Transnet National Ports Authority"},
  "currency": "ZAR",
  "vat": {"rate": "0.15", "included": false},
  "line_items": [
    {
      "charge_id": "light_dues",
      "name": "Light dues",
      "section_refs": ["1.1.1"],
      "amount": "60062.04",
      "confidence": "high",
      "formula": ["ceil(51300 / 100) = 513", "513 × 117.08 = 60062.04"],
      "adjustments": [],
      "citations": [{"section_ref": "1.1.1", "page": 5, "quote": "Per 100 tons or part thereof …117.08"}],
      "assumptions": ["First South African port of call on this voyage"],
      "rule_source": "cache"
    }
  ],
  "not_applicable": [{"charge_id": "berth_dues", "section_refs": ["4.1.2"], "reason": "Not applicable: requires ..."}],
  "not_priced": [{"charge_id": "samsa_levy", "section_refs": ["1.2"], "reason": "Rate set by external regulations"}],
  "on_request": [{"charge_id": "fresh_water_supply", "section_refs": ["4.5"], "reason": "Charged only when requested ..."}],
  "excluded": [{"charge_id": "cargo_dues_dry_bulk", "section_refs": ["7.2"], "reason": "Payable by the cargo owner, not the vessel."}],
  "failed": [],
  "total": "…",
  "warnings": [],
  "model": "…",
  "latency_ms": 4210
}
```

Each line item also carries `facts` (value, source, reason), `review_notes` and `open_issues`
from the rule's review, `confidence` (`low` when the rule was flagged) and `rule_source` (`cache`
or `compiled`).

Errors: `404 DOCUMENT_NOT_FOUND`, `409 DOCUMENT_NOT_READY`, `422 VALIDATION_ERROR`,
`422 PORT_REQUIRED`, `422 INSUFFICIENT_VESSEL_DATA`, `422 PORT_NOT_COVERED`, `422 UNKNOWN_CHARGE`,
`429 RATE_LIMITED`, `503 LLM_UNAVAILABLE`, `504 LLM_TIMEOUT`, `504 CALCULATION_TIMEOUT`. Every
calculation, failed or not, is stored (`GET /v1/calculations/{id}` returns it with its trace).

### Other routes

- `GET /v1/calculations/{id}`: the full record, including the `agent_step` trace.
- `GET /v1/calculations?port=&status=&limit=50&offset=0`: summaries, newest first.
- `POST /v1/documents` (multipart PDF): `202 {document_id, status}`, or `200` with the existing
  document for a duplicate checksum. `413` over `MAX_UPLOAD_MB`, `415` if it isn't a PDF.
- `GET /v1/documents`, `GET /v1/documents/{id}`: status and profile (authority, ports, currency,
  validity, counts).
- `GET /v1/documents/{id}/charges`: the charge catalogue.
- `GET /v1/documents/{id}/sections/{ref}`: section text, so citations can be checked.
- `POST /v1/rules/compile {document_id, port}`: compiles every candidate charge for a port (cache
  warm-up) and returns the rulebook.
- `GET /v1/rules?document_id=&port=`: the compiled rulebook, as JSON for human review.
- `GET /health` (process alive), `GET /ready` (database reachable; reports `documents_ready`.
  A system with no documents is still ready, because tariffs can be uploaded through the API).
- `GET /docs` (FastAPI's Swagger UI) is the supported UI for uploading PDFs and running
  calculations by hand; there is no separate frontend. It must work through Nginx. Declare
  `X-API-Key` as an `APIKeyHeader` security scheme so Swagger's Authorize button works when
  `API_AUTH_KEY` is set.

---

## 12. Concurrency, resilience, cost

1. One LLM semaphore per process, created at startup. Every chat and embedding call acquires it.
2. Every model call runs under `asyncio.timeout(LLM_TIMEOUT_SECONDS)`. The whole graph runs under
   `CALCULATION_TIMEOUT_SECONDS`, after which the response is `504` and a `failed` calculation row
   is written.
3. Retry only on 429 and 5xx, with exponential backoff and jitter, at most `LLM_MAX_RETRIES` times.
   Never retry a 400. A Structured Outputs refusal or a schema mismatch counts as a validation
   failure and uses the revision loop instead.
4. Charge sub-graphs run concurrently. One charge failing never fails the calculation (`partial`).
5. Nginx `proxy_read_timeout` = `CALCULATION_TIMEOUT_SECONDS + 15`.
6. **Expected cost and latency:** a cold port runs about 10 charges × (research ~3–5 calls +
   extract + critique) ≈ 50–70 calls, roughly 1–3 minutes. A warm port with JSON input makes one
   `resolve_facts` call per charge (batched into one call when possible) and takes seconds. Use
   `scripts/compile_rules.py` or `POST /v1/rules/compile` to warm the cache ahead of time.
7. Determinism: temperature 0, cached rules and versioned prompts. Two calls for the same vessel
   on a warm cache must return identical amounts.
8. Uploaded PDFs are untrusted input: size limit, MIME check, PyMuPDF only (nothing is executed),
   and prompt-injection framing (§10).

---

## 13. Validation case and expected accuracy

`eval/cases/sudestada_durban.json` holds the vessel profile (by file), `port: Durban`, a
plain-language query describing the same call, and the reference values, each mapped to the
tariff section of the line item it corresponds to (sections are stable across runs; catalogue ids
are not). That mapping lives in the eval case only, never in `app/`.

Hand-computing the TNPA rules for SUDESTADA (GT 51,300 → 513 units of 100 tons) gives:

| Reference item | Reference (ZAR) | Rule from the document | Computed (ZAR) | Δ |
|---|---|---|---|---|
| Light dues | 60,062.04 | §1.1.1 all other vessels: 513 × 117.08 | 60,062.04 | 0 |
| Port dues | 199,549.22 | §4.1.1: 513 × 192.73 + 513 × 57.79 × 3.39 days (pro rata) | 199,371.35 | −0.09% |
| Towage dues | 147,074.38 | §3.6 Durban band 50,001–100,000: (73,118.07 + 13 × 32.24) × 2 services | 147,074.38 | 0 |
| VTS dues | 33,315.75 | §2.1.1 Durban: 51,300 × 0.65 per GT | 33,345.00 | +0.09% |
| Pilotage dues | 47,189.94 | §3.3 Durban: (18,608.61 + 513 × 9.72) × 2 services | 47,189.94 | 0 |
| Running lines | 19,639.50 | §3.8 berthing services, other ports: (2,801.91 + 513 × 13.68) × 2 | 19,639.50 | 0 |

Known discrepancies. Document these in the README; **do not** code around them:

- **VTS:** 33,315.75 / 0.65 = 51,255 GT exactly, so the reference used GT 51,255, not the
  profile's 51,300. Every other charge is priced in whole 100-ton units, and both tonnages give 513
  units, so VTS (priced per single GT) is the only figure where the difference shows.
- **Port dues:** the reference implies 3.396 days alongside; the profile gives 3.39 (rounded).
- **Running lines:** the reference figure equals **§3.8 Berthing services** (mooring and unmooring),
  not §3.9 Running of vessel lines, which gives 2 × 1,654.56 = 3,309.12. The agent reports both as
  separate line items; the eval maps the reference label to §3.8.
- **Port dues duration:** the tariff charges port dues "from passing the entrance inwards until
  passing the entrance outwards". The reference uses days alongside (≈3.4) rather than the
  arrival→departure window (7.12 days, which would give 309,854.10). That is why
  `time_in_port_days` prefers `days_alongside` (§6.1). This ambiguity is the one most likely to hurt
  accuracy, so it gets a dedicated test.
- **Towage** is priced per service by tonnage band. The craft allocation table (3 tugs for this
  size) is not a multiplier; the critic prompt must not "fix" that. This is a general reading rule
  ("a fee per service is not multiplied by the number of craft unless the tariff says per tug").

`eval/run_eval.py` runs the live system (real OpenAI, cold cache by default, `--warm` to reuse
rules) on every case. It prints expected vs computed, absolute and percentage error and the
section used, writes `eval/report.json`, and exits non-zero if any item is outside the case
tolerance (1%). `make eval` runs it.

**Generalisation case:** `scripts/make_synthetic_tariff.py` renders a PDF for an invented port with
a different structure: unnumbered headings, tonnage bands keyed on NT instead of GT, marginal
tiers, a per-metre LOA charge, a "per tug" towage fee, and reductions with exclusivity. Its
expected values are computed by hand in `eval/cases/synthetic_port.yaml`. It must pass without any
change under `app/`.

---

## 14. Testing strategy

All tests run offline with `LLM_PROVIDER=fake` against a throwaway Postgres database, as in the
support-assistant project.

- **Engine (unit, the largest suite):** every component kind; `ceil` / `floor` / `pro_rata`; `above`
  offsets; band selection at exact boundaries; marginal tiers; min/max; multipliers; adjustment
  stacking and exclusive groups; exemptions; missing quantities; half-up rounding.
- **Golden rules:** `tests/fixtures/rules/durban/*.json` holds hand-written `ChargeRule`s for the
  six reference charges (test data only). The engine on SUDESTADA must reproduce the "Computed"
  column of §13 exactly. This test pins the math independently of the LLM.
- **Grounding:** number normalisation (`2  801.91`, `8  970.00`, `1 654.56`, `0.65`, `35%`), and
  rejection of a rule whose rate isn't in the cited text.
- **Ingestion (integration, real PDF, no LLM):** the section tree contains `3.6`; the towage table
  is a single markdown table with all port columns; running headers are stripped; re-ingesting is a
  no-op.
- **Retrieval:** `eval/retrieval_cases.json` (about 15 queries → expected section refs), run with
  real embeddings in `make eval-retrieval` and with the fake embedder in CI to check FTS + RRF
  wiring.
- **Graph (with `FakeClient` replaying recorded structured outputs):** cache hit skips research;
  an ungrounded rule triggers re-extraction; a critic `revise` loops and then stops at the limit;
  `not_applicable` and `not_priced` routing; one failing charge gives `partial`.
- **API:** request validation, both input forms, error codes, persistence of failed calculations.
- **`test_no_hardcoding.py`:** greps `app/` for the port names, charge names and rate literals that
  appear in `data/tariffs/` and the eval cases, and fails if any are found.

---

## 15. Docker, Compose, deployment

- **Dockerfile:** multi-stage, `python:3.12-slim`, uv, non-root user, `HEALTHCHECK` on `/health`,
  no dev dependencies, `data/tariffs/` copied in. Same shape as the support-assistant project.
- **compose.yaml:** `db` (`pgvector/pgvector:pg16`, healthcheck), `api` (healthcheck on `/ready`),
  `nginx` (`:8080`, `proxy_read_timeout` from §12, `client_max_body_size` = `MAX_UPLOAD_MB`).
  `compose.override.yaml` adds hot reload and exposes the DB port for local development.
- **Entrypoint:** `alembic upgrade head` → `python scripts/ingest.py --dir $TARIFFS_DIR`
  (idempotent) → uvicorn.
- **Makefile:** `up`, `down`, `test`, `lint`, `migrate`, `ingest`, `compile PORT=…`, `eval`,
  `eval-retrieval`, `logs`.
- **Bonus, public endpoint:** deploy the same image to a container host (Render, Railway or
  Fly.io) with managed Postgres that supports pgvector, or to a single VM running Compose. Set
  `API_AUTH_KEY` so the public URL can't spend the OpenAI budget, warm the Durban rulebook after
  deploying, and put the URL and a curl example in the README.

---

## 16. Build phases

Each phase ends at a **gate**. Show the verification output, then stop.

### Phase 0: Skeleton
pyproject (uv), app factory, `/health`, config, structlog, middleware, ruff, pytest, Makefile,
Dockerfile, compose with pgvector, `.env.example`.
**Gate:** `make lint test` is green; `docker compose up --build` → `curl localhost:8080/health` → 200.

### Phase 1: Data model
ORM models and the first Alembic migration (§7), including the `vector` extension and the indexes.
**Gate:** `alembic upgrade head` / `downgrade base` / `upgrade head` is reversible; `\dx` shows
`vector`.

### Phase 2: Rule DSL and engine (no LLM)
`domain/`, `rules/dsl.py`, `rules/engine.py`, `rules/grounding.py`, the golden Durban rules as
test fixtures, and `scripts/calculate.py --rules <dir>`.
**Gate:** the golden test reproduces the "Computed" column of §13 exactly; the engine unit suite
passes with ≥ 95% branch coverage on `rules/`.

### Phase 3: LLM layer
`LLMClient` (structured output, tool-calling, usage tracking), `Embedder`, OpenAI and fake
implementations, semaphore/timeout/retry, prompt registry with versions.
**Gate:** a smoke script runs a structured extraction against both clients; `LLM_PROVIDER=fake`
works with no API key.

### Phase 4: Ingestion
Parser, cleaner, structure, chunker, embeddings, FTS, document profile, `scripts/ingest.py`,
startup ingestion.
**Gate:** the TNPA PDF ingests to `ready`; re-running is a no-op; the integration tests from §14
pass; the profile lists the TNPA ports and ZAR / 15% VAT.

### Phase 5: Retrieval and charge catalogue
Hybrid search + RRF, section expansion, definition lookup, the catalogue step,
`GET /v1/documents/{id}/charges`.
**Gate:** `make eval-retrieval` recall@5 ≥ 0.9; the TNPA catalogue contains light dues, VTS,
pilotage, towage, berthing, running of lines, port dues and berth dues (payer vessel), plus cargo
dues (payer cargo).

### Phase 6: Rule-compilation agent
`research` → `extract_rule` → `validate_rule` → `critique`, the rule cache, `services/rulebook.py`,
`scripts/compile_rules.py --port Durban --out rulebook.json`, `POST /v1/rules/compile`,
`GET /v1/rules`.
**Gate:** the compiled Durban rulebook, run through the engine for SUDESTADA, matches the golden
fixtures' amounts for all six charges; every number is grounded; the exported JSON reads well to a
human.

### Phase 7: End-to-end calculation
`normalize_input` (JSON + NL), `resolve_document`, `screen_charges`, `resolve_facts`, `compute`,
`aggregate`, persistence, `POST /v1/calculations`, `GET /v1/calculations[/{id}]`.
**Gate:** `make eval` passes for SUDESTADA (cold and warm cache); an NL query describing the same
call returns the same amounts; a warm-cache call takes under 15 s.

### Phase 8: Documents API and observability
Upload with background ingestion, section route, `agent_step` trace, Langfuse spans per node,
`API_AUTH_KEY`.
**Gate:** upload a PDF through Swagger UI at `localhost:8080/docs` → `ready` → calculate; every log line for a request shares a
`request_id`; a calculation's trace shows the research tool calls and citations.

### Phase 9: Generalisation proof
Synthetic tariff generator, `eval/cases/synthetic_port.yaml`, `test_no_hardcoding.py`.
**Gate:** the synthetic case passes `make eval` with `git diff --stat app/` empty since the
Phase 8 commit; the guard test is green.

### Phase 10: Containerisation and README
Final Dockerfile, compose, nginx, entrypoint, and a README covering what it does, quick start
(Docker and local), CLI usage, architecture (linking `docs/architecture.md`), configuration table,
API reference with curl examples, the accuracy table and discrepancies from §13, design decisions,
and limitations.
**Gate:** `docker compose down -v && docker compose up --build` from a clean clone →
`curl localhost:8080/v1/calculations` with the SUDESTADA profile works; `make test lint` passes.

### Phase 11 (bonus): Public deployment
§15. **Gate:** the public URL answers `/health` and a keyed calculation request.

---

## 17. Definition of done

- [ ] Clean clone → `cp .env.example .env` → add the API key → `docker compose up --build` → working system
- [ ] SUDESTADA / Durban: all six reference charges within 1%, each with formula, citations, assumptions
- [ ] Synthetic port priced correctly with no code change
- [ ] No tariff data under `app/` (guard test green)
- [ ] `make test` passes offline; `make lint` passes
- [ ] Failed and partial calculations visible in the database and the API
- [ ] The compiled rulebook for a port is exportable and readable by a human
- [ ] README explains every design decision a reviewer might question, including why the LLM never
      calculates, why pgvector rather than a vector DB, why LangGraph here, and the reference-data
      discrepancies
- [ ] No secrets in git history

---

## 18. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Table parsing scrambles multi-column port tables | `find_tables()` keeps columns (verified on §3.3, §3.6, §3.8, §3.9 of the TNPA PDF); atomic table chunks; optional vision fallback; grounding rejects numbers that aren't in the text |
| Wrong port column ("other ports" vs a named port) | The extract prompt states the column-selection rule; `notes` records the column used; the critic checks it; eval catches regressions |
| Ambiguous quantities (time basis, number of services, per tug vs per service) | Explicit `Basis` definitions (§6.1), assumptions surfaced in every line item, request `overrides`, dedicated tests |
| LLM hallucinates a rate | Grounding validator, citations with quotes, critic, and cache writes only after approval |
| Missed charge | The catalogue is built from the whole outline; screening keeps uncertain charges; the eval checks the expected set |
| Non-determinism between runs | Temperature 0, versioned prompts, rule cache, golden-rule tests |
| Cold-cache latency and cost | Cache warm-up CLI and endpoint, concurrent fan-out, semaphore; cost per cold port logged |
| Prompt injection through an uploaded PDF | Excerpts framed as data, read-only tools, the LLM never executes anything or computes totals |
| A table PyMuPDF doesn't detect (e.g. TNPA §7.2 commodity rates) becomes number-heavy text that ranks poorly | The catalogue points each charge at its sections, and the agent reads whole sections; `PARSER_VISION_FALLBACK` can transcribe such pages |
| PyMuPDF is AGPL-licensed | Acceptable for a take-home; `ingestion/parser.py` is behind a `PdfParser` protocol, so `pdfplumber` (MIT) can replace it |

---

## 19. Notes for the implementing agent

- Prefer boring, readable code. Plain loops and explicit branches beat clever comprehensions; a
  human reviewer will read this.
- Type-annotate signatures. Pydantic models are the boundary types; don't pass raw dicts between
  layers.
- Build the engine first and make it boringly correct. Most accuracy bugs will look like LLM bugs
  but turn out to be unit or rounding bugs.
- When the eval misses a figure, find the general cause (prompt rule, DSL gap, retrieval miss),
  fix that, and add a test. Never special-case a port or charge.
- If a tariff mechanic genuinely doesn't fit the DSL, stop and propose a DSL extension with an
  example from the document rather than adding a free-form expression field.
- Verify the current OpenAI model ids and the current `langchain-openai` / `langgraph` versions
  against live documentation before pinning them.
- If a requirement here looks ambiguous or wrong, **stop and ask** rather than guessing.
- Keep each phase's diff reviewable. Do not refactor earlier phases while implementing a later one.
