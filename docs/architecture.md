# Architecture

The guiding rule: **the LLM reads, the code calculates.** An agent finds and interprets tariff
rules in the document and writes them down as typed, cited data. A deterministic engine turns that
data into money. Nothing about any specific port or charge lives in the code.

## 1. System overview

```mermaid
flowchart LR
  client["API client / CLI<br/>vessel profile or NL query"]
  reviewer["Reviewer<br/>rulebooks · agent traces"]
  openai["OpenAI API<br/>chat + embeddings"]
  langfuse["Langfuse (optional)<br/>traces · tokens · cost"]

  subgraph compose["Docker Compose deployment"]
    direction LR
    nginx["Nginx :8080<br/>reverse proxy"]

    subgraph api["API container :8000"]
      direction TB
      entry["Entrypoint<br/>migrate + ingest bundled tariffs"]
      routes["FastAPI routers<br/>documents · rules · calculations · health"]
      services["Services<br/>documents · rulebook · calculations"]
      ingestion["Ingestion pipeline<br/>parse · structure · chunk · embed · catalogue"]
      agent["Calculation agent<br/>LangGraph"]
      retrieval["Hybrid retrieval<br/>pgvector + full-text + RRF"]
      engine["Rule engine<br/>Decimal · no LLM"]
      llm["LLM client<br/>structured outputs · semaphore · retry"]

      routes --> services
      services --> ingestion
      services --> agent
      agent --> retrieval
      agent --> engine
      agent --> llm
      ingestion --> llm
    end

    pdfs[("data/tariffs/*.pdf")]
    pg[("PostgreSQL 16 + pgvector<br/>documents · sections · chunks · charges<br/>compiled rules · calculations · agent steps")]
  end

  client -->|"JSON REST /v1/*"| nginx --> routes
  reviewer -->|"GET /v1/rules · /v1/calculations/{id}"| nginx
  pdfs -. "startup" .-> entry -. "idempotent ingest" .-> ingestion
  ingestion --> pg
  retrieval --> pg
  services <--> pg
  llm -->|"async HTTPS"| openai
  llm -.-> langfuse
```

Solid arrows show the request path; dotted arrows show startup work. Postgres is the only
datastore: pgvector for semantic search, a generated `tsvector` column for lexical search, and
JSONB for compiled rules and traces.

## 2. Module layering

```mermaid
flowchart TD
  routers["routers/"] --> services["services/"]
  services --> agent["agent/<br/>graph · nodes · tools"]
  services --> ingestion["ingestion/"]
  agent --> retrieval["retrieval/"]
  agent --> rules["rules/<br/>dsl · engine · grounding"]
  agent --> llm["llm/<br/>client · embeddings · prompts"]
  ingestion --> llm
  rules --> domain["domain/<br/>VesselCall · numbers"]
  agent --> domain
  retrieval --> db[("db / models")]
  services --> db
```

`domain/` and `rules/` are pure Python, with no I/O and no LLM. They hold all of the arithmetic
and are the most heavily tested part of the system.

## 3. Ingestion: from PDF to a searchable, structured tariff

```mermaid
flowchart LR
  pdf["Tariff PDF"] --> reg["Register<br/>sha256 → idempotent"]
  reg --> parse["Parse (PyMuPDF)<br/>text spans + fonts<br/>find_tables() → markdown"]
  parse --> clean["Clean<br/>strip running headers/footers<br/>fix hyphenation"]
  clean --> tree["Section tree<br/>numbering + font cues<br/>LLM outline fallback"]
  tree --> chunk["Chunk<br/>per section · tables atomic<br/>breadcrumb prefix"]
  chunk --> index["Embed + tsvector"]
  tree --> profile["LLM: document profile<br/>authority · ports · currency<br/>VAT · validity · conventions"]
  tree --> catalogue["LLM: charge catalogue<br/>name · sections · payer · trigger"]
  index --> db[("Postgres + pgvector")]
  profile --> db
  catalogue --> db
```

Two design points matter most:

- **Tables stay whole and keep their context.** A tariff table only makes sense together with its
  heading and the sentence that introduces it ("Per service based on vessel's tonnage:"), so each
  table chunk carries both. PyMuPDF's `find_tables()` keeps the port columns of the TNPA tables
  intact; plain text extraction scrambles them.
- **The charge catalogue is discovered, not configured.** One LLM pass over the outline lists every
  charge the document defines, who pays it, and what triggers it. That list drives the calculation,
  so a document with charges we've never seen (e.g. "waste reception fee", "ISPS dues") just works.

## 4. Calculation agent

```mermaid
flowchart TD
  start(["POST /v1/calculations"]) --> norm["normalize_input<br/>profile JSON and/or NL query → VesselCall"]
  norm --> doc["resolve_document<br/>document covering port + arrival date"]
  doc --> screen["screen_charges<br/>catalogue → candidates<br/>excluded · not applicable (with reason)"]
  screen -->|"Send() · one branch per charge"| load

  subgraph per_charge["Charge sub-graph (runs concurrently for each candidate)"]
    direction TB
    load{"load_rule<br/>cached for<br/>document + port?"}
    research["research (ReAct)<br/>search_tariff · read_section<br/>lookup_definition · list_sections"]
    extract["extract_rule<br/>Structured Outputs → ChargeRule<br/>every number cited"]
    validate{"validate_rule<br/>schema + grounding"}
    facts["resolve_facts<br/>quantities: code<br/>qualitative facts: LLM"]
    compute["compute<br/>rule engine · Decimal"]
    critique{"critique<br/>LLM reviews rule + trace<br/>against the sources"}
    result["line item · not applicable · not priced"]

    load -->|"miss"| research --> extract --> validate
    validate -->|"invalid / ungrounded<br/>≤ N revisions"| extract
    validate -->|"ok"| facts
    load -->|"hit"| facts
    facts --> compute --> critique
    critique -->|"revise · ≤ N revisions"| extract
    critique -->|"approved → write rule to cache"| result
  end

  result --> agg["aggregate<br/>order · total · warnings<br/>persist calculation + agent steps"]
  agg --> done(["response with line items,<br/>formulas, citations, assumptions"])
```

On a cache hit the critic is skipped by default (`AGENT_CRITIC_ON_CACHE_HIT=false`), because the
rule was already approved. Facts are still resolved per call, since they depend on the vessel. When
revisions run out, the item is kept and flagged `low_confidence` with the critic's issues, never
silently dropped.

### Why this is agentic RAG and not a fixed pipeline

- **Retrieval is driven by the agent.** In `research`, the model decides what to search for, which
  sections to open, and which definitions it needs (e.g. what "tonnage" means for this document)
  before it commits to a rule.
- **It checks its own work.** Grounding (code) and the critic (LLM) can send a rule back for
  re-extraction with concrete feedback.
- **The set of charges comes from the document.** Discovery and screening decide what to compute;
  the code has no list of charges.

## 5. One request, cold vs warm cache

```mermaid
sequenceDiagram
  autonumber
  participant C as Client
  participant A as API / services
  participant G as Agent graph
  participant R as Retrieval (pgvector + FTS)
  participant L as OpenAI
  participant E as Rule engine
  participant D as Postgres

  C->>A: POST /v1/calculations (vessel, port)
  A->>G: run(VesselCall)
  G->>D: document + catalogue for port
  G->>L: screen_charges
  par for each candidate charge
    G->>D: load compiled rule
    alt cache miss
      loop research (≤ AGENT_MAX_TOOL_CALLS)
        G->>L: next tool call?
        G->>R: search_tariff / read_section / lookup_definition
        R->>D: vector + full-text query
      end
      G->>L: extract_rule (Structured Outputs)
      G->>G: validate_rule (schema + grounding)
    end
    G->>L: resolve_facts
    G->>E: compute(rule, quantities, facts)
    E-->>G: amount + trace
    opt cache miss
      G->>L: critique
      G->>D: save approved rule
    end
  end
  G->>D: save calculation + agent steps
  G-->>A: result
  A-->>C: 200 line items + citations
```

## 6. Worked example: how the agent finds and prices towage at Durban

This shows what each step does with the real TNPA document. The numbers below appear in this doc
and in test fixtures only, never in `app/`.

1. **Discovery.** At ingestion, the catalogue lists `tugs_vessel_assistance` → §3.6, payer
   `vessel`, trigger `per_service`.
2. **Screening.** A bulk carrier entering and leaving Durban → `candidate`.
3. **Research.** `search_tariff("tug assistance fee per service vessel tonnage")` → the §3.6 table
   chunk ranks first. `read_section("3.6")` returns the craft allocation table, the fee table with
   one column per port, and the surcharge bullets. `lookup_definition("tonnage")` returns "gross
   tonnage as per the Tonnage Convention 1969". `read_section("3.1")` returns the ordinary working
   hours (Durban: 24 hours).
4. **Extraction.** Durban has its own column. The model emits a banded rule (abridged):

   ```json
   {
     "charge_id": "tugs_vessel_assistance",
     "name": "Tugs/vessel assistance and/or attendance",
     "section_refs": ["3.6"],
     "port_key": "Durban",
     "payer": "vessel",
     "status": "priced",
     "components": [{
       "kind": "banded", "id": "towage_fee", "label": "Fee per service by vessel tonnage",
       "basis": "gross_tonnage",
       "bands": [
         {"lower": "50000", "upper": "100000", "base_fee": "73118.07",
          "increment": {"kind": "per_unit", "id": "inc", "label": "per 100 tons above 50 000",
                        "rate": "32.24",
                        "units": {"basis": "gross_tonnage", "unit_size": "100",
                                  "rounding": "ceil", "above": "50000"}}},
         {"lower": "100000", "upper": null, "base_fee": "93548.13", "increment": {"…": "…"}}
       ]
     }],
     "multiplier": {"basis": "num_services", "unit_size": "1", "rounding": "ceil"},
     "adjustments": [
       {"id": "outside_hours", "kind": "surcharge", "percent": "25", "applies_to": "all",
        "when": [{"fact": "service_outside_ordinary_working_hours", "op": "eq", "value": true}]}
     ],
     "facts": [{"name": "service_outside_ordinary_working_hours", "type": "bool",
                "description": "Service commences or terminates outside ordinary working hours (§3.1)",
                "default_assumption": "false"}],
     "notes": ["Fee is per service by tonnage band; the craft allocation table limits the number of tugs and is not a multiplier."]
   }
   ```

5. **Validation.** `73 118.07`, `32.24`, `50 000`, `100 000`, `93 548.13` and `25%` are all found in
   the cited chunks after normalisation, so the rule is grounded.
6. **Facts.** Quantities: `gross_tonnage = 51 300`, `num_services = 2` (assumption: one entering
   and one leaving movement). The surcharge fact resolves to `false`: §3.1 says Durban runs a
   24-hour service, so no service falls outside ordinary hours.
7. **Compute.** Band 50 000–100 000 → `ceil((51 300 − 50 000) / 100) = 13` → `13 × 32.24 = 419.12`
   → `73 118.07 + 419.12 = 73 537.19` per service → `× 2 = 147 074.38`.
8. **Critique.** Checks the column (Durban, not "other"), the band, the rounding, and that the fee
   is per service rather than per tug → `approved`, and the rule is cached for (TNPA 2024/25,
   Durban).

The next Durban calculation reuses the rule and goes straight to step 6.

## 7. Data model

```mermaid
erDiagram
  tariff_document ||--o{ document_section : has
  document_section ||--o{ document_section : parent_of
  document_section ||--o{ chunk : contains
  tariff_document ||--o{ charge_catalogue_entry : defines
  tariff_document ||--o{ compiled_rule : "compiled for port"
  tariff_document ||--o{ calculation : priced_by
  calculation ||--o{ agent_step : traced_by

  tariff_document {
    uuid id PK
    text title
    text authority
    char checksum UK
    jsonb ports
    text currency
    numeric vat_rate
    text status
  }
  document_section {
    int id PK
    text ref
    text title
    text path
    int level
    int page_start
  }
  chunk {
    int id PK
    text kind
    text content
    int page
    vector embedding
    tsvector tsv
  }
  charge_catalogue_entry {
    int id PK
    text charge_id
    text payer
    text trigger
    jsonb section_refs
  }
  compiled_rule {
    int id PK
    text port_key
    text charge_id
    jsonb rule
    text prompt_version
    jsonb critic_verdict
  }
  calculation {
    uuid id PK
    text port_key
    jsonb vessel_call
    jsonb result
    numeric total
    text status
  }
  agent_step {
    int id PK
    text charge_id
    text node
    text tool
    jsonb output
    int latency_ms
  }
```

## 8. How this meets the brief

| Brief asks for | Where it's handled |
|---|---|
| Consume a complex tariff PDF | Ingestion: section tree, atomic tables, hybrid index (§3) |
| Natural-language vessel query | `normalize_input` accepts NL, structured JSON, or both |
| All applicable charges | Charge catalogue + screening, with explicit `not_applicable` / `not_priced` / `excluded` lists |
| Accuracy | Deterministic `Decimal` engine, grounding checks, critic loop, golden-rule tests, live eval |
| Other ports and document updates without code changes | DSL of general tariff mechanics; rules compiled per (document checksum, port); synthetic-port eval; no-hardcoding guard test |
| Shows how the agent finds and interprets rules | Citations on every line item, an exportable rulebook, the `agent_step` trace, Langfuse spans |
| OpenAI models | All chat and embedding calls go through `langchain-openai` |
| Senior-level code | Layered modules, typed contracts, error taxonomy, resilience policy, offline tests |
| Bonus: API endpoint | FastAPI behind Nginx in Compose; public deployment with an API key (BUILD_PLAN §15) |
