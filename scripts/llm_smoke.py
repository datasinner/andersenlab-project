"""Smoke test for the LLM layer: one structured extraction and one embedding call.

Extracts a ChargeRule from an invented tariff excerpt with the extract_rule
prompt, checks it is grounded in the excerpt, prices a sample vessel with
the engine, then embeds three sentences and compares their similarity.

    uv run python scripts/llm_smoke.py          # OpenAI (needs OPENAI_API_KEY)
    uv run python scripts/llm_smoke.py --fake   # offline, scripted fake client
"""

import argparse
import asyncio
import math
import sys
from decimal import Decimal

from app.domain.numbers import format_money
from app.domain.vessel import VesselCall, resolve_quantities
from app.llm.client import FakeLLMClient, LLMClient, LLMOutputError, OpenAIClient
from app.llm.embeddings import Embedder, FakeEmbedder, OpenAIEmbedder
from app.llm.prompts import PROMPTS
from app.llm.prompts.extract_rule import quantity_glossary
from app.llm.resilience import LLMError, ResiliencePolicy
from app.rules.dsl import ChargeRule
from app.rules.engine import evaluate_rule
from app.rules.grounding import check_grounding

CHUNK_ID = 7
EXCERPT = """\
2.1 PILOTAGE DUES - PORT OF EXAMPLEVILLE
Pilotage is compulsory. Fees are payable per service (entering or leaving the port).
Basic fee per service .................. 4 250.00
Plus per 100 gross tons or part thereof .......... 6.15
A surcharge of 30% is payable if the service commences or terminates between 22:00 and 06:00.
Vessels of the national navy are exempt."""

# (4 250.00 + ceil(51 300 / 100) × 6.15) × 2 services
EXPECTED_AMOUNT = Decimal("14809.90")

# What a correct extraction looks like; the fake client replays it.
FAKE_RULE = {
    "charge_id": "pilotage_dues",
    "name": "Pilotage dues",
    "section_refs": ["2.1"],
    "port_key": "Exampleville",
    "currency": "XTS",
    "payer": "vessel",
    "status": "priced",
    "exemptions": [
        {
            "description": "Vessels of the national navy",
            "when": [{"fact": "is_national_navy_vessel", "op": "eq", "value": True}],
        }
    ],
    "components": [
        {
            "kind": "fixed",
            "id": "basic_fee",
            "label": "Basic fee per service",
            "amount": "4 250.00",
        },
        {
            "kind": "per_unit",
            "id": "tonnage_fee",
            "label": "Per 100 gross tons or part thereof",
            "rate": "6.15",
            "units": {"basis": "gross_tonnage", "unit_size": "100", "rounding": "ceil"},
        },
    ],
    "multiplier": {"basis": "num_services", "unit_size": "1", "rounding": "ceil"},
    "adjustments": [
        {
            "id": "night_service",
            "kind": "surcharge",
            "description": "Service commences or terminates between 22:00 and 06:00",
            "percent": "30",
            "when": [{"fact": "service_at_night", "op": "eq", "value": True}],
        }
    ],
    "facts": [
        {
            "name": "is_national_navy_vessel",
            "type": "bool",
            "description": "Vessel of the national navy",
            "default_value": False,
        },
        {
            "name": "service_at_night",
            "type": "bool",
            "description": "Service commences or terminates between 22:00 and 06:00",
            "default_value": False,
        },
    ],
    "citations": [
        {"chunk_id": CHUNK_ID, "section_ref": "2.1", "page": 3, "quote": "Basic fee per service"}
    ],
}

SENTENCES = [
    "Pilotage fee payable per service for vessels entering the port",
    "Charges for a pilot when a ship enters or leaves the harbour",
    "Fresh water supplied to vessels is quoted on application",
]


async def main(use_fake: bool) -> int:
    policy = ResiliencePolicy.from_settings()
    llm: LLMClient
    embedder: Embedder
    if use_fake:
        fake = FakeLLMClient()
        fake.script("smoke_extract_rule", FAKE_RULE)
        llm, embedder = fake, FakeEmbedder()
    else:
        llm, embedder = OpenAIClient(policy), OpenAIEmbedder(policy)

    prompt = PROMPTS["extract_rule"]
    print(f"Chat model: {llm.model_name} | prompt {prompt.id}")
    messages = prompt.render(
        quantities=quantity_glossary(),
        port="Exampleville",
        charge_name="Pilotage dues",
        currency="XTS",
        research_notes="(none)",
        feedback="",
        excerpts=f'<tariff_excerpt chunk_id="{CHUNK_ID}" section="2.1" page="3">\n'
        f"{EXCERPT}\n</tariff_excerpt>",
    )
    try:
        result = await llm.generate_structured(ChargeRule, messages, name="smoke_extract_rule")
    except LLMOutputError as exc:
        print(f"Extraction failed: {exc}\n  " + "\n  ".join(exc.errors), file=sys.stderr)
        return 1
    except LLMError as exc:
        print(f"Model call failed: {exc}", file=sys.stderr)
        return 1
    rule = result.value
    print(
        f"Extracted '{rule.name}' in {result.latency_ms} ms "
        f"({result.usage.prompt_tokens} prompt + {result.usage.completion_tokens} completion "
        "tokens)"
    )
    print(f"  components: {[component.kind + ':' + component.id for component in rule.components]}")
    print(f"  multiplier: {rule.multiplier.basis.value if rule.multiplier else None}")
    print(f"  adjustments: {[(a.kind, str(a.percent)) for a in rule.adjustments]}")
    print(f"  exemptions: {[exemption.description for exemption in rule.exemptions]}")

    issues = check_grounding(rule, {CHUNK_ID: EXCERPT})
    print("Grounding: " + ("OK" if not issues else f"{len(issues)} issue(s)"))
    for issue in issues:
        print(f"  - {issue.path}: {issue.message}")

    vessel = VesselCall.from_profile({"vessel": {"gross_tonnage": 51300}}, port="Exampleville")
    item = evaluate_rule(rule, resolve_quantities(vessel))
    verdict = "matches" if item.amount == EXPECTED_AMOUNT else "DIFFERS FROM"
    print(
        f"Engine, GT 51,300 and 2 services: {format_money(item.amount or Decimal(0))} "
        f"({verdict} the expected {format_money(EXPECTED_AMOUNT)})"
    )
    for line in item.formula:
        print(f"  {line}")

    vectors = await embedder.embed(SENTENCES, name="smoke_embed")
    print(f"Embedding model: {embedder.model_name} | {len(vectors)} vectors of {len(vectors[0])}")
    print(f"  similar sentences:   {_cosine(vectors[0], vectors[1]):.3f}")
    print(f"  unrelated sentences: {_cosine(vectors[0], vectors[2]):.3f}")
    return 0 if not issues else 1


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    return dot / (math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fake", action="store_true", help="use the offline fake clients")
    raise SystemExit(asyncio.run(main(parser.parse_args().fake)))
