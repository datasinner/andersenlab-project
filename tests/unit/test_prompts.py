import json
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from app.domain.vessel import Basis
from app.llm.prompts import PROMPTS, PromptTemplate, versions
from app.llm.prompts.extract_rule import quantity_glossary

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_render_fills_both_templates():
    prompt = PromptTemplate(
        name="example", version="3", system="You price $what.", user='JSON stays {"a": 1}: $x'
    )
    system, user = prompt.render(what="dues", x="42")
    assert isinstance(system, SystemMessage)
    assert isinstance(user, HumanMessage)
    assert system.content == "You price dues."
    assert user.content == 'JSON stays {"a": 1}: 42'
    assert prompt.id == "example@3"


def test_render_refuses_to_leave_a_placeholder_unfilled():
    prompt = PromptTemplate(name="example", version="1", system="$a", user="$b")
    with pytest.raises(KeyError):
        prompt.render(a="only one")


def test_registry_ids_and_combined_versions():
    assert PROMPTS["extract_rule"].id == "extract_rule@1"
    assert versions("extract_rule") == "extract_rule@1"


def test_extract_rule_prompt_lists_every_quantity():
    system, _ = PROMPTS["extract_rule"].render(
        quantities=quantity_glossary(), port="P", charge_name="C", currency="XTS", excerpts="E"
    )
    for basis in Basis:
        assert f"- {basis.value}:" in system.content


def test_prompts_contain_no_knowledge_of_the_validation_tariff():
    """Prompts teach general tariff reading; facts about the TNPA document
    (ports, authority, section numbers, rates) must not leak into them."""
    excerpts = json.loads((FIXTURES / "tnpa_excerpts.json").read_text())
    rates = {"117.08", "0.65", "18 608.61", "73 118.07", "2  801.91", "192.73", "57.79"}
    forbidden = {"Durban", "Transnet", "TNPA", "Saldanha", "Richards Bay", "SAMSA", *rates}
    assert excerpts  # the fixture the rates come from
    for prompt in PROMPTS.values():
        text = prompt.system + prompt.user
        assert not {term for term in forbidden if term in text}, prompt.id
