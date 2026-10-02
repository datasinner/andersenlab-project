from typing import Any

import pytest
from pydantic import BaseModel, Field, RootModel

from app.llm.schema import UnsupportedSchemaError, strict_json_schema
from app.rules.dsl import ChargeRule

FORBIDDEN = {"oneOf", "discriminator", "default", "title", "minLength", "maxLength"}


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def test_charge_rule_schema_meets_strict_mode_rules():
    schema = strict_json_schema(ChargeRule)

    assert schema["type"] == "object"
    for node in _walk(schema):
        assert not FORBIDDEN & node.keys(), FORBIDDEN & node.keys()
        if "$ref" in node:
            assert len(node) == 1, "strict mode allows no siblings next to $ref"
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert node["required"] == list(node["properties"])


def test_component_union_becomes_any_of():
    components = strict_json_schema(ChargeRule)["properties"]["components"]
    assert "anyOf" in components["items"]
    assert len(components["items"]["anyOf"]) == 4


def test_amounts_are_strings_with_guidance():
    definitions = strict_json_schema(ChargeRule)["$defs"]
    fixed_fee = definitions["FixedFee"]["properties"]["amount"]
    assert fixed_fee["type"] == "string"
    assert "exactly as printed" in fixed_fee["description"]


class _WithDescribedReference(BaseModel):
    inner: "_Inner" = Field(description="The inner part")


class _Inner(BaseModel):
    value: int


def test_reference_with_siblings_is_inlined():
    schema = strict_json_schema(_WithDescribedReference)
    inner = schema["properties"]["inner"]
    assert "$ref" not in inner
    assert inner["description"] == "The inner part"
    assert inner["properties"]["value"]["type"] == "integer"


class _WithDict(BaseModel):
    extra: dict[str, Any]


def test_free_form_objects_are_rejected():
    with pytest.raises(UnsupportedSchemaError, match="free-form objects"):
        strict_json_schema(_WithDict)


def test_root_must_be_an_object():
    with pytest.raises(UnsupportedSchemaError, match="root of the schema must be an object"):
        strict_json_schema(RootModel[list[int]])
