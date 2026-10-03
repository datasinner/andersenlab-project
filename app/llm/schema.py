"""Pydantic model → JSON schema accepted by OpenAI Structured Outputs (strict).

Strict mode accepts a subset of JSON Schema: every object must list all its
properties as required and set additionalProperties to false, unions must be
anyOf (not oneOf/discriminator), and keywords such as default, title,
minLength and maxLength are rejected. Pydantic emits all of those, so this
module rewrites its schema. The model's own validation still runs on the
response, so constraints dropped here (min_length, ge, patterns on
Decimals, ...) are still enforced, just after generation instead of during.
"""

import copy
from typing import Any

from pydantic import BaseModel

# Keywords strict mode rejects or that only add noise to the prompt.
_DROPPED_KEYWORDS = {
    "default",
    "title",
    "minLength",
    "maxLength",
    "examples",
    "discriminator",
    "uniqueItems",
}


class UnsupportedSchemaError(ValueError):
    pass


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema(mode="validation")
    definitions = schema.pop("$defs", {})
    strict = _rewrite(schema, definitions)
    if definitions:
        strict["$defs"] = {name: _rewrite(node, definitions) for name, node in definitions.items()}
    if strict.get("type") != "object":
        raise UnsupportedSchemaError(f"{model.__name__}: the root of the schema must be an object")
    return strict


def _rewrite(node: Any, definitions: dict[str, Any]) -> Any:
    if isinstance(node, list):
        return [_rewrite(item, definitions) for item in node]
    if not isinstance(node, dict):
        return node

    node = dict(node)
    if "$ref" in node and len(node) > 1:
        # Strict mode doesn't allow siblings next to $ref (pydantic puts a
        # description there), so inline the referenced definition.
        reference = node.pop("$ref")
        node = {**copy.deepcopy(_resolve(reference, definitions)), **node}
    if "oneOf" in node:
        node["anyOf"] = node.pop("oneOf")
    if "allOf" in node and len(node["allOf"]) == 1:
        node = {**node.pop("allOf")[0], **node}

    rewritten: dict[str, Any] = {}
    for key, value in node.items():
        if key in _DROPPED_KEYWORDS:
            continue
        if key == "properties":
            rewritten[key] = {name: _rewrite(sub, definitions) for name, sub in value.items()}
        else:
            rewritten[key] = _rewrite(value, definitions)

    if rewritten.get("type") == "object":
        if rewritten.get("additionalProperties") not in (None, False):
            raise UnsupportedSchemaError(
                "free-form objects (dict fields) can't be used in strict structured output"
            )
        properties = rewritten.setdefault("properties", {})
        rewritten["required"] = list(properties)
        rewritten["additionalProperties"] = False
    return rewritten


def _resolve(reference: str, definitions: dict[str, Any]) -> dict[str, Any]:
    prefix = "#/$defs/"
    if not reference.startswith(prefix) or reference[len(prefix) :] not in definitions:
        raise UnsupportedSchemaError(f"unresolvable reference {reference}")
    return definitions[reference[len(prefix) :]]
