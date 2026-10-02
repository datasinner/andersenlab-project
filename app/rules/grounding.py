"""Grounding: every number in a rule must appear in the text it cites.

This is the cheap, deterministic guard against a model inventing or
mistyping a rate. It checks:

- every cited chunk exists;
- every citation quote appears in its chunk: the same words in the same order,
  ignoring case, punctuation and typography (curly quotes, dashes, dot leaders);
- every number in the rule (fees, rates, band bounds, tier bounds, unit
  sizes, offsets, minimum/maximum, percentages, numeric condition values)
  appears somewhere in the cited chunks.

0 and 1 are skipped: they are structural ("from zero", "per unit") and need
not be printed.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

from app.domain.numbers import NumberFormatError, extract_numbers, format_number, parse_number
from app.rules.dsl import (
    BandedFee,
    ChargeRule,
    Citation,
    Condition,
    FixedFee,
    PerUnitFee,
    TieredFee,
    Units,
)

_STRUCTURAL_NUMBERS = {Decimal(0), Decimal(1)}
_MARKDOWN_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
_WORDS = re.compile(r"[^\W_]+")


@dataclass(frozen=True)
class GroundingIssue:
    path: str
    message: str


def check_grounding(rule: ChargeRule, chunk_texts: Mapping[int, str]) -> list[GroundingIssue]:
    issues: list[GroundingIssue] = []

    cited_texts: list[str] = []
    for path, citation in _citations(rule):
        text = chunk_texts.get(citation.chunk_id)
        if text is None:
            issues.append(GroundingIssue(path, f"cites unknown chunk {citation.chunk_id}"))
            continue
        cited_texts.append(text)
        if not quote_in(citation.quote, text):
            issues.append(GroundingIssue(path, f"quote not found in chunk {citation.chunk_id}"))

    printed = extract_numbers("\n".join(cited_texts))
    for path, value in _numbers(rule):
        if value in _STRUCTURAL_NUMBERS or value in printed:
            continue
        issues.append(
            GroundingIssue(path, f"{format_number(value)} does not appear in the cited text")
        )
    return issues


def quote_in(quote: str, text: str) -> bool:
    """Whether `quote` appears in `text`: the same words in the same order,
    ignoring case, punctuation and typography."""
    return _normalize(quote) in _normalize(text)


def _normalize(text: str) -> str:
    words = _WORDS.findall(_MARKDOWN_BREAK.sub(" ", text).casefold())
    return " " + " ".join(words) + " "


def _citations(rule: ChargeRule) -> list[tuple[str, Citation]]:
    found = [(f"citations[{i}]", citation) for i, citation in enumerate(rule.citations)]
    for i, exemption in enumerate(rule.exemptions):
        if exemption.citation is not None:
            found.append((f"exemptions[{i}].citation", exemption.citation))
    for i, adjustment in enumerate(rule.adjustments):
        if adjustment.citation is not None:
            found.append((f"adjustments[{i}].citation", adjustment.citation))
    return found


def _numbers(rule: ChargeRule) -> list[tuple[str, Decimal]]:
    found: list[tuple[str, Decimal]] = []
    if rule.minimum is not None:
        found.append(("minimum", rule.minimum))
    if rule.maximum is not None:
        found.append(("maximum", rule.maximum))
    if rule.multiplier is not None:
        found.extend(_unit_numbers("multiplier", rule.multiplier))

    for i, component in enumerate(rule.components):
        path = f"components[{i}]"
        found.extend(_condition_numbers(f"{path}.when", component.when))
        if isinstance(component, FixedFee):
            found.append((f"{path}.amount", component.amount))
        elif isinstance(component, PerUnitFee):
            found.append((f"{path}.rate", component.rate))
            found.extend(_unit_numbers(f"{path}.units", component.units))
            if component.per_time is not None:
                found.extend(_unit_numbers(f"{path}.per_time", component.per_time))
        elif isinstance(component, BandedFee):
            for j, band in enumerate(component.bands):
                band_path = f"{path}.bands[{j}]"
                found.append((f"{band_path}.lower", band.lower))
                if band.upper is not None:
                    found.append((f"{band_path}.upper", band.upper))
                found.append((f"{band_path}.base_fee", band.base_fee))
                if band.increment is not None:
                    found.append((f"{band_path}.increment.rate", band.increment.rate))
                    found.extend(
                        _unit_numbers(f"{band_path}.increment.units", band.increment.units)
                    )
        elif isinstance(component, TieredFee):
            for j, tier in enumerate(component.tiers):
                tier_path = f"{path}.tiers[{j}]"
                if tier.up_to is not None:
                    found.append((f"{tier_path}.up_to", tier.up_to))
                if tier.width is not None:
                    found.append((f"{tier_path}.width", tier.width))
                found.append((f"{tier_path}.rate", tier.rate))
                found.append((f"{tier_path}.unit_size", tier.unit_size))

    found.extend(_condition_numbers("applies_when", rule.applies_when))
    for i, exemption in enumerate(rule.exemptions):
        found.extend(_condition_numbers(f"exemptions[{i}].when", exemption.when))
    for i, adjustment in enumerate(rule.adjustments):
        found.append((f"adjustments[{i}].percent", adjustment.percent))
        found.extend(_condition_numbers(f"adjustments[{i}].when", adjustment.when))
    return found


def _unit_numbers(path: str, units: Units) -> list[tuple[str, Decimal]]:
    return [(f"{path}.unit_size", units.unit_size), (f"{path}.above", units.above)]


def _condition_numbers(path: str, conditions: list[Condition]) -> list[tuple[str, Decimal]]:
    found: list[tuple[str, Decimal]] = []
    for i, condition in enumerate(conditions):
        values = condition.value if isinstance(condition.value, list) else [condition.value]
        for value in values:
            if isinstance(value, bool):
                continue
            try:
                found.append((f"{path}[{i}].value", parse_number(value)))
            except NumberFormatError:
                continue  # text values are not numbers to ground
    return found
