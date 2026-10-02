"""Deterministic evaluation of a ChargeRule for one vessel call.

No LLM, no I/O: given a rule, the call's quantities and its resolved facts,
produce a LineItem with the amount and a step-by-step formula trace.

Order of evaluation (BUILD_PLAN §6.3):
  1. a rule that isn't priced passes its status through;
  2. a matching exemption, or a failed applies_when condition, makes the
     charge not applicable;
  3. each applicable component is evaluated in full Decimal precision;
  4. components are summed, clamped to minimum/maximum, then multiplied;
  5. applicable adjustments are added (largest one per exclusive group),
     each as a percentage of the amount it targets;
  6. the result is rounded half-up to cents.
"""

import math
from collections.abc import Mapping
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, Field

from app.domain.numbers import NumberFormatError, format_number, parse_number, round_money
from app.domain.vessel import Basis, ResolvedQuantities
from app.rules.dsl import (
    Adjustment,
    BandedFee,
    ChargeRule,
    Citation,
    Component,
    Condition,
    FactSpec,
    FixedFee,
    Operator,
    PerUnitFee,
    Rounding,
    TieredFee,
    Units,
)

FactValue = bool | Decimal | str

_HUNDRED = Decimal(100)
_BASIS_NAMES = {basis.value for basis in Basis}


class RuleEvaluationError(Exception):
    """The rule can't be evaluated for this call as written."""


class MissingQuantityError(RuleEvaluationError):
    def __init__(self, basis: Basis) -> None:
        self.basis = basis
        super().__init__(f"The vessel call has no value for '{basis.value}'")


class MissingFactError(RuleEvaluationError):
    def __init__(self, fact: str) -> None:
        self.fact = fact
        super().__init__(f"Fact '{fact}' was not resolved and has no default")


class LineItemStatus(StrEnum):
    CHARGED = "charged"
    NOT_APPLICABLE = "not_applicable"
    ON_APPLICATION = "on_application"
    NOT_PRICED = "not_priced_in_document"


class AppliedAdjustment(BaseModel):
    id: str
    kind: str
    description: str
    percent: Decimal
    amount: Decimal  # signed: negative for reductions


class LineItem(BaseModel):
    charge_id: str
    name: str
    section_refs: list[str]
    currency: str
    status: LineItemStatus
    amount: Decimal | None = None
    formula: list[str] = Field(default_factory=list)
    adjustments: list[AppliedAdjustment] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    reason: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def evaluate_rule(
    rule: ChargeRule,
    quantities: ResolvedQuantities,
    facts: Mapping[str, FactValue | str] | None = None,
) -> LineItem:
    return _Evaluation(rule, quantities, facts or {}).run()


class _Evaluation:
    def __init__(
        self,
        rule: ChargeRule,
        quantities: ResolvedQuantities,
        facts: Mapping[str, FactValue | str],
    ) -> None:
        self.rule = rule
        self.quantities = quantities
        self.facts = facts
        self.formula: list[str] = []
        self.assumptions: list[str] = []

    # -- orchestration -----------------------------------------------------

    def run(self) -> LineItem:
        rule = self.rule
        if rule.status == "on_application":
            return self._result(
                LineItemStatus.ON_APPLICATION, reason=self._notes_or("Quoted on application.")
            )
        if rule.status == "not_priced_in_document":
            return self._result(
                LineItemStatus.NOT_PRICED,
                reason=self._notes_or("The document does not state a rate for this charge."),
            )
        if rule.status == "not_applicable_at_port":
            return self._result(
                LineItemStatus.NOT_APPLICABLE,
                reason=self._notes_or(f"Not charged at {rule.port_key}."),
            )

        for exemption in rule.exemptions:
            if self._all_hold(exemption.when):
                citations = [exemption.citation] if exemption.citation else rule.citations
                return self._result(
                    LineItemStatus.NOT_APPLICABLE,
                    reason=f"Exempt: {exemption.description}",
                    citations=citations,
                )
        for condition in rule.applies_when:
            if not self._holds(condition):
                return self._result(
                    LineItemStatus.NOT_APPLICABLE,
                    reason=f"Not applicable: requires {self._describe(condition)}",
                )

        component_amounts: dict[str, Decimal] = {}
        for component in rule.components:
            if not self._all_hold(component.when):
                self.formula.append(f"{component.label}: not applicable to this call")
                continue
            component_amounts[component.id] = self._component(component)
        if not component_amounts:
            return self._result(
                LineItemStatus.NOT_APPLICABLE,
                reason="No pricing component of this charge applies to the vessel call",
            )

        amount = sum(component_amounts.values(), Decimal(0))
        if len(component_amounts) > 1:
            self.formula.append(f"Subtotal: {format_number(amount)}")
        amount = self._clamp(amount)

        factor = Decimal(1)
        if rule.multiplier is not None:
            factor, _ = self._units(rule.multiplier)
            amount *= factor
            self.formula.append(
                f"× {format_number(factor)} ({rule.multiplier.basis.value}) = "
                f"{format_number(amount)}"
            )

        applied = self._adjustments(amount, component_amounts, factor)
        for adjustment in applied:
            amount += adjustment.amount
        if applied:
            amount = max(amount, Decimal(0))
            self.formula.append(f"After adjustments: {format_number(amount)}")

        total = round_money(amount)
        self.formula.append(f"Amount (rounded to cents): {total:,.2f}")
        return self._result(LineItemStatus.CHARGED, amount=total, adjustments=applied)

    def _result(
        self,
        status: LineItemStatus,
        *,
        amount: Decimal | None = None,
        reason: str | None = None,
        adjustments: list[AppliedAdjustment] | None = None,
        citations: list[Citation] | None = None,
    ) -> LineItem:
        return LineItem(
            charge_id=self.rule.charge_id,
            name=self.rule.name,
            section_refs=self.rule.section_refs,
            currency=self.rule.currency,
            status=status,
            amount=amount,
            formula=self.formula,
            adjustments=adjustments or [],
            assumptions=self.assumptions,
            reason=reason,
            citations=citations if citations is not None else list(self.rule.citations),
            notes=self.rule.notes,
        )

    def _notes_or(self, default: str) -> str:
        return " ".join(self.rule.notes) or default

    # -- components --------------------------------------------------------

    def _component(self, component: Component) -> Decimal:
        if isinstance(component, FixedFee):
            self.formula.append(f"{component.label}: {format_number(component.amount)}")
            return component.amount
        if isinstance(component, PerUnitFee):
            return self._per_unit(component)
        if isinstance(component, BandedFee):
            return self._banded(component)
        if isinstance(component, TieredFee):
            return self._tiered(component)
        raise RuleEvaluationError(f"Unknown component kind: {component!r}")  # pragma: no cover

    def _per_unit(self, component: PerUnitFee) -> Decimal:
        count, count_expression = self._units(component.units)
        amount = count * component.rate
        steps = _unless_trivial(count_expression, count)
        product = f"{format_number(count)} × {format_number(component.rate)}"
        if component.per_time is not None:
            periods, period_expression = self._units(component.per_time)
            amount *= periods
            steps.extend(_unless_trivial(period_expression, periods))
            product += f" × {format_number(periods)}"
        steps.append(f"{product} = {format_number(amount)}")
        self.formula.append(f"{component.label}: " + "; ".join(steps))
        return amount

    def _banded(self, component: BandedFee) -> Decimal:
        quantity = self._quantity(component.basis)
        for band in component.bands:
            if quantity < band.lower or (band.upper is not None and quantity > band.upper):
                continue
            upper = format_number(band.upper) if band.upper is not None else "∞"
            steps = [
                f"{component.basis.value} {format_number(quantity)} is in band "
                f"{format_number(band.lower)}–{upper}",
                f"base {format_number(band.base_fee)}",
            ]
            amount = band.base_fee
            if band.increment is not None:
                count, expression = self._units(band.increment.units)
                extra = count * band.increment.rate
                amount += extra
                steps.append(f"{expression} = {format_number(count)}")
                steps.append(
                    f"{format_number(count)} × {format_number(band.increment.rate)} = "
                    f"{format_number(extra)}"
                )
                steps.append(
                    f"{format_number(band.base_fee)} + {format_number(extra)} = "
                    f"{format_number(amount)}"
                )
            self.formula.append(f"{component.label}: " + "; ".join(steps))
            return amount
        raise RuleEvaluationError(
            f"Component '{component.id}': no band contains "
            f"{component.basis.value} = {format_number(quantity)}"
        )

    def _tiered(self, component: TieredFee) -> Decimal:
        quantity = self._quantity(component.basis)
        amount = Decimal(0)
        steps: list[str] = []
        previous_bound = Decimal(0)
        for tier in component.tiers:
            upper = quantity if tier.up_to is None else min(quantity, tier.up_to)
            slice_size = upper - previous_bound
            if slice_size <= 0:
                break
            count = _round_units(slice_size / tier.unit_size, tier.rounding)
            charge = count * tier.rate
            amount += charge
            steps.append(
                f"{format_number(slice_size)} at {format_number(tier.rate)} per "
                f"{format_number(tier.unit_size)}: {format_number(count)} × "
                f"{format_number(tier.rate)} = {format_number(charge)}"
            )
            if tier.up_to is None:
                break
            previous_bound = tier.up_to
        steps.append(f"total {format_number(amount)}")
        self.formula.append(f"{component.label}: " + "; ".join(steps))
        return amount

    def _units(self, units: Units) -> tuple[Decimal, str]:
        quantity = self._quantity(units.basis)
        measured = max(quantity - units.above, Decimal(0))
        count = _round_units(measured / units.unit_size, units.rounding)

        operand = format_number(quantity)
        if units.above:
            operand = f"({operand} − {format_number(units.above)})"
        if units.unit_size != 1:
            operand = f"{operand} / {format_number(units.unit_size)}"
        if units.rounding == Rounding.PRO_RATA:
            return count, operand
        return count, f"{units.rounding.value}({operand})"

    def _clamp(self, amount: Decimal) -> Decimal:
        rule = self.rule
        if rule.minimum is not None and amount < rule.minimum:
            self.formula.append(f"Minimum fee applies: {format_number(rule.minimum)}")
            return rule.minimum
        if rule.maximum is not None and amount > rule.maximum:
            self.formula.append(f"Maximum fee applies: {format_number(rule.maximum)}")
            return rule.maximum
        return amount

    # -- adjustments -------------------------------------------------------

    def _adjustments(
        self, amount: Decimal, component_amounts: Mapping[str, Decimal], factor: Decimal
    ) -> list[AppliedAdjustment]:
        candidates: list[tuple[Adjustment, Decimal]] = []
        for adjustment in self.rule.adjustments:
            if not self._all_hold(adjustment.when):
                continue
            if adjustment.applies_to == "all":
                base = amount
            else:
                base = factor * sum(
                    (component_amounts.get(component_id, Decimal(0)))
                    for component_id in adjustment.applies_to
                )
            effect = base * adjustment.percent / _HUNDRED
            if adjustment.kind == "reduction":
                effect = -effect
            candidates.append((adjustment, effect))

        best_in_group: dict[str, tuple[Adjustment, Decimal]] = {}
        chosen: list[tuple[Adjustment, Decimal]] = []
        for adjustment, effect in candidates:
            group = adjustment.exclusive_group
            if group is None:
                chosen.append((adjustment, effect))
            elif group not in best_in_group or abs(effect) > abs(best_in_group[group][1]):
                best_in_group[group] = (adjustment, effect)
        chosen.extend(best_in_group.values())
        chosen.sort(key=lambda item: self.rule.adjustments.index(item[0]))

        applied: list[AppliedAdjustment] = []
        for adjustment, effect in chosen:
            sign = "−" if effect < 0 else "+"
            self.formula.append(
                f"{adjustment.kind.capitalize()} {format_number(adjustment.percent)}% "
                f"({adjustment.description}): {sign}{format_number(abs(effect))}"
            )
            applied.append(
                AppliedAdjustment(
                    id=adjustment.id,
                    kind=adjustment.kind,
                    description=adjustment.description,
                    percent=adjustment.percent,
                    amount=effect,
                )
            )
        return applied

    # -- conditions, facts and quantities ------------------------------------

    def _all_hold(self, conditions: list[Condition]) -> bool:
        return all(self._holds(condition) for condition in conditions)

    def _holds(self, condition: Condition) -> bool:
        subject = self._subject(condition.fact)
        op = condition.op

        if isinstance(subject, bool):
            if op not in (Operator.EQ, Operator.NE):
                raise RuleEvaluationError(f"'{condition.fact}' is yes/no; it can't be '{op}'")
            expected = _as_bool(condition.value, condition.fact)
            return (subject == expected) if op == Operator.EQ else (subject != expected)

        if isinstance(subject, Decimal):
            if op == Operator.IN:
                assert isinstance(condition.value, list)
                return subject in {_as_number(item, condition.fact) for item in condition.value}
            target = _as_number(condition.value, condition.fact)
            return _compare(subject, op, target)

        text = subject.casefold()
        if op == Operator.IN:
            assert isinstance(condition.value, list)
            return text in {str(item).casefold() for item in condition.value}
        if op not in (Operator.EQ, Operator.NE):
            raise RuleEvaluationError(f"'{condition.fact}' is text; it can't be '{op}'")
        matches = text == str(condition.value).casefold()
        return matches if op == Operator.EQ else not matches

    def _subject(self, name: str) -> FactValue:
        if name in _BASIS_NAMES:
            return self._quantity(Basis(name))
        spec = self.rule.fact_spec(name)
        if spec is None:  # pragma: no cover - ChargeRule validation prevents this
            raise MissingFactError(name)
        if name in self.facts:
            return _coerce_fact(spec, self.facts[name])
        if spec.default_value is None:
            raise MissingFactError(name)
        value = _coerce_fact(spec, spec.default_value)
        self._assume(f"{spec.description}: assumed {_describe_value(value)}.")
        return value

    def _quantity(self, basis: Basis) -> Decimal:
        value = self.quantities.values.get(basis)
        if value is None:
            raise MissingQuantityError(basis)
        assumption = self.quantities.assumptions.get(basis)
        if assumption:
            self._assume(assumption)
        return value

    def _assume(self, text: str) -> None:
        if text not in self.assumptions:
            self.assumptions.append(text)

    def _describe(self, condition: Condition) -> str:
        spec = self.rule.fact_spec(condition.fact)
        subject = spec.description if spec else condition.fact
        value = condition.value
        rendered = ", ".join(value) if isinstance(value, list) else _describe_value(value)
        return f"{subject} {_OPERATOR_TEXT[condition.op]} {rendered}"


_OPERATOR_TEXT = {
    Operator.EQ: "=",
    Operator.NE: "≠",
    Operator.LT: "<",
    Operator.LE: "≤",
    Operator.GT: ">",
    Operator.GE: "≥",
    Operator.IN: "in",
}


def _unless_trivial(expression: str, value: Decimal) -> list[str]:
    """A formula step "expression = value", or nothing when the expression is
    just the value itself (a quantity used as-is)."""
    rendered = format_number(value)
    return [] if expression == rendered else [f"{expression} = {rendered}"]


def _round_units(value: Decimal, rounding: Rounding) -> Decimal:
    if rounding == Rounding.CEIL:
        return Decimal(math.ceil(value))
    if rounding == Rounding.FLOOR:
        return Decimal(math.floor(value))
    return value


def _compare(left: Decimal, op: Operator, right: Decimal) -> bool:
    if op == Operator.EQ:
        return left == right
    if op == Operator.NE:
        return left != right
    if op == Operator.LT:
        return left < right
    if op == Operator.LE:
        return left <= right
    if op == Operator.GT:
        return left > right
    return left >= right


def _coerce_fact(spec: FactSpec, value: FactValue | str) -> FactValue:
    if spec.type == "bool":
        return _as_bool(value, spec.name)
    if spec.type == "number":
        return _as_number(value, spec.name)
    return str(value)


def _as_bool(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().casefold() in ("true", "yes"):
        return True
    if isinstance(value, str) and value.strip().casefold() in ("false", "no"):
        return False
    raise RuleEvaluationError(f"'{name}' needs a yes/no value, got {value!r}")


def _as_number(value: object, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, str | int | Decimal):
        raise RuleEvaluationError(f"'{name}' needs a number, got {value!r}")
    try:
        return parse_number(value)
    except NumberFormatError as exc:
        raise RuleEvaluationError(f"'{name}' needs a number, got {value!r}") from exc


def _describe_value(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, Decimal):
        return format_number(value)
    return str(value)
