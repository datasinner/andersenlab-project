"""The rule DSL: a typed vocabulary of tariff mechanics.

A ChargeRule describes one charge at one port: which pricing components it
has (fixed fees, rates per unit "or part thereof", tonnage bands, marginal
tiers, time pro-rata), when it applies, exemptions, and percentage
reductions/surcharges. The agent extracts rules from a tariff document into
this shape; app.rules.engine turns them into money. The DSL describes how
tariffs work in general; which rules and numbers apply comes from the
document.

Numbers are Decimals, accepted as strings exactly as printed ("12 345.67")
and serialized to JSON as strings, so no value ever passes through a float.
"""

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    WithJsonSchema,
    model_validator,
)

from app.domain.numbers import parse_number
from app.domain.vessel import Basis

RULE_SCHEMA_VERSION = 5


def _to_decimal(value: object) -> object:
    if isinstance(value, str):
        return parse_number(value)
    return value


Amount = Annotated[
    Decimal,
    BeforeValidator(_to_decimal),
    PlainSerializer(lambda value: str(value), return_type=str, when_used="json"),
    WithJsonSchema(
        {
            "type": "string",
            "description": "A decimal number copied exactly as printed, e.g. '12345.67'.",
        }
    ),
]
NonNegativeAmount = Annotated[Amount, Field(ge=0)]
PositiveAmount = Annotated[Amount, Field(gt=0)]
Identifier = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=100)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Rounding(StrEnum):
    CEIL = "ceil"  # round up to whole units: "per 100 tons or part thereof"
    FLOOR = "floor"  # only complete units count
    PRO_RATA = "pro_rata"  # exact fraction: "a part of a 24 hour period ... pro rata"


class Operator(StrEnum):
    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    IN = "in"


class Condition(_Model):
    """`fact` names either a fact declared in the rule's `facts` or a Basis
    quantity (e.g. time_in_port_hours). Numbers in `value` are strings."""

    fact: Identifier
    op: Operator
    value: bool | str | list[str]

    @model_validator(mode="after")
    def _value_matches_operator(self) -> "Condition":
        if self.op == Operator.IN and not isinstance(self.value, list):
            raise ValueError(f"condition on '{self.fact}': 'in' needs a list value")
        if self.op != Operator.IN and isinstance(self.value, list):
            raise ValueError(f"condition on '{self.fact}': only 'in' takes a list value")
        return self


class Citation(_Model):
    chunk_id: int
    section_ref: str
    page: int
    quote: str = Field(min_length=1, description="Verbatim text from the cited chunk.")


class Units(_Model):
    """How many billable units of a quantity: max(quantity - above - less, 0)
    / unit_size, rounded."""

    basis: Basis
    unit_size: PositiveAmount = Decimal(1)
    rounding: Rounding
    above: NonNegativeAmount = Decimal(0)
    # A numeric fact or quantity to deduct as well: "the time in port less
    # the hours worked".
    less: Identifier | None = None


class _Component(_Model):
    id: Identifier
    label: str = Field(min_length=1)
    # The component only counts when all of these hold (e.g. a different
    # rate for vessels at their registered port).
    when: list[Condition] = Field(default_factory=list)


class FixedFee(_Component):
    kind: Literal["fixed"]
    amount: NonNegativeAmount


class PerUnitFee(_Component):
    """rate × units, optionally × time units (per 100 tons per 24 hours)."""

    kind: Literal["per_unit"]
    rate: NonNegativeAmount
    units: Units
    per_time: Units | None = None


class Increment(_Model):
    rate: NonNegativeAmount
    units: Units


class Band(_Model):
    """Applies when lower <= quantity <= upper (upper None = unbounded)."""

    lower: NonNegativeAmount
    upper: NonNegativeAmount | None
    base_fee: NonNegativeAmount
    increment: Increment | None = None


class BandedFee(_Component):
    """Exactly one band applies: the first, in listed order, containing the
    quantity."""

    kind: Literal["banded"]
    basis: Basis
    bands: list[Band] = Field(min_length=1)

    @model_validator(mode="after")
    def _bands_are_ordered(self) -> "BandedFee":
        for band in self.bands:
            if band.upper is not None and band.upper < band.lower:
                raise ValueError(f"component '{self.id}': band upper is below its lower bound")
        lowers = [band.lower for band in self.bands]
        if lowers != sorted(lowers):
            raise ValueError(f"component '{self.id}': bands must be listed by ascending lower")
        return self


class Tier(_Model):
    """Rate for one slice of the quantity. The slice ends at `up_to` when the
    tariff prints the bound ("up to 35 300 tons"), or `width` after the
    previous one when it prints the slice's size ("the following 90 days").
    Neither means the rest (only the last tier)."""

    up_to: NonNegativeAmount | None
    width: PositiveAmount | None = None
    rate: NonNegativeAmount
    unit_size: PositiveAmount = Decimal(1)
    rounding: Rounding

    @model_validator(mode="after")
    def _one_bound(self) -> "Tier":
        if self.up_to is not None and self.width is not None:
            raise ValueError("a tier gives either up_to or width, not both")
        return self

    @property
    def bounded(self) -> bool:
        return self.up_to is not None or self.width is not None


class TieredFee(_Component):
    """Marginal tiers: each slice of the quantity is charged at its own
    rate ("first 10 000 tons at X, the following 15 000 tons at Y")."""

    kind: Literal["tiered"]
    basis: Basis
    tiers: list[Tier] = Field(min_length=1)

    @model_validator(mode="after")
    def _tiers_are_ordered(self) -> "TieredFee":
        if any(not tier.bounded for tier in self.tiers[:-1]):
            raise ValueError(f"component '{self.id}': only the last tier may be unbounded")
        ends = self.tier_ends()
        finite = [end for end in ends if end is not None]
        if finite != sorted(finite) or len(set(finite)) != len(finite):
            raise ValueError(f"component '{self.id}': tier bounds must strictly increase")
        return self

    def tier_ends(self) -> list[Decimal | None]:
        """Where each tier's slice ends (None = unbounded)."""
        ends: list[Decimal | None] = []
        previous = Decimal(0)
        for tier in self.tiers:
            if tier.up_to is not None:
                end: Decimal | None = tier.up_to
            elif tier.width is not None:
                end = previous + tier.width
            else:
                end = None
            ends.append(end)
            if end is not None:
                previous = end
        return ends


class UnpricedCase(_Component):
    """A case the tariff leaves unpriced ("payable in terms of a special
    agreement", "on application") within a charge that is otherwise priced.
    When its conditions hold, the charge is reported as not priced for the
    call instead of being priced by the other components."""

    kind: Literal["unpriced"]
    when: list[Condition] = Field(min_length=1)


Component = Annotated[
    FixedFee | PerUnitFee | BandedFee | TieredFee | UnpricedCase, Field(discriminator="kind")
]


class Adjustment(_Model):
    id: Identifier
    kind: Literal["reduction", "surcharge"]
    description: str = Field(min_length=1)
    percent: PositiveAmount
    # Component ids the percentage applies to, or "all" for the whole charge.
    applies_to: list[Identifier] | Literal["all"] = "all"
    when: list[Condition] = Field(min_length=1)
    # Within one group only the largest applicable adjustment counts
    # ("not enjoyed in addition to ...").
    exclusive_group: Identifier | None = None
    citation: Citation | None = None


class Exemption(_Model):
    description: str = Field(min_length=1)
    when: list[Condition] = Field(min_length=1)
    citation: Citation | None = None


class FactSpec(_Model):
    name: Identifier
    type: Literal["bool", "number", "text"]
    description: str = Field(min_length=1, description="What the fact means, from the tariff.")
    # Value assumed when the vessel call doesn't settle the fact. Numbers
    # are strings. None means the fact must always be resolved.
    default_value: bool | str | None = None


class ChargeRule(_Model):
    charge_id: Identifier
    name: str = Field(min_length=1)
    section_refs: list[str] = Field(min_length=1)
    port_key: str = Field(min_length=1)
    currency: str = Field(min_length=3, max_length=3)
    payer: Literal["vessel", "cargo", "other"]
    # not_applicable_at_port: the document defines the charge only for other ports.
    status: Literal["priced", "on_application", "not_priced_in_document", "not_applicable_at_port"]
    applies_when: list[Condition] = Field(default_factory=list)
    exemptions: list[Exemption] = Field(default_factory=list)
    components: list[Component] = Field(default_factory=list)
    minimum: NonNegativeAmount | None = None
    maximum: NonNegativeAmount | None = None
    multiplier: Units | None = None
    adjustments: list[Adjustment] = Field(default_factory=list)
    facts: list[FactSpec] = Field(default_factory=list)
    citations: list[Citation] = Field(min_length=1)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _is_consistent(self) -> "ChargeRule":
        if self.status == "priced" and not self.components:
            raise ValueError("a priced rule needs at least one component")
        if self.minimum is not None and self.maximum is not None and self.maximum < self.minimum:
            raise ValueError("maximum is below minimum")

        component_ids = [component.id for component in self.components]
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("component ids must be unique")
        fact_names = [fact.name for fact in self.facts]
        if len(set(fact_names)) != len(fact_names):
            raise ValueError("fact names must be unique")
        if set(fact_names) & {basis.value for basis in Basis}:
            raise ValueError("a fact may not reuse the name of a Basis quantity")

        for adjustment in self.adjustments:
            if adjustment.applies_to == "all":
                continue
            unknown = set(adjustment.applies_to) - set(component_ids)
            if unknown:
                raise ValueError(
                    f"adjustment '{adjustment.id}' targets unknown components: {sorted(unknown)}"
                )

        known_subjects = set(fact_names) | {basis.value for basis in Basis}
        for condition in self.all_conditions():
            if condition.fact not in known_subjects:
                raise ValueError(
                    f"condition refers to '{condition.fact}', which is neither a declared fact "
                    "nor a Basis quantity"
                )

        numeric = {fact.name for fact in self.facts if fact.type == "number"}
        numeric |= {basis.value for basis in Basis}
        for units in self.all_units():
            if units.less is not None and units.less not in numeric:
                raise ValueError(
                    f"units deduct '{units.less}', which is neither a number fact nor a Basis "
                    "quantity"
                )
        return self

    def all_conditions(self) -> list[Condition]:
        conditions = list(self.applies_when)
        for exemption in self.exemptions:
            conditions.extend(exemption.when)
        for component in self.components:
            conditions.extend(component.when)
        for adjustment in self.adjustments:
            conditions.extend(adjustment.when)
        return conditions

    def all_units(self) -> list[Units]:
        found = [self.multiplier] if self.multiplier else []
        for component in self.components:
            if isinstance(component, PerUnitFee):
                found.append(component.units)
                if component.per_time is not None:
                    found.append(component.per_time)
            elif isinstance(component, BandedFee):
                found.extend(band.increment.units for band in component.bands if band.increment)
        return found

    def fact_spec(self, name: str) -> FactSpec | None:
        for fact in self.facts:
            if fact.name == name:
                return fact
        return None
