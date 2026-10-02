"""How the engine evaluates a ChargeRule, explained for the model.

Shared by the extraction and critique prompts, so the writer and the reviewer
of a rule agree on what it means. Mirrors app/rules/engine.py; keep the two
in step.
"""

from app.domain.vessel import BASIS_DESCRIPTIONS


def quantity_glossary() -> str:
    return "\n".join(
        f"- {basis.value}: {description}" for basis, description in BASIS_DESCRIPTIONS.items()
    )


RULE_SEMANTICS = """\
How the engine evaluates a ChargeRule:
- The engine supplies these quantities for every call. Conditions and units may use them \
directly; never declare them as facts:
$quantities
- A rule whose status is not "priced" reports that status. Otherwise the charge does not apply \
if all conditions of any exemption hold, or if any applies_when condition fails.
- If an unpriced component's conditions all hold, the charge is reported as not priced for the \
call (its label says why). Otherwise unpriced components are skipped.
- Each other component whose "when" conditions all hold is evaluated: fixed = amount; \
per_unit = rate × units (× time units when per_time is set); banded = the first band in the \
list with lower <= quantity <= upper (bands may share a printed boundary; the earlier band wins), \
giving \
its base_fee plus its increment; tiered = each slice of the quantity at its own rate, a slice \
ending at its up_to, or width after the previous slice's end.
- Units = max(quantity - above - less, 0) / unit_size, rounded as the rounding says, where less \
(optional) names a number fact or quantity to deduct: "the time in port less the hours worked".
- The components are summed, clamped to minimum and maximum, then multiplied by the multiplier.
- Each adjustment whose conditions hold adds or subtracts its percentage of the components it \
targets (after the multiplier); within an exclusive_group only the largest applies.
- A fact not supplied for a call takes its default_value.
- There are no counts beyond these quantities: an extra that depends on information a vessel \
call doesn't normally include (how long a delay lasted, how many extra tugs were ordered, a \
cancellation) is either a component or adjustment that applies only when a yes/no fact says it \
happened (default: it didn't), or is left out and mentioned in notes."""
